# -*- coding: utf-8 -*-
from __future__ import division
import aiida.common
from aiida.common import aiidalogger
from aiida.orm.workflow import Workflow
from aiida.orm import Code, Computer
from aiida.orm import CalculationFactory, DataFactory

__copyright__ = u"Copyright (c), This file is part of the AiiDA platform. For further information please visit http://www.aiida.net/. All rights reserved."
__license__ = "MIT license, see LICENSE.txt file."
__version__ = "0.7.0"
__authors__ = "The AiiDA team."

LapwbasisData = DataFactory('lapwbasis')
ParameterData = DataFactory('parameter')
KpointsData = DataFactory('array.kpoints')
StructureData = DataFactory('structure')

logger = aiidalogger.getChild('WorkflowXTiO3')

## ===============================================
##    WorkflowXTiO3_LAPW_STRESS_TENSOR
## ===============================================

class WorkflowXTiO3_LAPW_STRESS_TENSOR(Workflow):
    def __init__(self, **kwargs):

        super(WorkflowXTiO3_LAPW_STRESS_TENSOR, self).__init__(**kwargs)

    ## ===============================================
    ##    Structure generators
    ## ===============================================

    def get_structure(self, alat=4, x_material='Ba'):

       # cell = [[alat, 0., 0., ],
       #         [0., alat, 0., ],
       #         [0., 0., alat, ],
       # ]

       # # BaTiO3 cubic structure
       # s = StructureData(cell=cell)
       # s.append_atom(position=(0., 0., 0.), symbols=x_material)
       # s.append_atom(position=(alat / 2., alat / 2., alat / 2.), symbols=['Ti'])
       # s.append_atom(position=(alat / 2., alat / 2., 0.), symbols=['O'])
       # s.append_atom(position=(alat / 2., 0., alat / 2.), symbols=['O'])
       # s.append_atom(position=(0., alat / 2., alat / 2.), symbols=['O'])
       # s.store()

        s = load_node(890)
        s.store()

        return s

    def get_lapw_parameters(self):

        parameters = ParameterData(dict={
            'groundstate': {
                'xctype': 'GGA_PBE',
                'gmaxvr': '12.0',
            }}).store()

        return parameters

    def get_kpoints(self):

        kpoints = KpointsData()
        kpoints.set_kpoints_mesh([4, 4, 4])
        kpoints.store()

        return kpoints

    ## ===============================================
    ##    Calculations generators
    ## ===============================================

    def get_lapw_calculation(self, lapw_structure, lapw_parameters, lapw_kpoint):

        params = self.get_parameters()

        lapw_codename = params['lapw_codename']
        num_machines = params['num_machines']
        max_wallclock_seconds = params['max_wallclock_seconds']
        lapwbasis_family = params['lapwbasis_family']

        code = Code.get_from_string(lapw_codename)
        computer = code.get_remote_computer()

        LAPWCalc = CalculationFactory('exciting.exciting')

        calc = LAPWCalc(computer=computer)
        calc.set_max_wallclock_seconds(max_wallclock_seconds)
        calc.set_resources({"num_machines": num_machines})
        calc.store()

        calc.use_code(code)

        calc.use_structure(lapw_structure)
        calc.use_lapwbasis_from_family(lapwbasis_family)
        calc.use_parameters(lapw_parameters)
        calc.use_kpoints(lapw_kpoint)

        return calc

    ## ===============================================
    ##    Wf steps
    ## ===============================================

    @Workflow.step
    def start(self):

        params = self.get_parameters()
        x_material = params['x_material']

        self.append_to_report(x_material + "Ti03 STRESS TENSOR started")
        self.next(self.stress_tensor)

    @Workflow.step
    def stress_tensor(self):

        from aiida.orm import Code, Computer, CalculationFactory
        import numpy as np

        params = self.get_parameters()

        x_material = params['x_material']
        starting_alat = params['starting_alat']
        alat_steps = params['alat_steps']

        eps = np.linspace(-0.01, 0.01, alat_steps)
        eps = eps[eps.nonzero()]
        a_sweep = starting_alat * (np.ones((alat_steps-1,)) + eps)

        aiidalogger.info("Storing a_sweep as " + str(a_sweep))
        self.add_attribute('a_sweep', a_sweep)

        for a in a_sweep:
            self.append_to_report("Preparing structure {0} with alat {1}".format(x_material + "TiO3", a))

            calc = self.get_lapw_calculation(self.get_structure(alat=a, x_material=x_material),
                                           self.get_lapw_parameters(),
                                           self.get_kpoints())

            self.attach_calculation(calc)

        self.next(self.analyze)

    @Workflow.step
    def analyze(self):

        from aiida.orm.data.parameter import ParameterData
        import numpy as np

        x_material = self.get_parameter("x_material")
        a_sweep = self.get_attribute("a_sweep")

        aiidalogger.info("Retrieving a_sweep as {0}".format(a_sweep))

        # Get calculations
        start_calcs = self.get_step_calculations(self.stress_tensor)  # .get_calculations()

        # Calculate results
        #-----------------------------------------
        s_calcs = eps

        e_calcs = [c.res.energy for c in start_calcs]
        v_calcs = [c.res.volume for c in start_calcs]

        e_calcs = zip(*sorted(zip(a_sweep, e_calcs)))[1]
        v_calcs = zip(*sorted(zip(a_sweep, v_calcs)))[1]

        s_calcs = zip(*sorted(zip(a_sweep, s_calcs)))[1]

        #  Add to report
        #-----------------------------------------
        for i in range(len(a_sweep)):
            self.append_to_report(x_material + "Ti03 simulated with a=" + str(a_sweep[i]) + ", s=" + str(s_calcs[i]) + ", e=" + str(e_calcs[i]))

        s_pymatgen = s.get_pymatgen()

        #import pymatgen as mg
        #from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

        #finder = SpacegroupAnalyzer(s_pymatgen)

        #SGN = finder.get_space_group_number()

        SGN = s_pymatgen.get_space_group_info()[1]

        import spglib

        slatt = s.cell
        spos = np.squeeze(np.asarray(list(np.matrix(s.cell).T.I * np.matrix(x.position).T for x in s.sites)))
        snum = [1,] * len(s.sites)
        snum = np.ones(len(s.sites))
        scell = (slatt, spos, snum)
        SGN = int(spglib.get_symmetry_dataset(scell)["number"])


#%!%!%--- Classify the Space-Group Number ---%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!
        if (1 <= SGN and SGN <= 2):      # Triclinic
            LC = 'N'
            SCs= 6
        
        elif(3 <= SGN and SGN <= 15):    # Monoclinic
            LC = 'M'
            SCs= 4
        
        elif(16 <= SGN and SGN <= 74):   # Orthorhombic
            LC = 'O'
            SCs=  3
        
        elif(75 <= SGN and SGN <= 88):   # Tetragonal II
            LC = 'TII'
            SCs=  2
          
        elif(89 <= SGN and SGN <= 142):  # Tetragonal I
            LC = 'TI'
            SCs=  2
            
        elif(143 <= SGN and SGN <= 148): # Rhombohedral II 
            LC = 'RII'
            SCs=  2
            
        elif(149 <= SGN and SGN <= 167): # Rhombohedral I
            LC = 'RI'
            SCs=  2
            
        elif(168 <= SGN and SGN <= 176): # Hexagonal II
            LC = 'HII'
            SCs=  2
            
        elif(177 <= SGN and SGN <= 194): # Hexagonal I
            LC = 'HI'
            SCs=  2
            
        elif(195 <= SGN and SGN <= 206): # Cubic II
            LC = 'CII'
            SCs=  1
            
        elif(207 <= SGN and SGN <= 230): # Cubic I
            LC = 'CI'
            SCs=  1
            
        else: sys.exit('\n ... Oops ERROR: WRONG Space-Group Number !?!?!?\n')


#%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%#
#%!%!% ------------ Calculating the first derivative and Cross-Validation Error ------------ %!%!%#
#%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%#

        A1 = []
        
        mdri   = 0.002
        ordri  = 1
        
        strain = []
        energy = []

        strain = copy.copy(s_calcs)
        energy = copy.copy(e_calcs)
        
        coeffs = np.polyfit(strain, energy, ordri)
        A1.append(coeffs[ordri-1])
        
        A1 = np.array(A1)
        if (len(A1) != SCs):
            sys.exit('\n ... Oops ERROR: The number of data is NOT equal to ' + \
            str(SCs)+'\n')
        
        S = zeros((3,3))
        
#%!%!%--- Cubic structures ---%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%
        if (LC == 'CI' or \
            LC == 'CII'):
            S[0,0] = A1[0]/3.
            S[1,1] = S[0,0]
            S[2,2] = S[0,0]
#--------------------------------------------------------------------------------------------------

#%!%!%--- Hexagonal, Rhombohedral, and Tetragonal structures ---%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%
        if (LC == 'HI' or \
            LC == 'HII'or \
            LC == 'RI' or \
            LC == 'RII'or \
            LC == 'TI' or \
            LC == 'TII'):
            S[0,0] = (A1[0] - 1.*A1[1])/3.
            S[1,1] = S[0,0]
            S[2,2] = (A1[0] + 2.*A1[1])/3.
#--------------------------------------------------------------------------------------------------

#%!%!%--- Orthorhombic structures ---%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!
        if (LC == 'O'):
            S[0,0] = (A1[0] - 2.*A1[1] - 2.*A1[2])/3. 
            S[1,1] = (A1[0] + 2.*A1[1])/3.
            S[2,2] = (A1[0] + 2.*A1[2])/3.
#--------------------------------------------------------------------------------------------------

#%!%!%--- Monoclinic structures ---%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!
        if (LC == 'M'):
            S[0,0] = (A1[0] - 2.*A1[1] - 2.*A1[2])/3. 
            S[1,1] = (A1[0] + 2.*A1[1])/3.
            S[2,2] = (A1[0] + 2.*A1[2])/3.
        
            if (unique_axis == 'a'): S[1,2] = A1[3]/2.
            if (unique_axis == 'b'): S[0,2] = A1[3]/2.
            if (unique_axis == 'c'): S[0,1] = A1[3]/2.
#--------------------------------------------------------------------------------------------------

#%!%!%--- Triclinic structures ---%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%
        if (LC == 'N'):
            S[0,0] = (A1[0] + 2.* A1[2])/3.,
            S[1,1] = (A1[0] + 2.* A1[1])/3.,
            S[2,2] = (A1[0] - 2.*(A1[1] + A1[2]))/3.,
            S[1,2] = A1[3]/2.,
            S[0,2] = A1[4]/2.,
            S[0,1] = A1[5]/2.
#--------------------------------------------------------------------------------------------------

#%!%!%--- Calculating the Pressure ---%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%

        for i in range(2):
            for j in range(i+1, 3):
                S[j,i] = S[i,j] 

        eV_over_ang3_toGPa = 160.21766208
        v0i = int((alat_steps + 1) / 2)
        V0 = v_calcs [v0i]
        S = S / V0 * eV_over_ang3_toGPa
        P = (S[0,0]+S[1,1]+S[2,2])/-3.*eV_over_ang3_toGPa
#--------------------------------------------------------------------------------------------------

        self.append_to_report(x_material + "Ti03 simulated with a=" + str(a_sweep[v0i]) + ", e=" + str(e_calcs[v0i]))

        self.append_to_report("stress tensor matrix = " + str(S[0,0]) + str(S[0,1]) + str(S[0,2]))
        self.append_to_report("stress tensor matrix = " + str(S[1,0]) + str(S[1,1]) + str(S[1,2]))
        self.append_to_report("stress tensor matrix = " + str(S[2,0]) + str(S[2,1]) + str(S[2,2]))

        self.append_to_report("external pressure in GPa = " + str(P) )

        self.next(self.exit)

        #  Find optimal alat
        #-----------------------------------------

#        murnpars, ier = Murnaghan_fit(e_calcs, v_calcs)
#
#        # New optimal alat
#        optimal_alat = murnpars[3] ** (1 / 3.0)
#        self.add_attribute('optimal_alat', optimal_alat)
#
#        #  Build last calculation
#        #-----------------------------------------
#
#        calc = self.get_lapw_calculation(self.get_structure(alat=optimal_alat, x_material=x_material),
#                                       self.get_lapw_parameters(),
#                                       self.get_kpoints())
#        self.attach_calculation(calc)
#
#        self.next(self.final_step)

#   @Workflow.step
#    def final_step(self):
#
#        from aiida.orm.data.parameter import ParameterData
#
#        x_material = self.get_parameter("x_material")
#        optimal_alat = self.get_attribute("optimal_alat")
#
#        opt_calc = self.get_step_calculations(self.optimize)[0]  # .get_calculations()[0]
#        opt_e = opt_calc.get_outputs(type=ParameterData)[0].get_dict()['energy']
#
#        self.append_to_report(x_material + "Ti03 optimal with a=" + str(optimal_alat) + ", e=" + str(opt_e))
#
#        self.add_result("scf_converged", opt_calc)
#
#        self.next(self.exit)
#
#
#def Murnaghan_fit(e, v):
#    import pylab
#    import numpy as np
#    import scipy.optimize as optimize
#
#    e = np.array(e)
#    v = np.array(v)
#
#    # Fit with parabola for first guess
#    a, b, c = pylab.polyfit(v, e, 2)
#
#    # Initial parameters
#    v0 = -b / (2 * a)
#    e0 = a * v0 ** 2 + b * v0 + c
#    b0 = 2 * a * v0
#    bP = 4
#
#    def Murnaghan(vol, parameters):
#        E0 = parameters[0]
#        B0 = parameters[1]
#        BP = parameters[2]
#        V0 = parameters[3]
#
#        EM = E0 + B0 * vol / BP * ( ((V0 / vol) ** BP) / (BP - 1) + 1 ) - V0 * B0 / (BP - 1.0)
#
#        return EM
#
#    # Minimization function
#    def residuals(pars, y, x):
#        # we will minimize this function
#        err = y - Murnaghan(x, pars)
#        return err
#
#    p0 = [e0, b0, bP, v0]
#
#    return optimize.leastsq(residuals, p0, args=(e, v))
