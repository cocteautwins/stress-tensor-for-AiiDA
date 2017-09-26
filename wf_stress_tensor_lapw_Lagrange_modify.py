# -*- coding: utf-8 -*-
from __future__ import division
import aiida.common
from aiida.common import aiidalogger
from aiida.orm.workflow import Workflow
from aiida.orm import Code, Computer
from aiida.orm import CalculationFactory, DataFactory
from aiida.orm.utils import load_node

__copyright__ = u"Copyright (c), This file is part of the AiiDA platform. For further information please visit http://www.aiida.net/. All rights reserved."
__license__ = "MIT license, see LICENSE.txt file."
__version__ = "0.7.0"
__authors__ = "The AiiDA team."

LapwbasisData = DataFactory('lapwbasis')
ParameterData = DataFactory('parameter')
KpointsData = DataFactory('array.kpoints')
StructureData = DataFactory('structure')

logger = aiidalogger.getChild('WorkflowStressTensor')

## ===============================================
##    Workflow_LAPW_Stress_Tensor
## ===============================================

class Workflow_LAPW_stress_tensor_MLagrange(Workflow):
    def __init__(self, **kwargs):

        super(Workflow_LAPW_stress_tensor_MLagrange, self).__init__(**kwargs)

    ## ===============================================
    ##    Structure generators
    ## ===============================================

    def get_distorted_structure(self, structure_id, M_eps):

        import numpy as np

        s0 = load_node(structure_id)

        distorted_cell = np.dot(s0.cell, M_eps)

        s = StructureData(cell=distorted_cell)
        
        for site in s0.sites:
            kind_name = site.kind_name
            frac_coor = np.squeeze(np.asarray(list(np.matrix(s0.cell).T.I * np.matrix(site.position).T)))
            distorted_position = np.squeeze(np.asarray(list(np.matrix(s.cell).T * np.matrix(frac_coor).T)))
            s.append_atom(position=distorted_position, symbols=kind_name)

        s.store()

        return s

    def get_Lagrange_distorted_structure(self, structure_id, M_MLagrange_eps):

        import numpy as np

        s0 = load_node(structure_id)

        one = np.identity(3)

        #deform = (np.dot(M_Lagrange_eps.T, M_Lagrange_eps) - one) / 2.

        distorted_cell = np.transpose(np.dot(M_MLagrange_eps, np.transpose(s0.cell)))

        s = StructureData(cell=np.asarray(distorted_cell))

        for site in s0.sites:
            kind_name = site.kind_name
            frac_coor = np.squeeze(np.asarray(list(np.matrix(s0.cell).T.I * np.matrix(site.position).T)))
            distorted_position = np.squeeze(np.asarray(list(np.matrix(s.cell).T * np.matrix(frac_coor).T)))
            s.append_atom(position=distorted_position, symbols=kind_name)

        s.store()

        return s

    def get_distorted_index(self, structure_id, LC):

        import numpy as np

        s0 = load_node(structure_id)

        if (LC == 'CI' or \
            LC == 'CII'):
            def_list = ['1']

        if (LC == 'HI' or \
            LC == 'HII'or \
            LC == 'RI' or \
            LC == 'RII'or \
            LC == 'TI' or \
            LC == 'TII'):
            def_list = ['1','3']

        if (LC == 'O'):
            def_list = ['1','2','3']

        if (LC == 'M'):
            if (s0.cell_angles[0] != 90): unique_axis == 'a'
            if (s0.cell_angles[1] != 90): unique_axis == 'b'
            if (s0.cell_angles[2] != 90): unique_axis == 'c'

            if (unique_axis == 'a'): def_list = ['1','2','3','4']
            if (unique_axis == 'b'): def_list = ['1','2','3','5']
            if (unique_axis == 'c'): def_list = ['1','2','3','6']

        if (LC == 'N'):
            def_list = ['1','2','3','4','5','6']

        return def_list

    def def_strain_dict(self, def_mtx_index='1'):

        def_str_dic = {             \
        '1':
        '\n[ 1+eps  0      0     ]' \
        '\n[ 0      1+eps  0     ]' \
        '\n[ 0      0      1+eps ]',\
        
        '2':
        '\n[(1+eps)^-.5   0           0          ]' \
        '\n[ 0           (1+eps)^+1.  0          ]' \
        '\n[ 0            0          (1+eps)^-.5 ]',\
        
        '3':
        '\n[(1+eps)^-.5   0           0          ]' \
        '\n[ 0           (1+eps)^-.5  0          ]' \
        '\n[ 0            0          (1+eps)^+1. ]',\
        
        '4':
        '\n[ 1/(1-eps^2)  0           0          ]' \
        '\n[ 0            1          eps         ]' \
        '\n[ 0           eps          1          ]',\
        
        '5':
        '\n[ 1           0           eps         ]' \
        '\n[ 0           1/(1-eps^2)  0          ]' \
        '\n[eps          0            1          ]',\
        
        '6':
        '\n[ 1          eps           0          ]' \
        '\n[eps          1            0          ]' \
        '\n[ 0           0            1/(1-eps^2)]'}
        
        return def_str_dict[def_mtx_index]

    def get_strain_matrix(self, eps=0.0, def_mtx_index='1'):

        import numpy as np

        def_mtx_dic = {                                       \
        '1' : [[1.+eps      , 0.          , 0.             ],
               [0.          , 1.+eps      , 0.             ],
               [0.          , 0.          , 1+eps          ]],\

        '2' : [[(1+eps)**-.5, 0.          , 0.             ],
               [ 0.         , 1.+eps      , 0.             ],
               [ 0.         , 0.          ,(1+eps)**-.5    ]],\

        '3' : [[(1+eps)**-.5, 0.          , 0.             ],
               [ 0.         , (1+eps)**-.5, 0.             ],
               [ 0.         , 0.          , 1.+eps         ]],\

        '4' : [[1./(1-eps**2), 0.           , 0.           ],
               [ 0.          , 1.           ,eps           ],
               [ 0.          ,eps           , 1.           ]],\

        '5' : [[ 1.          , 0.           ,eps           ],
               [ 0.          , 1./(1-eps**2), 0.           ],
               [eps          , 0.           , 1.           ]],\

        '6' : [[ 1.          ,eps           , 0.           ],
               [eps          , 1.           , 0.           ],
               [ 0.          , 0.           , 1./(1-eps**2)]]}

        M_eps = np.array(def_mtx_dic[def_mtx_index])

        return M_eps


    def get_Lagrange_strain_matrix(self, eps=0.0, def_mtx_index='1'):

        import numpy as np

        def_mtx_dic = {                                       \
        '1' : [[1.+eps      , 0.          , 0.             ],
               [0.          , 1.          , 0.             ],
               [0.          , 0.          , 1.             ]],\

        '2' : [[ 1.         , 0.          , 0.             ],
               [ 0.         , 1.+eps      , 0.             ],
               [ 0.         , 0.          , 1.             ]],\

        '3' : [[ 1.         , 0.          , 0.             ],
               [ 0.         , 1.          , 0.             ],
               [ 0.         , 0.          , 1.+eps         ]],\

        '4' : [[ 1.          , eps          , 0.           ],
               [ 0.          , 1.           , 0.           ],
               [ 0.          , 0.           , 1.           ]],\

        '5' : [[ 1.          , 0.           , eps          ],
               [ 0.          , 1.           , 0.           ],
               [ 0.          , 0.           , 1.           ]],\

        '6' : [[ 1.          , 0.           , 0.           ],
               [ 0.          , 1.           , eps           ],
               [ 0.          , 0.           , 1.           ]]}

        M_Lagrange_eps = np.array(def_mtx_dic[def_mtx_index])

        return M_Lagrange_eps


    def get_MLagrange_strain_matrix(self, eps=0.0, def_mtx_index='1'):

        import numpy as np

        def_mtx_dic = {                                       \
        '1' : [[eps         , 0.          , 0.             ],
               [0.          , 0.          , 0.             ],
               [0.          , 0.          , 0.             ]],\

        '2' : [[ 0.         , 0.          , 0.             ],
               [ 0.         , eps         , 0.             ],
               [ 0.         , 0.          , 0.             ]],\

        '3' : [[ 0.         , 0.          , 0.             ],
               [ 0.         , 0.          , 0.             ],
               [ 0.         , 0.          , eps            ]],\

        '4' : [[ 0.          , 0.           , 0.           ],
               [ 0.          , 0.           , eps/2.       ],
               [ 0.          , eps/2.       , 0.           ]],\

        '5' : [[ 0.          , 0.           , eps/2.       ],
               [ 0.          , 0.           , 0.           ],
               [ eps/2.      , 0.           , 0.           ]],\

        '6' : [[ 0.          , eps/2.       , 0.           ],
               [ eps/2.      , 0.           , 0.           ],
               [ 0.          , 0.           , 0.           ]]}

        eta_matrix = np.mat(def_mtx_dic[def_mtx_index])

        one_matrix = np.mat([[ 1.0, 0.0, 0.0],
                             [ 0.0, 1.0, 0.0],
                             [ 0.0, 0.0, 1.0]]) 

        eps_matrix = eta_matrix

        x = eta_matrix + 0.5 * np.dot(eps_matrix, eps_matrix)
        eps_matrix = x

        M_MLagrange_eps = one_matrix + eps_matrix

        return M_MLagrange_eps


    ## ===============================================
    ##    Space group number for structure 
    ##    spglib is needed
    ## ===============================================

    def get_space_group_number(self, structure_id):

        import numpy as np
        import spglib

        s0 = load_node(structure_id)
        slatt = s0.cell
        spos = np.squeeze(np.asarray(list(np.matrix(s0.cell).T.I * np.matrix(x.position).T for x in s0.sites)))
        snum = np.ones(len(s0.sites))
        scell = (slatt, spos, snum)
        SGN = int(spglib.get_symmetry_dataset(scell)["number"])

        return SGN

    ## ===============================================
    ##    Laue dictionary and
    ##    Number of independent stress components for structure 
    ## ===============================================

    def get_Laue_dict(self, space_group_number):

        SGN = space_group_number

        if (1 <= SGN and SGN <= 2):      # Triclinic
            LC = 'N'
            # SCs= 6
        
        elif(3 <= SGN and SGN <= 15):    # Monoclinic
            LC = 'M'
            # SCs= 4
        
        elif(16 <= SGN and SGN <= 74):   # Orthorhombic
            LC = 'O'
            # SCs= 3
        
        elif(75 <= SGN and SGN <= 88):   # Tetragonal II
            LC = 'TII'
            # SCs= 2
        
        elif(89 <= SGN and SGN <= 142):  # Tetragonal I
            LC = 'TI'
            # SCs= 2
        
        elif(143 <= SGN and SGN <= 148): # Rhombohedral II 
            LC = 'RII'
            # SCs= 2
        
        elif(149 <= SGN and SGN <= 167): # Rhombohedral I
            LC = 'RI'
            # SCs= 2
        
        elif(168 <= SGN and SGN <= 176): # Hexagonal II
            LC = 'HII'
            # SCs= 2
        
        elif(177 <= SGN and SGN <= 194): # Hexagonal I
            LC = 'HI'
            # SCs= 2
        
        elif(195 <= SGN and SGN <= 206): # Cubic II
            LC = 'CII'
            # SCs= 1
        
        elif(207 <= SGN and SGN <= 230): # Cubic I
            LC = 'CI'
            # SCs= 1

        return LC

    def def_Laue_dict(self, LC):

        Laue_dic = {            \
        'CI' :'Cubic I'        ,\
        'CII':'Cubic II'       ,\
        'HI' :'Hexagonal I'    ,\
        'HII':'Hexagonal II'   ,\
        'RI' :'Rhombohedral I' ,\
        'RII':'Rhombohedral II',\
        'TI' :'Tetragonal I'   ,\
        'TII':'Tetragonal II'  ,\
        'O'  :'Orthorhombic'   ,\
        'M'  :'Monoclinic'     ,\
        'N'  :'Triclinic'}

        return Laue_dict[LC] 

    def get_stress_tensor_matrix(self, structure_id, LC, A1):

        import numpy as np

        s0 = load_node(structure_id)

        S = np.zeros((3,3))

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

            if (s0.cell_angles[0] != 90): unique_axis == 'a'
            if (s0.cell_angles[1] != 90): unique_axis == 'b'
            if (s0.cell_angles[2] != 90): unique_axis == 'c'

            if (unique_axis == 'a'): S[1,2] = A1[3]/2.
            if (unique_axis == 'b'): S[0,2] = A1[3]/2.
            if (unique_axis == 'c'): S[0,1] = A1[3]/2.
        #--------------------------------------------------------------------------------------------------

        #%!%!%--- Triclinic structures ---%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%
        if (LC == 'N'):
            S[0,0] = (A1[0] + 2.* A1[2])/3.
            S[1,1] = (A1[0] + 2.* A1[1])/3.
            S[2,2] = (A1[0] - 2.*(A1[1] + A1[2]))/3.
            S[1,2] = A1[3]/2.
            S[0,2] = A1[4]/2.
            S[0,1] = A1[5]/2.
        #--------------------------------------------------------------------------------------------------

        S[0,0] = A1[0]
        S[1,1] = A1[1]
        S[2,2] = A1[2]
        S[1,2] = A1[3]
        S[0,2] = A1[4]
        S[0,1] = A1[5]


        for i in range(2):
            for j in range(i+1, 3):
                S[j,i] = S[i,j]

        V0 = load_node(structure_id).get_cell_volume()
        S = S / V0

        return S

    def get_lapw_parameters(self):

        parameters = ParameterData(dict={
            'groundstate': {
                'xctype': 'GGA_PBE',
                'gmaxvr': '30.0',
                'rgkmax': '10.0',
                'nosym': 'true',
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
        structure_id = params['structure_id']

        x_material = load_node(structure_id).get_formula()

        self.append_to_report(x_material + " stress tensor started with structure ID " + str(structure_id))

        self.next(self.stress_tensor)

    @Workflow.step
    def stress_tensor(self):

        from aiida.orm import Code, Computer, CalculationFactory
        import numpy as np
        import spglib

        params = self.get_parameters()

        use_symmetry = True
        use_symmetry = params['use_symmetry']
        structure_id = params['structure_id']
        x_material = load_node(structure_id).get_formula()
        
        # alat_steps = params['alat_steps']

        alat_steps = 5
        eps = np.linspace(-0.008, 0.008, alat_steps).tolist()

        aiidalogger.info("Storing eps as " + str(eps))
        self.add_attribute('eps', eps)

        if use_symmetry:
           SGN = self.get_space_group_number(structure_id=structure_id)
        else:
           SGN = 1
        
        self.append_to_report(x_material + " structure has space group number " + str(SGN))

        LC = self.get_Laue_dict(space_group_number=SGN)
        self.add_attribute('LC', LC)

        # distorted_structure_index = range(len(eps * SCs))
        # aiidalogger.info("Storing distorted_structure_index as " + str(distorted_structure_index))
        # self.add_attribute('distorted_structure_index', distorted_structure_index)

        def_list = self.get_distorted_index(structure_id=structure_id, LC=LC)

        distorted_structure_index = []
        eps_index = 0
        for i in def_list:
            for a in eps:
                eps_index = eps_index + 1
                distorted_structure_index.append(eps_index) 

                # M_eps = self.get_strain_matrix(eps=a, def_mtx_index=i)

                M_MLagrange_eps = self.get_MLagrange_strain_matrix(eps=a, def_mtx_index=i)
                
                self.append_to_report("Preparing structure {0} with alat_strain {1} for SC {2}".format(x_material, a, i))

                calc = self.get_lapw_calculation(self.get_Lagrange_distorted_structure(structure_id=structure_id, M_MLagrange_eps=M_MLagrange_eps),
                                               self.get_lapw_parameters(),
                                               self.get_kpoints())

                self.attach_calculation(calc)

        self.add_attribute('distorted_structure_index', distorted_structure_index)

        self.next(self.analyze)

    @Workflow.step
    def analyze(self):

        from aiida.orm.data.parameter import ParameterData
        import numpy as np
        import scipy.optimize as scimin

        #%!%!%--- CONSTANTS ---%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!
        _e    =  1.602176565e-19         # elementary charge
        # Bohr  =  5.291772086e-11         # a.u. to meter
        # Ha2eV = 27.211396132             # Ha to eV
        # Tokbar= (_e*Ha2eV)/(1e8*Bohr**3) # Ha/[a.u.]^3 to kbar
        # ToGPa = (_e*Ha2eV)/(1e9*Bohr**3) # Ha/[a.u.]^3 to GPa
        angstrom = 1.0e-10               # angstrom to meter
        ToGPa = _e/(1e9*angstrom**3)     # eV/[\AA]^3 to GPa
        Tokbar = _e/(1e8*angstrom**3)    # eV/[\AA]^3 to kbar
        #__________________________________________________________________________________________________

        eps = self.get_attribute("eps")
        distorted_structure_index = self.get_attribute("distorted_structure_index")

        params = self.get_parameters()
        structure_id = params['structure_id']        
        x_material = load_node(structure_id).get_formula()

        aiidalogger.info("Retrieving eps as {0}".format(eps))

        # Get calculations/get_step_calculations
        start_calcs = self.get_step_calculations(self.stress_tensor)  # .get_calculations()

        # Calculate results
        #-----------------------------------------

        e_calcs = [c.res.energy for c in start_calcs]

        e_calcs = zip(*sorted(zip(distorted_structure_index, e_calcs)))[1]

        #  Add to report
        #-----------------------------------------
        for i in range(len(distorted_structure_index)):
            self.append_to_report(x_material + " simulated with strain =" + str(distorted_structure_index[i]) + ", e=" + str(e_calcs[i]))

        #  Obtain stress tensor by polyfit
        #-----------------------------------------
        alat_steps = 5
        SCs = int(len(distorted_structure_index) / alat_steps)
        A1 = []

        order = 3
        eps = np.linspace(-0.008, 0.008, alat_steps).tolist()

        for i in range(SCs)[::-1]:
            energy = e_calcs[i*alat_steps:(i+1)*alat_steps] 
            coeffs = np.polyfit(eps, energy, order)
            A1.append(coeffs[order-1]*ToGPa)
            
            #lsq_coeffs, ier = lsq_fit(energy, eps)
            #A1.append(lsq_coeffs[order-1]*ToGPa)

        A1 = np.array(A1)

        LC = self.get_attribute("LC")

        S = self.get_stress_tensor_matrix(structure_id=structure_id, LC=LC, A1=A1)

        self.append_to_report(x_material + " simulated with structure_id =" + str(structure_id) + " has stress tensor S11, S12, S13=" + str(S[0,:]))
        self.append_to_report(x_material + " simulated with structure_id =" + str(structure_id) + " has stress tensor S21, S22, S23=" + str(S[1,:]))
        self.append_to_report(x_material + " simulated with structure_id =" + str(structure_id) + " has stress tensor S31, S32, S33=" + str(S[2,:]))

        P = (S[0,0]+S[1,1]+S[2,2])/-3.

        self.append_to_report(x_material + " simulated with structure_id =" + str(structure_id) + " has pressure =" + str(P) + " GPa")

        self.next(self.exit)

    ## ===============================================
    ##    Least square fitting function
    ## ===============================================
def lsq_fit(e, v):
    import pylab
    import numpy as np
    import scipy.optimize as optimize

    e = np.array(e)
    v = np.array(v)

    # Fit with parabola for first guess
    # a, b, c, d = pylab.polyfit(v , e, 3)

    a = 1.
    b = 2.
    c = 3.
    d = 4.

    def fitfunc(strain, parameters):
        a = parameters[0]
        b = parameters[1]
        c = parameters[2]
        d = parameters[3]

        EM = d + c * strain + b * strain ** 2 + a * strain ** 3 

        return EM

    # Minimization function
    def residuals(pars, y, x): # array of residuals
        # we will minimize this function
        err = y - fitfunc(x, pars)
        return err

    p0 = [a, b, c, d]

    return optimize.leastsq(residuals, p0, args=(e, v))


