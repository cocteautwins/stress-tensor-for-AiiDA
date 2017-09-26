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

logger = aiidalogger.getChild('WorkflowEOS')

## ===============================================
##    Workflow_LAPW_EOS
## ===============================================

class Workflow_LAPW_EOS(Workflow):
    def __init__(self, **kwargs):

        super(Workflow_LAPW_EOS, self).__init__(**kwargs)

    ## ===============================================
    ##    Structure generators
    ## ===============================================

    def get_structure(self, structure_id, scale=1.0):

        import numpy as np

        s0 = load_node(structure_id)

        scaled_cell = np.dot(s0.cell, scale).tolist()

        s = StructureData(cell=scaled_cell)

        for site in s0.sites:
            scaled_position = np.dot(list(site.position), scale)
            kind_name = site.kind_name
            s.append_atom(position=scaled_position, symbols=kind_name)

        # old_positions = [val for sublist in list(x.position for x in s0.sites) for val in sublist]
        # new_positions = [val for sublist in list(scaled_position) for val in sublist]

        # isite = 0
        # start_index = 0
        # end_index = 0

        # for isite in range(len(s0.sites))
        #     start_index = isite * 3
        #     end_index = (isite + 1) * 3
        #     s.sites[isite].position = new_positions[start_index, end_index]

        # cell = [[alat, 0., 0., ],
        #         [0., alat, 0., ],
        #         [0., 0., alat, ],
        # ]

        # BaTiO3 cubic structure
        # s = StructureData(cell=cell)
        # s.append_atom(position=(0., 0., 0.), symbols=x_material)
        # s.append_atom(position=(alat / 2., alat / 2., alat / 2.), symbols=['Ti'])
        # s.append_atom(position=(alat / 2., alat / 2., 0.), symbols=['O'])
        # s.append_atom(position=(alat / 2., 0., alat / 2.), symbols=['O'])
        # s.append_atom(position=(0., alat / 2., alat / 2.), symbols=['O'])

        s.store()

        return s

    def get_lapw_parameters(self):

        parameters = ParameterData(dict={
            'groundstate': {
                'xctype': 'GGA_PBE',
                'gmaxvr': '30.0',
                'rgkmax': '10.0',
            }}).store()

        return parameters

    def get_kpoints(self):

        kpoints = KpointsData()
        kpoints.set_kpoints_mesh([8, 8, 8])
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

        # s = self.get_structure(structure_pk, scale)
        x_material = load_node(structure_id).get_formula()

        # self.append_to_report("EOS started")
        self.append_to_report(x_material + " EOS started with structure ID " + str(structure_id))

        self.next(self.eos)

    @Workflow.step
    def eos(self):

        from aiida.orm import Code, Computer, CalculationFactory
        import numpy as np

        params = self.get_parameters()

        # x_material = params['x_material']
        # starting_alat = params['starting_alat']

        structure_id = params['structure_id']
        x_material = load_node(structure_id).get_formula()
        
        alat_steps = params['alat_steps']

        # a_sweep = np.linspace(starting_alat * 0.85, starting_alat * 1.15, alat_steps).tolist()

        alat_scale = np.linspace(0.98, 1.02, alat_steps).tolist()

        aiidalogger.info("Storing alat_scale as " + str(alat_scale))
        self.add_attribute('alat_scale', alat_scale)

        scale_index = []
        scale_i = 0
        for a in alat_scale:
            scale_i = scale_i + 1
            scale_index.append(scale_i)
            self.append_to_report("Preparing structure {0} with alat_scale {1}".format(x_material, a))

            calc = self.get_lapw_calculation(self.get_structure(structure_id=structure_id, scale=a),
                                           self.get_lapw_parameters(),
                                           self.get_kpoints())

            self.attach_calculation(calc)

        self.add_attribute('scale_index', scale_index)

        self.next(self.optimize)

    @Workflow.step
    def optimize(self):

        from aiida.orm.data.parameter import ParameterData

        alat_scale = self.get_attribute("alat_scale")

        scale_index = self.get_attribute("scale_index")

        params = self.get_parameters()
        structure_id = params['structure_id']
        x_material = load_node(structure_id).get_formula()

        aiidalogger.info("Retrieving alat_scale as {0}".format(alat_scale))

        # Get calculations/get_step_calculations
        start_calcs = self.get_step_calculations(self.eos)  # .get_calculations()

        # Calculate results
        #-----------------------------------------

        e_calcs = [c.res.energy for c in start_calcs]
        v_calcs = [c.res.volume for c in start_calcs]

        # e_calcs = zip(*sorted(zip(alat_scale, e_calcs)))[1]
        # v_calcs = zip(*sorted(zip(alat_scale, v_calcs)))[1]

        e_calcs = zip(*sorted(zip(scale_index, e_calcs)))[1]
        v_calcs = zip(*sorted(zip(scale_index, v_calcs)))[1]

        #  Add to report
        #-----------------------------------------
        for i in range(len(alat_scale)):
            self.append_to_report(x_material + " simulated with alat_scale=" + str(alat_scale[i]) + ", e=" + str(e_calcs[i]))

        #  Find optimal alat
        #-----------------------------------------

        murnpars, ier = Murnaghan_fit(e_calcs, v_calcs)

        # New optimal alat
        optimal_alat = murnpars[3] ** (1 / 3.0)
        self.add_attribute('optimal_alat', optimal_alat)
        vol0 = load_node(structure_id).get_cell_volume()
        optimal_scale = (murnpars[3] / vol0) ** (1 / 3.0)
        self.add_attribute('optimal_scale', optimal_scale)

        #  Build last calculation
        #-----------------------------------------

        calc = self.get_lapw_calculation(self.get_structure(structure_id=structure_id, scale=optimal_scale),
                                       self.get_lapw_parameters(),
                                       self.get_kpoints())
        self.attach_calculation(calc)

        self.next(self.final_step)

    @Workflow.step
    def final_step(self):

        from aiida.orm.data.parameter import ParameterData

        optimal_scale = self.get_attribute("optimal_scale")

        params = self.get_parameters()
        structure_id = params['structure_id']

        x_material = load_node(structure_id).get_formula()

        opt_calc = self.get_step_calculations(self.optimize)[0]  # .get_calculations()[0]
        opt_e = opt_calc.get_outputs(type=ParameterData)[0].get_dict()['energy']

        self.append_to_report(x_material + " optimal with alat_scale=" + str(optimal_scale) + ", e=" + str(opt_e))

        self.add_result("scf_converged", opt_calc)

        self.next(self.exit)


def Murnaghan_fit(e, v):
    import pylab
    import numpy as np
    import scipy.optimize as optimize

    e = np.array(e)
    v = np.array(v)

    # Fit with parabola for first guess
    a, b, c = pylab.polyfit(v, e, 2)

    # Initial parameters
    v0 = -b / (2 * a)
    e0 = a * v0 ** 2 + b * v0 + c
    b0 = 2 * a * v0
    bP = 4

    def Murnaghan(vol, parameters):
        E0 = parameters[0]
        B0 = parameters[1]
        BP = parameters[2]
        V0 = parameters[3]

        EM = E0 + B0 * vol / BP * ( ((V0 / vol) ** BP) / (BP - 1) + 1 ) - V0 * B0 / (BP - 1.0)

        return EM

    # Minimization function
    def residuals(pars, y, x):
        # we will minimize this function
        err = y - Murnaghan(x, pars)
        return err

    p0 = [e0, b0, bP, v0]

    return optimize.leastsq(residuals, p0, args=(e, v))
