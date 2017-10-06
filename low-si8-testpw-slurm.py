#!/usr/bin/env python
#from aiida.backends.utils import load_dbenv
#load_dbenv()
codename = 'QE5.4@daint-gpu'
from aiida.orm import Code
code = Code.get_from_string(codename)
from aiida.orm import DataFactory
StructureData = DataFactory('structure')
alat = 5.4 # angstrom
cell = [[alat, 0.10, 0.20,],
        [0.30, alat, 0.40,],
        [0.50, 0.60, alat,],
       ]
s = StructureData(cell=cell)
s.append_atom(position=(0.,0.,0.),symbols='Si')
s.append_atom(position=(alat/2.,alat/2.,0.),symbols='Si')
s.append_atom(position=(alat/2.,0.,alat/2.),symbols='Si')
s.append_atom(position=(0.,alat/2.,alat/2.),symbols='Si')
s.append_atom(position=(alat/4.,alat/4.,alat/4.),symbols='Si')
s.append_atom(position=(3.*alat/4.,3.*alat/4.,alat/4.),symbols='Si')
s.append_atom(position=(3.*alat/4.,alat/4.,3.*alat/4.),symbols='Si')
s.append_atom(position=(alat/4.,3.*alat/4.,3.*alat/4.),symbols='Si')
ParameterData = DataFactory('parameter')
parameters = ParameterData(dict={
          'CONTROL': {
              'calculation': 'relax',
              'restart_mode': 'from_scratch',
              'wf_collect': True,
              'tprnfor': True,
	      'tstress': True,
              },
          'SYSTEM': {
              'ecutwfc': 30.,
              'ecutrho': 240.,
              },
          'ELECTRONS': {
              'conv_thr': 1.e-6,
              }})
KpointsData = DataFactory('array.kpoints')
kpoints = KpointsData()
kpoints.set_kpoints_mesh([4,4,4])
calc = code.new_calc()
calc.set_max_wallclock_seconds(30*60) # 30 min
calc.set_resources({"num_machines": 1})
calc.use_structure(s)
calc.use_code(code)
calc.use_parameters(parameters)
calc.use_kpoints(kpoints)
calc.use_pseudos_from_family('SSSP_eff_PBE')
calc.set_custom_scheduler_commands("#SBATCH --constraint=gpu")
calc.label = "1st test"
calc.store_all()
print "created calculation; with uuid='{}' and PK={}".format(calc.uuid,calc.pk)
calc.submit()
