#from aiida.backends.utils import load_dbenv
#load_dbenv()

from aiida.orm import DataFactory

bohr_to_ang = 0.52917720859

StructureData = DataFactory('structure')

alat = 1.0 # angstrom
cell = [[3.0677637236, 0.0000000000, 0.0000000000,],
        [-1.5338818618, 2.6567613175, 0.0000000000,],
        [0.0000000000, 0.0000000000, 3.5090940000,],
       ]

# BaTiO3 cubic structure
s = StructureData(cell=cell)
s.append_atom(position=(-0.0017823509, 0.0039548708, 3.5000838822),symbols='Mg')
s.append_atom(position=(1.5394467539, 0.8872066829, 1.7626961684),symbols='B')
s.append_atom(position=(0.0045786882, 1.7799996426, 1.7556181597),symbols='B')


s.store()

print "created structure with uuid='{}' and PK={}".format(s.uuid,s.pk)




