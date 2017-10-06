#from aiida.backends.utils import load_dbenv
#load_dbenv()

from aiida.orm import DataFactory

bohr_to_ang = 0.52917720859

StructureData = DataFactory('structure')

alat = 5.026641*bohr_to_ang # angstrom
cell = [[0., alat*0.697090, alat*0.716984,],
        [alat*0.756772, 0., alat*0.756772,],
        [alat*0.756772, alat*0.756772, 0.,],
       ]

# BaTiO3 cubic structure
s = StructureData(cell=cell)
s.append_atom(position=(0.,0.,0.),symbols='Li')
s.append_atom(position=(alat*0.7567718,alat*0.7269308,alat*0.7368778),symbols='F')

s.store()

print "created structure with uuid='{}' and PK={}".format(s.uuid,s.pk)




