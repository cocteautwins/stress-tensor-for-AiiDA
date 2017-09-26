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

UpfData = DataFactory('upf')
ParameterData = DataFactory('parameter')
KpointsData = DataFactory('array.kpoints')
StructureData = DataFactory('structure')

logger = aiidalogger.getChild('WorkflowElasticConstants')

## ===============================================
##    Workflow_PW_Elastic_Constants_Lagrange
## ===============================================

class Workflow_PW_elastic_constants_Lagrange(Workflow):
    def __init__(self, **kwargs):

        super(Workflow_PW_elastic_constants_Lagrange, self).__init__(**kwargs)

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

    def get_Lagrange_distorted_structure(self, structure_id, M_Lagrange_eps):

        import numpy as np

        s0 = load_node(structure_id)

        one = np.identity(3)

        deform = (np.dot(M_Lagrange_eps.T, M_Lagrange_eps) - one) / 2.

        #distorted_cell = np.dot((deform + one) , s0.cell)
        distorted_cell = np.dot(s0.cell, (deform + one))

        s = StructureData(cell=distorted_cell)

        for site in s0.sites:
            kind_name = site.kind_name
            frac_coor = np.squeeze(np.asarray(list(np.matrix(s0.cell).T.I * np.matrix(site.position).T)))
            distorted_position = np.squeeze(np.asarray(list(np.matrix(s.cell).T * np.matrix(frac_coor).T)))
            s.append_atom(position=distorted_position, symbols=kind_name)

        s.store()

        return s

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

    def get_Lagrange_distorted_index(self, structure_id, LC):

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

            if (unique_axis == 'a'): def_list = ['1','2','3','6']
            if (unique_axis == 'b'): def_list = ['1','2','3','5']
            if (unique_axis == 'c'): def_list = ['1','2','3','4']

        if (LC == 'N'):
            def_list = ['1','2','3','6','5','4']

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

    def def_EC_dict(self, LC):

        EC_dict = {                                                                     \
        'CI':'\
            for, space group-number between 207 and 230, Cubic I structure.        \n\n\
                       C11     C12     C12      0       0       0                  \n\
                       C12     C11     C12      0       0       0                  \n\
                       C12     C12     C11      0       0       0                  \n\
                        0       0       0      C44      0       0                  \n\
                        0       0       0       0      C44      0                  \n\
                        0       0       0       0       0      C44                 \n',\
        'CII':'\
            for, space group-number between 195 and 206, Cubic II structure.       \n\n\
                       C11     C12     C12      0       0       0                  \n\
                       C12     C11     C12      0       0       0                  \n\
                       C12     C12     C11      0       0       0                  \n\
                        0       0       0      C44      0       0                  \n\
                        0       0       0       0      C44      0                  \n\
                        0       0       0       0       0      C44                 \n',\
        'HI':'\
            for, space group-number between 177 and 194, Hexagonal I structure.    \n\n\
                       C11     C12     C13      0       0       0                  \n\
                       C12     C11     C13      0       0       0                  \n\
                       C13     C13     C33      0       0       0                  \n\
                        0       0       0      C44      0       0                  \n\
                        0       0       0       0      C44      0                  \n\
                        0       0       0       0       0   (C11-C12)/2            \n',\
        'HII':'\
            for, space group-number between 168 and 176, Hexagonal II structure.   \n\n\
                       C11     C12     C13      0       0       0                  \n\
                       C12     C11     C13      0       0       0                  \n\
                       C13     C13     C33      0       0       0                  \n\
                        0       0       0      C44      0       0                  \n\
                        0       0       0       0      C44      0                  \n\
                        0       0       0       0       0   (C11-C12)/2            \n',\
        'RI':'\
            for, space group-number between 149 and 167, Rhombohedral I structure. \n\n\
                       C11     C12     C13     C14      0       0                  \n\
                       C12     C11     C13    -C14      0       0                  \n\
                       C13     C13     C33      0       0       0                  \n\
                       C14    -C14      0      C44      0       0                  \n\
                        0       0       0       0      C44     C14                 \n\
                        0       0       0       0      C14  (C11-C12)/2            \n',\
        'RII':'\
            for, space group-number between 143 and 148, Rhombohedral II structure.\n\n\
                       C11     C12     C13     C14     C15      0                  \n\
                       C12     C11     C13    -C14    -C15      0                  \n\
                       C13     C13     C33      0       0       0                  \n\
                       C14    -C14      0      C44      0     -C15                 \n\
                       C15    -C15      0       0      C44     C14                 \n\
                        0       0       0     -C15     C14  (C11-C12)/2            \n',\
        'TI':'\
            for, space group-number between 89 and 142, Tetragonal I structure.    \n\n\
                       C11     C12     C13      0       0       0                  \n\
                       C12     C11     C13      0       0       0                  \n\
                       C13     C13     C33      0       0       0                  \n\
                        0       0       0      C44      0       0                  \n\
                        0       0       0       0      C44      0                  \n\
                        0       0       0       0       0      C66                 \n',\
        'TII':'\
            for, space group-number between 75 and 88, Tetragonal II structure.    \n\n\
                       C11     C12     C13      0       0      C16                 \n\
                       C12     C11     C13      0       0     -C16                 \n\
                       C13     C13     C33      0       0       0                  \n\
                        0       0       0      C44      0       0                  \n\
                        0       0       0       0      C44      0                  \n\
                       C16    -C16      0       0       0      C66                 \n',\
        'O':'\
            for, space group-number between 16 and 74, Orthorhombic structure.     \n\n\
                       C11     C12     C13      0       0       0                  \n\
                       C12     C22     C23      0       0       0                  \n\
                       C13     C23     C33      0       0       0                  \n\
                        0       0       0      C44      0       0                  \n\
                        0       0       0       0      C55      0                  \n\
                        0       0       0       0       0      C66                 \n',\
        'M':'\
            for, space group-number between 3 and 15, Monoclinic structure.        \n\n\
                       C11     C12     C13      0       0      C16                 \n\
                       C12     C22     C23      0       0      C26                 \n\
                       C13     C23     C33      0       0      C36                 \n\
                        0       0       0      C44     C45      0                  \n\
                        0       0       0      C45     C55      0                  \n\
                       C16     C26     C36      0       0      C66                 \n',\
        'N':'\
            for, space group-number between 1 and 2, Triclinic structure.          \n\n\
                       C11     C12     C13     C14      C15    C16                 \n\
                       C12     C22     C23     C24      C25    C26                 \n\
                       C13     C23     C33     C34      C35    C36                 \n\
                       C14     C24     C34     C44      C45    C46                 \n\
                       C15     C25     C35     C45      C55    C56                 \n\
                       C16     C26     C36     C46      C56    C66                 \n'}
        
        return EC_dict[LC]

    def def_Ls_dict(self, Ls_dict_index='01'):

        Ls_Dic={                       \
        '01':[ 1., 1., 1., 0., 0., 0.],\
        '02':[ 1., 0., 0., 0., 0., 0.],\
        '03':[ 0., 1., 0., 0., 0., 0.],\
        '04':[ 0., 0., 1., 0., 0., 0.],\
        '05':[ 0., 0., 0., 2., 0., 0.],\
        '06':[ 0., 0., 0., 0., 2., 0.],\
        '07':[ 0., 0., 0., 0., 0., 2.],\
        '08':[ 1., 1., 0., 0., 0., 0.],\
        '09':[ 1., 0., 1., 0., 0., 0.],\
        '10':[ 1., 0., 0., 2., 0., 0.],\
        '11':[ 1., 0., 0., 0., 2., 0.],\
        '12':[ 1., 0., 0., 0., 0., 2.],\
        '13':[ 0., 1., 1., 0., 0., 0.],\
        '14':[ 0., 1., 0., 2., 0., 0.],\
        '15':[ 0., 1., 0., 0., 2., 0.],\
        '16':[ 0., 1., 0., 0., 0., 2.],\
        '17':[ 0., 0., 1., 2., 0., 0.],\
        '18':[ 0., 0., 1., 0., 2., 0.],\
        '19':[ 0., 0., 1., 0., 0., 2.],\
        '20':[ 0., 0., 0., 2., 2., 0.],\
        '21':[ 0., 0., 0., 2., 0., 2.],\
        '22':[ 0., 0., 0., 0., 2., 2.],\
        '23':[ 0., 0., 0., 2., 2., 2.],\
        '24':[-1., .5, .5, 0., 0., 0.],\
        '25':[ .5,-1., .5, 0., 0., 0.],\
        '26':[ .5, .5,-1., 0., 0., 0.],\
        '27':[ 1.,-1., 0., 0., 0., 0.],\
        '28':[ 1.,-1., 0., 0., 0., 2.],\
        '29':[ 0., 1.,-1., 0., 0., 2.],\
        '30':[ .5, .5,-1., 0., 0., 2.],\
        '31':[ 1., 0., 0., 2., 2., 0.],\
        '32':[ 1., 1.,-1., 0., 0., 0.],\
        '33':[ 1., 1., 1.,-2.,-2.,-2.],\
        '34':[ .5, .5,-1., 2., 2., 2.],\
        '35':[ 0., 0., 0., 2., 2., 4.],\
        '36':[ 1., 2., 3., 4., 5., 6.],\
        '37':[-2., 1., 4.,-3., 6.,-5.],\
        '38':[ 3.,-5.,-1., 6., 2.,-4.],\
        '39':[-4.,-6., 5., 1.,-3., 2.],\
        '40':[ 5., 4., 6.,-2.,-1.,-3.],\
        '41':[-6., 3.,-2., 5.,-4., 1.]}
        
        return Ls_Dic[Ls_dict_index]


    def def_Ls_str(self, Ls_dict_index='01'):
        
        Ls_str={                                     \
        '01':'(  eta,  eta,  eta,  0.0,  0.0,  0.0)',\
        '02':'(  eta,  0.0,  0.0,  0.0,  0.0,  0.0)',\
        '03':'(  0.0,  eta,  0.0,  0.0,  0.0,  0.0)',\
        '04':'(  0.0,  0.0,  eta,  0.0,  0.0,  0.0)',\
        '05':'(  0.0,  0.0,  0.0, 2eta,  0.0,  0.0)',\
        '06':'(  0.0,  0.0,  0.0,  0.0, 2eta,  0.0)',\
        '07':'(  0.0,  0.0,  0.0,  0.0,  0.0, 2eta)',\
        '08':'(  eta,  eta,  0.0,  0.0,  0.0,  0.0)',\
        '09':'(  eta,  0.0,  eta,  0.0,  0.0,  0.0)',\
        '10':'(  eta,  0.0,  0.0, 2eta,  0.0,  0.0)',\
        '11':'(  eta,  0.0,  0.0,  0.0, 2eta,  0.0)',\
        '12':'(  eta,  0.0,  0.0,  0.0,  0.0, 2eta)',\
        '13':'(  0.0,  eta,  eta,  0.0,  0.0,  0.0)',\
        '14':'(  0.0,  eta,  0.0, 2eta,  0.0,  0.0)',\
        '15':'(  0.0,  eta,  0.0,  0.0, 2eta,  0.0)',\
        '16':'(  0.0,  eta,  0.0,  0.0,  0.0, 2eta)',\
        '17':'(  0.0,  0.0,  eta, 2eta,  0.0,  0.0)',\
        '18':'(  0.0,  0.0,  eta,  0.0, 2eta,  0.0)',\
        '19':'(  0.0,  0.0,  eta,  0.0,  0.0, 2eta)',\
        '20':'(  0.0,  0.0,  0.0, 2eta, 2eta,  0.0)',\
        '21':'(  0.0,  0.0,  0.0, 2eta,  0.0, 2eta)',\
        '22':'(  0.0,  0.0,  0.0,  0.0, 2eta, 2eta)',\
        '23':'(  0.0,  0.0,  0.0, 2eta, 2eta, 2eta)',\
        '24':'( -eta,.5eta,.5eta,  0.0,  0.0,  0.0)',\
        '25':'(.5eta, -eta,.5eta,  0.0,  0.0,  0.0)',\
        '26':'(.5eta,.5eta, -eta,  0.0,  0.0,  0.0)',\
        '27':'(  eta, -eta,  0.0,  0.0,  0.0,  0.0)',\
        '28':'(  eta, -eta,  0.0,  0.0,  0.0, 2eta)',\
        '29':'(  0.0,  eta, -eta,  0.0,  0.0, 2eta)',\
        '30':'(.5eta,.5eta, -eta,  0.0,  0.0, 2eta)',\
        '31':'(  eta,  0.0,  0.0, 2eta, 2eta,  0.0)',\
        '32':'(  eta,  eta, -eta,  0.0,  0.0,  0.0)',\
        '33':'(  eta,  eta,  eta,-2eta,-2eta,-2eta)',\
        '34':'(.5eta,.5eta, -eta, 2eta, 2eta, 2eta)',\
        '35':'(  0.0,  0.0,  0.0, 2eta, 2eta, 4eta)',\
        '36':'( 1eta, 2eta, 3eta, 4eta, 5eta, 6eta)',\
        '37':'(-2eta, 1eta, 4eta,-3eta, 6eta,-5eta)',\
        '38':'( 3eta,-5eta,-1eta, 6eta, 2eta,-4eta)',\
        '39':'(-4eta,-6eta, 5eta, 1eta,-3eta, 2eta)',\
        '40':'( 5eta, 4eta, 6eta,-2eta,-1eta,-3eta)',\
        '41':'(-6eta, 3eta,-2eta, 5eta,-4eta, 1eta)'}
        
        return Ls_str[Ls_dict_index]



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

    def get_LC_ECs(self, space_group_number, order=2):

        SGN = space_group_number

        if (1 <= SGN and SGN <= 2):      # Triclinic
            LC = 'N'
            if (order == 2): ECs = 21
            if (order == 3): ECs = 56
        
        elif(3 <= SGN and SGN <= 15):    # Monoclinic
            LC = 'M'
            if (order == 2): ECs = 13
            if (order == 3): ECs = 32
        
        elif(16 <= SGN and SGN <= 74):   # Orthorhombic
            LC = 'O'
            if (order == 2): ECs =  9
            if (order == 3): ECs = 20
        
        elif(75 <= SGN and SGN <= 88):   # Tetragonal II
            LC = 'TII'
            if (order == 2): ECs =  7
            if (order == 3): ECs = 16
        
        elif(89 <= SGN and SGN <= 142):  # Tetragonal I
            LC = 'TI'
            if (order == 2): ECs =  6
            if (order == 3): ECs = 12
        
        elif(143 <= SGN and SGN <= 148): # Rhombohedral II 
            LC = 'RII'
            if (order == 2): ECs =  7
            if (order == 3): ECs = 20
        
        elif(149 <= SGN and SGN <= 167): # Rhombohedral I
            LC = 'RI'
            if (order == 2): ECs =  6
            if (order == 3): ECs = 14
        
        elif(168 <= SGN and SGN <= 176): # Hexagonal II
            LC = 'HII'
            if (order == 2): ECs =  5
            if (order == 3): ECs = 12
        
        elif(177 <= SGN and SGN <= 194): # Hexagonal I
            LC = 'HI'
            if (order == 2): ECs =  5
            if (order == 3): ECs = 10
        
        elif(195 <= SGN and SGN <= 206): # Cubic II
            LC = 'CII'
            if (order == 2): ECs =  3
            if (order == 3): ECs =  8
        
        elif(207 <= SGN and SGN <= 230): # Cubic I
            LC = 'CI'
            if (order == 2): ECs =  3
            if (order == 3): ECs =  6
        
        return LC, ECs


    def get_Lag_strain_list(self, LC):

        if (order == 2):
            if (LC == 'CI' or \
                LC == 'CII'):
                Lag_strain_list = ['01','08','23']
            if (LC == 'HI' or \
                LC == 'HII'):
                Lag_strain_list = ['01','26','04','03','17']
            if (LC == 'RI'):
                Lag_strain_list = ['01','08','04','02','05','10']
            if (LC == 'RII'):
                Lag_strain_list = ['01','08','04','02','05','10','11']
            if (LC == 'TI'):
                Lag_strain_list = ['01','26','27','04','05','07']
            if (LC == 'TII'):
                Lag_strain_list = ['01','26','27','28','04','05','07']
            if (LC == 'O'):
                Lag_strain_list = ['01','26','25','27','03','04','05','06','07']
            if (LC == 'M'):
                Lag_strain_list = ['01','25','24','28','29','27','20','12','03','04','05','06','07']
            if (LC == 'N'):
                Lag_strain_list = ['02','03','04','05','06','07','08','09','10','11',\
                                   '12','13','14','15','16','17','18','19','20','21','22']
    
        if (order == 3):
            if (LC == 'CI'):
                Lag_strain_list = ['01','08','23','32','10','11']
            if (LC == 'CII'):
                Lag_strain_list = ['01','08','23','32','10','11','12','09']
            if (LC == 'HI'):
                Lag_strain_list = ['01','26','04','03','17','30','08','02','10','14']
            if (LC == 'HII'):
                Lag_strain_list = ['01','26','04','03','17','30','08','02','10','14','12','31']
            if (LC == 'RI'):
                Lag_strain_list = ['01','08','04','02','05','10','11','26','09','03','17','34','33','35']
            if (LC == 'RII'):
                sys.exit('\n.... Oops SORRY: Not implemented yet. \n')
            if (LC == 'TI'):
                sys.exit('\n.... Oops SORRY: Not implemented yet. \n')
            if (LC == 'TII'):
                sys.exit('\n.... Oops SORRY: Not implemented yet. \n')
            if (LC == 'O'):
                sys.exit('\n.... Oops SORRY: Not implemented yet. \n')
            if (LC == 'M'):
                sys.exit('\n.... Oops SORRY: Not implemented yet. \n')
            if (LC == 'N'):
                sys.exit('\n.... Oops SORRY: Not implemented yet. \n')
    
        return Lag_strain_list

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

    def get_elastic_constant_matrix(self, structure_id, LC, A2):

        import numpy as np

        s0 = load_node(structure_id)

        C = np.zeros((6,6))
        
        #%!%!%--- Cubic structures ---%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%
        if (LC == 'CI' or \
            LC == 'CII'):
            C[0,0] =-2.*(A2[0]-3.*A2[1])/3.
            C[1,1] = C[0,0]
            C[2,2] = C[0,0]
            C[3,3] = A2[2]/6.
            C[4,4] = C[3,3]
            C[5,5] = C[3,3]
            C[0,1] = (2.*A2[0]-3.*A2[1])/3.
            C[0,2] = C[0,1]
            C[1,2] = C[0,1]
        #--------------------------------------------------------------------------------------------------
        
        #%!%!%--- Hexagonal structures ---%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%
        if (LC == 'HI' or \
            LC == 'HII'):
            C[0,0] = 2.*A2[3]
            C[0,1] = 2./3.*A2[0] + 4./3.*A2[1] - 2.*A2[2] - 2.*A2[3]
            C[0,2] = 1./6.*A2[0] - 2./3.*A2[1] + 0.5*A2[2]
            C[1,1] = C[0,0]
            C[1,2] = C[0,2]
            C[2,2] = 2.*A2[2]
            C[3,3] =-0.5*A2[2] + 0.5*A2[4]
            C[4,4] = C[3,3]
            C[5,5] = .5*(C[0,0] - C[0,1])
        #--------------------------------------------------------------------------------------------------
        
        #%!%!%--- Rhombohedral I structures ---%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!
        if (LC == 'RI'):
            C[0,0] = 2.*A2[3]
            C[0,1] = A2[1]- 2.*A2[3]
            C[0,2] = .5*( A2[0] - A2[1] - A2[2])
            C[0,3] = .5*(-A2[3] - A2[4] + A2[5])
            C[1,1] = C[0,0]
            C[1,2] = C[0,2]
            C[1,3] =-C[0,3]
            C[2,2] = 2.*A2[2]
            C[3,3] = .5*A2[4]
            C[4,4] = C[3,3]
            C[4,5] = C[0,3]
            C[5,5] = .5*(C[0,0] - C[0,1])
        #--------------------------------------------------------------------------------------------------
        
        #%!%!%--- Rhombohedral II structures ---%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%
        if (LC == 'RII'):
            C[0,0] = 2.*A2[3]
            C[0,1] = A2[1]- 2.*A2[3]
            C[0,2] = .5*( A2[0] - A2[1] - A2[2])
            C[0,3] = .5*(-A2[3] - A2[4] + A2[5])
            C[0,4] = .5*(-A2[3] - A2[4] + A2[6])
            C[1,1] = C[0,0]
            C[1,2] = C[0,2]
            C[1,3] =-C[0,3]
            C[1,4] =-C[0,4]
            C[2,2] = 2.*A2[2]
            C[3,3] = .5*A2[4]
            C[3,5] =-C[0,4]
            C[4,4] = C[3,3]
            C[4,5] = C[0,3]
            C[5,5] = .5*(C[0,0] - C[0,1])
        #--------------------------------------------------------------------------------------------------
        #%!%!%--- Tetragonal I structures ---%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!
        if (LC == 'TI'):
            C[0,0] = (A2[0]+2.*A2[1])/3.+.5*A2[2]-A2[3]
            C[0,1] = (A2[0]+2.*A2[1])/3.-.5*A2[2]-A2[3]
            C[0,2] = A2[0]/6.-2.*A2[1]/3.+.5*A2[3]
            C[1,1] = C[0,0]
            C[1,2] = C[0,2]
            C[2,2] = 2.*A2[3]
            C[3,3] = .5*A2[4]
            C[4,4] = C[3,3]
            C[5,5] = .5*A2[5]
        #--------------------------------------------------------------------------------------------------
        
        #%!%!%--- Tetragonal II structures ---%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%
        if (LC == 'TII'):
            C[0,0] = (A2[0]+2.*A2[1])/3.+.5*A2[2]-A2[4]
            C[1,1] = C[0,0]
            C[0,1] = (A2[0]+2.*A2[1])/3.-.5*A2[2]-A2[4]
            C[0,2] = A2[0]/6.-(2./3.)*A2[1]+.5*A2[4]
            C[0,5] = (-A2[2]+A2[3]-A2[6])/4.
            C[1,2] = C[0,2]
            C[1,5] =-C[0,5]
            C[2,2] = 2.*A2[4]
            C[3,3] = .5*A2[5]
            C[4,4] = C[3,3]
            C[5,5] = .5*A2[6]
        #--------------------------------------------------------------------------------------------------
        
        #%!%!%--- Orthorhombic structures ---%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!
        if (LC == 'O'):
            C[0,0] = 2.*A2[0]/3.+4.*A2[1]/3.+A2[3]-2.*A2[4]-2.*A2[5]
            C[0,1] = 1.*A2[0]/3.+2.*A2[1]/3.-.5*A2[3]-A2[5]
            C[0,2] = 1.*A2[0]/3.-2.*A2[1]/3.+4.*A2[2]/3.-.5*A2[3]-A2[4]
            C[1,1] = 2.*A2[4]
            C[1,2] =-2.*A2[1]/3.-4.*A2[2]/3.+.5*A2[3]+A2[4]+A2[5]
            C[2,2] = 2.*A2[5]
            C[3,3] = .5*A2[6]
            C[4,4] = .5*A2[7]
            C[5,5] = .5*A2[8]
        #--------------------------------------------------------------------------------------------------
        
        #%!%!%--- Monoclinic structures ---%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!
        if (LC == 'M'):
            C[0,0] = 2.*A2[0]/3.+8.*(A2[1]+A2[2])/3.-2.*(A2[5]+A2[8]+A2[9])
            C[0,1] = A2[0]/3.+4.*(A2[1]+A2[2])/3.-2.*A2[5]-A2[9]
            C[0,2] =(A2[0]-4.*A2[2])/3.+A2[5]-A2[8]
            C[0,5] =-1.*A2[0]/6.-2.*(A2[1]+A2[2])/3.+.5*(A2[5]+A2[7]+A2[8]+A2[9]-A2[12])
            C[1,1] = 2.*A2[8]
            C[1,2] =-4.*(2.*A2[1]+A2[2])/3.+2.*A2[5]+A2[8]+A2[9]
            C[1,5] =-1.*A2[0]/6.-2.*(A2[1]+A2[2])/3.-.5*A2[3]+A2[5]+.5*(A2[7]+A2[8]+A2[9])
            C[2,2] = 2.*A2[9]
            C[2,5] =-1.*A2[0]/6.+2.*A2[1]/3.-.5*(A2[3]+A2[4]-A2[7]-A2[8]-A2[9]-A2[12])
            C[3,3] = .5*A2[10]
            C[3,4] = .25*(A2[6]-A2[10]-A2[11])
            C[4,4] = .5*A2[11]
            C[5,5] = .5*A2[12]
        #--------------------------------------------------------------------------------------------------
        
        #%!%!%--- Triclinic structures ---%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%
        if (LC == 'N'):
            C[0,0] = 2.*A2[0]
            C[0,1] = 1.*(-A2[0]-A2[1]+A2[6])
            C[0,2] = 1.*(-A2[0]-A2[2]+A2[7])
            C[0,3] = .5*(-A2[0]-A2[3]+A2[8])
            C[0,4] = .5*(-A2[0]+A2[9]-A2[4])
            C[0,5] = .5*(-A2[0]+A2[10]-A2[5])
            C[1,1] = 2.*A2[1]
            C[1,2] = 1.*(A2[11]-A2[1]-A2[2])
            C[1,3] = .5*(A2[12]-A2[1]-A2[3])
            C[1,4] = .5*(A2[13]-A2[1]-A2[4])
            C[1,5] = .5*(A2[14]-A2[1]-A2[5])
            C[2,2] = 2.*A2[2]
            C[2,3] = .5*(A2[15]-A2[2]-A2[3])
            C[2,4] = .5*(A2[16]-A2[2]-A2[4])
            C[2,5] = .5*(A2[17]-A2[2]-A2[5])
            C[3,3] = .5*A2[3]
            C[3,4] = .25*(A2[18]-A2[3]-A2[4])
            C[3,5] = .25*(A2[19]-A2[3]-A2[5])
            C[4,4] = .5*A2[4]
            C[4,5] = .25*(A2[20]-A2[4]-A2[5])
            C[5,5] = .5*A2[5]
        #--------------------------------------------------------------------------------------------------
        
        for i in range(5):
            for j in range(i+1,6):
                C[j,i] = C[i,j]

        V0 = load_node(structure_id).get_cell_volume()
        C = C / V0
        
        return C


    def get_stress_tensor_matrix(self, structure_id, LC, A1):

        import numpy as np

        s0 = load_node(structure_id)

        S = np.zeros((3,3))

        #%!%!%--- Cubic structures ---%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%
        if (LC == 'CI' or \
            LC == 'CII'):
            S[0,0] = A1[0]
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
            S[0,0] = A1[0]
            S[1,1] = S[0,0]
            S[2,2] = A1[1]
        #--------------------------------------------------------------------------------------------------

        #%!%!%--- Orthorhombic structures ---%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!
        if (LC == 'O'):
            S[0,0] = A1[0]
            S[1,1] = A1[1]
            S[2,2] = A1[2]
        #--------------------------------------------------------------------------------------------------

        #%!%!%--- Monoclinic structures ---%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!
        if (LC == 'M'):
            S[0,0] = A1[0]
            S[1,1] = A1[1]
            S[2,2] = A1[2]

            if (s0.cell_angles[0] != 90): unique_axis == 'a'
            if (s0.cell_angles[1] != 90): unique_axis == 'b'
            if (s0.cell_angles[2] != 90): unique_axis == 'c'

            if (unique_axis == 'a'): S[1,2] = A1[3]
            if (unique_axis == 'b'): S[0,2] = A1[3]
            if (unique_axis == 'c'): S[0,1] = A1[3]
        #--------------------------------------------------------------------------------------------------

        #%!%!%--- Triclinic structures ---%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%
        if (LC == 'N'):
            S[0,0] = A1[0]
            S[1,1] = A1[1]
            S[2,2] = A1[2]
            S[1,2] = A1[3]
            S[0,2] = A1[4]
            S[0,1] = A1[5]
        #--------------------------------------------------------------------------------------------------

        for i in range(2):
            for j in range(i+1, 3):
                S[j,i] = S[i,j]

        V0 = load_node(structure_id).get_cell_volume()
        S = S / V0

        return S

    def get_pw_parameters(self):

        parameters = ParameterData(dict={
            'CONTROL': {
                'calculation': 'scf',
                'restart_mode': 'from_scratch',
                'wf_collect': True,
                'tprnfor': True,
                'tstress': True,
            },
            'SYSTEM': {
                'ecutwfc': 80.,
                'ecutrho': 640.,
            },
            'ELECTRONS': {
                'conv_thr': 1.e-10,
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

    def get_pw_calculation(self, pw_structure, pw_parameters, pw_kpoint):

        params = self.get_parameters()

        pw_codename = params['pw_codename']
        num_machines = params['num_machines']
        max_wallclock_seconds = params['max_wallclock_seconds']
        pseudo_family = params['pseudo_family']

        code = Code.get_from_string(pw_codename)
        computer = code.get_remote_computer()

        QECalc = CalculationFactory('quantumespresso.pw')

        calc = QECalc(computer=computer)
        calc.set_max_wallclock_seconds(max_wallclock_seconds)
        calc.set_resources({"num_machines": num_machines})
        calc.store()

        calc.use_code(code)

        calc.use_structure(pw_structure)
        calc.use_pseudos_from_family(pseudo_family)
        calc.use_parameters(pw_parameters)
        calc.use_kpoints(pw_kpoint)

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
        self.add_attribute('alat_steps', alat_steps)

        if use_symmetry:
           SGN = self.get_space_group_number(structure_id=structure_id)
        else:
           SGN = 1
        
        self.append_to_report(x_material + " structure has space group number " + str(SGN))

        LC, ECs = self.get_LC_ECs(space_group_number=SGN)

        self.add_attribute('LC', LC)
        self.add_attribute('ECs', ECs)

        # distorted_structure_index = range(len(eps * SCs))
        # aiidalogger.info("Storing distorted_structure_index as " + str(distorted_structure_index))
        # self.add_attribute('distorted_structure_index', distorted_structure_index)

        def_list = self.get_Lagrange_distorted_index(structure_id=structure_id, LC=LC)
        self.add_attribute('def_list', def_list)

        distorted_structure_index = []
        eps_index = 0
        for i in def_list:
            for a in eps:

                eps_index = eps_index + 1

                distorted_structure_index.append(eps_index)

        self.add_attribute('distorted_structure_index', distorted_structure_index)


        for ii in distorted_structure_index:

            a = eps[ii % alat_steps - 1]
            i = def_list[int((ii - 1) / alat_steps)]

            M_Lagrange_eps = self.get_Lagrange_strain_matrix(eps=a, def_mtx_index=i)

            distorted_structure = self.get_Lagrange_distorted_structure(structure_id=structure_id, M_Lagrange_eps=M_Lagrange_eps)

            calc = self.get_pw_calculation(distorted_structure,
                                               self.get_pw_parameters(),
                                               self.get_kpoints())

            self.attach_calculation(calc)

            self.append_to_report("Preparing structure {0} with alat_strain {1} for SC {2}".format(x_material, a, i))
            self.append_to_report("Distorted structure with index {0} has ID {1}".format(ii, distorted_structure.id))
            self.append_to_report("Calculation with pk {0}".format(calc.id))

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

        alat_steps = self.get_attribute("alat_steps")
        eps = self.get_attribute("eps")
        distorted_structure_index = self.get_attribute("distorted_structure_index")

        #for i in range(len(eps)):
        #    self.append_to_report(" simulated with strain =" + ", e=" + str(eps[i]))

        def_list = self.get_attribute("def_list")

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

        e_calcs_correct = e_calcs[::-1]

        #  Add to report
        #-----------------------------------------
        for i in range(len(distorted_structure_index)):
            self.append_to_report(x_material + " simulated with strain =" + str(distorted_structure_index[i]) + ", e=" + str(e_calcs_correct[i]))

        #  Obtain stress tensor by polyfit
        #-----------------------------------------
        #alat_steps = 5
        #SCs = int(len(distorted_structure_index) / alat_steps)

        SCs = self.get_attribute("SCs")
        A2 = []
        energy = []
        fit_order = 3
        #eps = np.linspace(-0.008, 0.008, alat_steps).tolist()

        for i in range(ECs):
            energy = e_calcs_correct[i*alat_steps:(i+1)*alat_steps]
            coeffs = np.polyfit(eps, energy, fit_order)
            A1.append(coeffs[fit_order-1]*ToGPa)
 
            #lsq_coeffs, ier = lsq_fit(energy, eps)
            #A1.append(lsq_coeffs[order-1]*ToGPa)

        A2 = np.array(A2)

        LC = self.get_attribute("LC")


        C = self.get_elastic_constant_matrix(structure_id=structure_id, LC=LC, A2=A2)

        self.append_to_report(x_material + " simulated with structure_id =" + str(structure_id) + " has stress tensor in GPa C11 ... C16=" + str(C[0,:]))
        self.append_to_report(x_material + " simulated with structure_id =" + str(structure_id) + " has stress tensor in GPa C21 ... C26=" + str(C[1,:]))
        self.append_to_report(x_material + " simulated with structure_id =" + str(structure_id) + " has stress tensor in GPa C31 ... C36=" + str(C[2,:]))
        self.append_to_report(x_material + " simulated with structure_id =" + str(structure_id) + " has stress tensor in GPa C41 ... C46=" + str(C[3,:]))
        self.append_to_report(x_material + " simulated with structure_id =" + str(structure_id) + " has stress tensor in GPa C51 ... C56=" + str(C[4,:]))
        self.append_to_report(x_material + " simulated with structure_id =" + str(structure_id) + " has stress tensor in GPa C61 ... C66=" + str(C[5,:]))

        S  = np.linalg.inv(C)

        self.append_to_report(x_material + " simulated with structure_id =" + str(structure_id) + " has Elastic compliance matrix in 1/GPa S11 ... S16=" + str(S[0,:]))
        self.append_to_report(x_material + " simulated with structure_id =" + str(structure_id) + " has Elastic compliance matrix in 1/GPa S21 ... S26=" + str(S[1,:]))
        self.append_to_report(x_material + " simulated with structure_id =" + str(structure_id) + " has Elastic compliance matrix in 1/GPa S31 ... S36=" + str(S[2,:]))
        self.append_to_report(x_material + " simulated with structure_id =" + str(structure_id) + " has Elastic compliance matrix in 1/GPa S41 ... S46=" + str(S[3,:]))
        self.append_to_report(x_material + " simulated with structure_id =" + str(structure_id) + " has Elastic compliance matrix in 1/GPa S51 ... S56=" + str(S[4,:]))
        self.append_to_report(x_material + " simulated with structure_id =" + str(structure_id) + " has Elastic compliance matrix in 1/GPa S61 ... S66=" + str(S[5,:]))

        eigval=np.linalg.eig(C)

        self.append_to_report(x_material + " simulated with structure_id =" + str(structure_id) + " has Eigenvalues of elastic constant (stiffness) matrix:=" + str(eigval[0,:]))
        self.append_to_report(x_material + " simulated with structure_id =" + str(structure_id) + " has Eigenvalues of elastic constant (stiffness) matrix:=" + str(eigval[1,:]))
        self.append_to_report(x_material + " simulated with structure_id =" + str(structure_id) + " has Eigenvalues of elastic constant (stiffness) matrix:=" + str(eigval[2,:]))
        self.append_to_report(x_material + " simulated with structure_id =" + str(structure_id) + " has Eigenvalues of elastic constant (stiffness) matrix:=" + str(eigval[3,:]))
        self.append_to_report(x_material + " simulated with structure_id =" + str(structure_id) + " has Eigenvalues of elastic constant (stiffness) matrix:=" + str(eigval[4,:]))
        self.append_to_report(x_material + " simulated with structure_id =" + str(structure_id) + " has Eigenvalues of elastic constant (stiffness) matrix:=" + str(eigval[5,:]))

#%!%!%--- Calculating the elastic moduli ---%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%
        BV = (C[0,0]+C[1,1]+C[2,2]+2*(C[0,1]+C[0,2]+C[1,2]))/9
        GV = ((C[0,0]+C[1,1]+C[2,2])-(C[0,1]+C[0,2]+C[1,2])+3*(C[3,3]+C[4,4]+C[5,5]))/15
        EV = (9*BV*GV)/(3*BV+GV)
        nuV= (1.5*BV-GV)/(3*BV+GV)
        BR = 1/(S[0,0]+S[1,1]+S[2,2]+2*(S[0,1]+S[0,2]+S[1,2]))
        GR =15/(4*(S[0,0]+S[1,1]+S[2,2])-4*(S[0,1]+S[0,2]+S[1,2])+3*(S[3,3]+S[4,4]+S[5,5]))
        ER = (9*BR*GR)/(3*BR+GR)
        nuR= (1.5*BR-GR)/(3*BR+GR)
        BH = 0.50*(BV+BR)
        GH = 0.50*(GV+GR)
        EH = (9.*BH*GH)/(3.*BH+GH)
        nuH= (1.5*BH-GH)/(3.*BH+GH)
        AVR= 100.*(GV-GR)/(GV+GR)
#--------------------------------------------------------------------------------------------------

        self.append_to_report(x_material + " simulated with structure_id =" + str(structure_id) + " has Voigt bulk  modulus, B_V (Gpa)" + BV)
        self.append_to_report(x_material + " simulated with structure_id =" + str(structure_id) + " has Voigt shear modulus, G_V (Gpa)" + GV)

        self.append_to_report(x_material + " simulated with structure_id =" + str(structure_id) + " has Reuss bulk  modulus, B_R (Gpa)" + BR)
        self.append_to_report(x_material + " simulated with structure_id =" + str(structure_id) + " has Reuss shear modulus, G_R (Gpa)" + GR)

        self.append_to_report(x_material + " simulated with structure_id =" + str(structure_id) + " has Hill bulk  modulus, B_H (Gpa)" + BH)
        self.append_to_report(x_material + " simulated with structure_id =" + str(structure_id) + " has Hill shear modulus, G_H (Gpa)" + GH)

        self.append_to_report(x_material + " simulated with structure_id =" + str(structure_id) + " has Voigt Young modulus, E_V (Gpa)" + EV)
        self.append_to_report(x_material + " simulated with structure_id =" + str(structure_id) + " has Voigt Poisson ratio, nu_V" + nuV)
        
        self.append_to_report(x_material + " simulated with structure_id =" + str(structure_id) + " has Reuss Young  modulus, E_R (Gpa)" + ER)
        self.append_to_report(x_material + " simulated with structure_id =" + str(structure_id) + " has Reuss Poisson ratio, nu_R " + nuR)

        self.append_to_report(x_material + " simulated with structure_id =" + str(structure_id) + " has Hill Young  modulus, E_H (Gpa)" + EH)
        self.append_to_report(x_material + " simulated with structure_id =" + str(structure_id) + " has Hill Poisson modulus, nu_H (Gpa)" + nuH)

        self.append_to_report(x_material + " simulated with structure_id =" + str(structure_id) + " has Elastic Anisotropy in polycrystalline, AVR =" + nuH)

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


