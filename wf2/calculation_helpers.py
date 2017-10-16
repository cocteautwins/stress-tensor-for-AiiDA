from aiida.orm.data.array.kpoints import KpointsData
from aiida.orm.data.parameter import ParameterData
from aiida.orm.data.structure import StructureData
from aiida.common.example_helpers import test_and_get_code
from aiida.orm import Group, load_node
import numpy as np

def create_calculation_parameters(code, partition, num_ranks_per_node, num_ranks_kp, num_ranks_diag):
    """
    Create a dictionary with parameters for a job execution.
    """

    if partition not in ['cpu', 'gpu']:
        raise RuntimeError('wrong partition name')
    
    num_cores = {'cpu' : 36, 'gpu' : 12}

    if num_cores[partition] % num_ranks_per_node != 0:
        raise RuntimeError('wrong number of ranks per node')
    
    num_threads = num_cores[partition] / num_ranks_per_node

    # total number of ranks
    num_ranks = num_ranks_kp * num_ranks_diag
    
    # get number of nodes
    num_nodes = max(1, num_ranks / num_ranks_per_node)
    
    print("partition: %s"%partition)
    print("number of nodes : %i"%num_nodes)

    # create dictionary to store parameters
    params = {}
    
    params['job_tag'] = "%iN:%iR:%iT @ %s"%(num_nodes, num_ranks_per_node, num_threads, partition)

    environment_variables = {'OMP_NUM_THREADS': str(num_threads),\
                             'MKL_NUM_THREADS': str(num_threads),\
                             'KMP_AFFINITY': 'granularity=fine,compact,1'}
    if partition == 'gpu' and num_ranks_per_node > 1:
        environment_variables['CRAY_CUDA_MPS'] = '1'

    params['environment_variables'] = environment_variables

    #calc.set_custom_scheduler_commands('#SBATCH -A, --account=mr21')
    if partition == 'cpu':
        params['custom_scheduler_commands'] = u"#SBATCH -C mc"
    if partition == 'gpu':
        params['custom_scheduler_commands'] = u"#SBATCH -C gpu"

    # create settings
    if code.get_input_plugin_name() == 'quantumespresso.pw':
        if partition == 'cpu':
            settings = ParameterData(dict={
                'CMDLINE' : ['-npool', str(num_ranks_kp), '-ndiag', str(num_ranks_diag)]})
        if partition == 'gpu':
            settings = ParameterData(dict={
                #'CMDLINE' : ['-npool', str(num_ranks_kp), '-ndiag', str(num_ranks_diag), '-sirius', '-sirius_cfg', '/users/antonk/codes/config.json']})
                'CMDLINE' : ['-npool', str(num_ranks_kp), '-ndiag', str(num_ranks_diag), '-sirius']})
            
        parameters = ParameterData(dict={
            'CONTROL': {
                'calculation'  : 'scf',
                'restart_mode' : 'from_scratch',
                'disk_io'      : 'none'
                },
            'SYSTEM': {
                'ecutwfc': 80.,
                'ecutrho': 640.,
                'occupations': 'smearing',
                'smearing': 'gauss',
                'degauss': 0.1
                },
            'ELECTRONS': {
                'conv_thr': 1.e-9,
                'electron_maxstep': 100,
                'mixing_beta': 0.7
                }})

    if code.get_input_plugin_name() == 'exciting.exciting':
        parameters = ParameterData(dict={'groundstate' : {'xctype' : 'GGA_PBE',
                                         'gmaxvr' : '30.0',
                                         'rgkmax' : '12.0',
                                         }})
        settings = ParameterData(dict={})


    params['calculation_settings'] = settings
    params['calculation_parameters'] = parameters
    params['mpirun_extra_params'] = ['-n', str(num_ranks), '-c', str(num_threads), '--hint=nomultithread','--unbuffered']
    params['calculation_resources'] = {'num_machines': num_nodes, 'num_mpiprocs_per_machine': num_ranks_per_node}
    params['code'] = code

    return params

def create_calculation(structure, params, calc_label, calc_desc):
    """
    Create calculation object from structure and a dictionary of parameters.
    Calculation has to be stored in DB by the caller.
    """
    code = params['code']
    
    calc = code.new_calc()
    calc.set_max_wallclock_seconds(params.get('calculation_wallclock_seconds', 3600)) # in second
    calc.set_resources(params['calculation_resources'])
    
    calc.use_structure(structure)
    calc.use_parameters(params['calculation_parameters'])
    calc.use_kpoints(params['kpoints'])
    calc.use_settings(params['calculation_settings'])
    if code.get_input_plugin_name() == 'quantumespresso.pw':
        calc.use_pseudos_from_family(params['atomic_files'])
    calc.set_environment_variables(params['environment_variables'])
    calc.set_mpirun_extra_params(params['mpirun_extra_params'])
    calc.set_custom_scheduler_commands(params['custom_scheduler_commands'])
    calc.label = calc_label
    calc.description = calc_desc

    return calc

def scaled_structure(structure, scale):

    new_structure = StructureData(cell=np.array(structure.cell)*scale)

    for site in structure.sites:
        new_structure.append_atom(position=np.array(site.position)*scale, \
                                  symbols=structure.get_kind(site.kind_name).symbol,\
                                  name=site.kind_name)
    new_structure.label = 'created inside stress tensor run'
    new_structure.description = "auxiliary structure for stress tensor "\
                                "created from the original structure with PK=%i, "\
                                "lattice constant scaling: %f"%(structure.pk, scale)

    return new_structure

def get_Lagrange_distorted_structure(structure_id, M_Lagrange_eps):

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

def get_Lagrange_distorted_index(structure_id, LC):

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


def get_Lagrange_strain_matrix(eps=0.0, def_mtx_index='1'):

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

def get_space_group_number(structure_id):

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

def get_Laue_dict(space_group_number):

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

def def_Laue_dict(LC):

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

def get_stress_tensor_matrix(structure_id, LC, A1):

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

def submit_stress_tensor(**kwargs):
    # get code
    #code = Code.get(label='pw.sirius.x', computername='piz_daint', useremail='antonk@cscs.ch')
    code = test_and_get_code('pw.sirius.x', expected_code_type='quantumespresso.pw')
    #code.set_prepend_text(prep_text)

    # calculation should always belong to some group, otherwise things get messy after some time
    stress_tensor_grp, created = Group.get_or_create(name=kwargs['group'])
    
    # create parameters
    params = create_calculation_parameters(code,
                                           kwargs.get('partition', 'cpu'),
                                           kwargs.get('num_ranks_per_node', 36),
                                           kwargs.get('num_ranks_kp', 1),
                                           kwargs.get('num_ranks_diag', 1))
    # load structure
    structure = load_node(kwargs['structure_pk'])
    
    # generate k-points
    params['kpoints'] = KpointsData()
    params['kpoints'].set_kpoints_mesh(kwargs.get('kmesh', [24, 24, 24]), offset=(0.0, 0.0, 0.0))
    params['atomic_files'] = kwargs['atomic_files']
    params['calculation_wallclock_seconds'] = kwargs.get('time_limit', 3600)
    params['structure'] = structure
    params['num_points'] = 5
    params['group'] = kwargs['group']
    params['kpoints'].store()
    params['calculation_parameters'].store()
    params['calculation_settings'].store()

    stress_tensor_dict = {}
    stress_tensor_dict['label'] = 'stress_tensor_' + structure.get_formula() + '_' + code.label
    stress_tensor_dict['description'] = "Stress tensor for structure with PK=%i"%structure.pk
    stress_tensor_dict['calc_pk'] = []
    stress_tensor_dict['num_points'] = params['num_points']
    stress_tensor_dict['structure_pk'] = structure.pk
    stress_tensor_dict['code_pk'] = code.pk
    stress_tensor_dict['job_tag'] = params['job_tag']

    # volume scales from 0.94 to 1.06, alat scales as pow(1/3)
    scales = np.linspace(0.992, 1.008, num=params['num_points']).tolist()

    eps = np.linspace(-0.008, 0.008, num=params['num_points']).tolist()
    #scales = np.linspace(0.99, 1.05, num=params['num_points']).tolist()

    use_symmetry = .False.

    if use_symmetry:
       SGN = get_space_group_number(structure_id=structure_id)
    else:
       SGN = 1

    LC = self.get_Laue_dict(space_group_number=SGN)

    def_list = get_Lagrange_distorted_index(structure_id=structure_id, LC=LC)

    SCs = len(def_list)

    alat_steps = params['num_points']

    distorted_structure_index = []
    eps_index = 0
    for i in def_list:
        for a in eps:

            eps_index = eps_index + 1

            distorted_structure_index.append(eps_index)

    for ii in distorted_structure_index:

        a = eps[ii % alat_steps - 1]
        i = def_list[int((ii - 1) / alat_steps)]

        M_Lagrange_eps = get_Lagrange_strain_matrix(eps=a, def_mtx_index=i)

        structure_new = get_Lagrange_distorted_structure(structure_id=structure_id, M_Lagrange_eps=M_Lagrange_eps)

        structure_new.store()
        
        calc_label = 'gs_' + structure.get_formula() + '_' + code.label
        calc_desc = params['job_tag']
    
        # create calculation
        calc = create_calculation(structure_new, params, calc_label, calc_desc)
        calc.store()
        print "created calculation with uuid='{}' and PK={}".format(calc.uuid, calc.pk)
        stress_tensor_grp.add_nodes([calc])
        calc.submit()
        stress_tensor_dict['calc_pk'].append(calc.pk)
    
    stress_tensor_node = ParameterData(dict=stress_tensor_dict)
    stress_tensor_node.store()
    stress_tensor_grp.add_nodes([stress_tensor_node])
    print "created stress tensor node with uuid='{}' and PK={}".format(stress_tensor_node.uuid, stress_tensor_node.pk)
