# stress-tensor-for-AiiDA

Example for running workflow under verdi shell:

from aiida.workflows.wf_stress_tensor_exciting_Lagrange import Workflow_LAPW_stress_tensor_Lagrange 
params = {'lapw_codename':'Exciting@daint-gpu', 'num_machines':1, 'num_mpiprocs_per_machine':1, 'max_wallclock_seconds':360*60, 'lapwbasis_family':'high_quality_lapw_species', 'structure_id':1763, 'use_symmetry': True}
wf = Workflow_LAPW_stress_tensor_Lagrange(params=params) 
wf.start()   

List all workflows:
verdi workflow list -a

Check out results for specific system:
verdi workflow report pk

