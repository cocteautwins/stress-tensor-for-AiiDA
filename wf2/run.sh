#!/bin/bash

spk=1763
grp='stress_tensor_wf2'

echo $grp

# QE, CPU
verdi run stress_tensor_wf2.py --group=$grp --kmesh 8 8 8 --partition=cpu --ranks_kp=1 --ranks_diag=36 --ranks_per_node=36 --code=QE6.1@daint $spk

# QE, GPU
verdi run stress_tensor_wf2.py --group=$grp --kmesh 8 8 8 --partition=gpu --ranks_kp=1 --ranks_diag=1 --ranks_per_node=1 --code=QE6.1@daint-gpu $spk

# Exciting
verdi run stress_tensor_wf2.py --group=$grp --kmesh 8 8 8 --partition=cpu --ranks_kp=36 --ranks_diag=1 --ranks_per_node=36 --code=Exciting@daint-gpu $spk
