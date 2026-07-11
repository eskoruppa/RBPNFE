#!/usr/bin/env python3

import sys,glob,os

num_cores = 1
os.environ["OMP_NUM_THREADS"] = f"{num_cores}"
os.environ["OPENBLAS_NUM_THREADS"] = f"{num_cores}"
os.environ["MKL_NUM_THREADS"] = f"{num_cores}"
os.environ["VECLIB_MAXIMUM_THREADS"] = f"{num_cores}"
os.environ["NUMEXPR_NUM_THREADS"] = f"{num_cores}"


import numpy as np
import time
import rbpnfe

params_model = 'md'
params_model = 'cgnaplus'
hard_constraint = True
ncores = 4
verbose = False
use_correction = True

shl_open_left = 0
shl_open_right = 0

factors = [0.6,0.6,0.70,1.0,1.0,0.4]  

nfe = rbpnfe.NucFreeEnergy(
    params_model = params_model,
    hardconstraint=hard_constraint,
    rescale_factors=factors
    )

nbp = 1147
seq = ''.join(['ATCG'[np.random.randint(4)] for i in range(nbp)])

t1 = time.time()
fes = nfe.eval_landscape(
    seq,
    shl_open_left = shl_open_left,
    shl_open_right = shl_open_right,
    use_correction = use_correction,
    ncores = ncores,
    verbose = verbose
    )
t2 = time.time()
print(f'Time elapsed {t2-t1:.4f} s') 

