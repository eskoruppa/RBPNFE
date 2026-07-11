#!/usr/bin/env python3

import sys,glob,os

num_cores = 1
os.environ["OMP_NUM_THREADS"] = f"{num_cores}"
os.environ["OPENBLAS_NUM_THREADS"] = f"{num_cores}"
os.environ["MKL_NUM_THREADS"] = f"{num_cores}"
os.environ["VECLIB_MAXIMUM_THREADS"] = f"{num_cores}"
os.environ["NUMEXPR_NUM_THREADS"] = f"{num_cores}"

import rbpnfe

params_model = 'md'
hard_constraint = True

seq  = "ATCGAGAATCCCGGTGCCGAGGCCGCTCAATTGGTCGTAGACAGCTCTAGCACCGCTTAAACGCACGTACGCGCTGTCCCCCGCGTTTTAACCGCCAAGGGGATTACTCCCTAGTCTCCAGGCACGTGTCAGATATATACATCCGAT"

shl_open_left = 0
shl_open_right = 0

nfe = rbpnfe.NucFreeEnergy(
    params_model = params_model,
    hardconstraint=hard_constraint,
    )

nout = nfe.eval(
    seq,
    shl_open_left = shl_open_left,
    shl_open_right = shl_open_right,
    use_correction = True,
    )

print(f'Full Free Energy:         {nout["F"]:.2f} kT')
print(f'Fluctuation Contribution: {nout["F_fluctuation"]:.2f} kT')
print(f'Enthalpic Contribution:   {nout["F_enthalpy"]:.2f} kT')


import numpy as np
import time
nbp = 500
seqs = ''.join(['ATCG'[np.random.randint(4)] for i in range(nbp)])

for i in range(len(seqs)-len(seq)+1):
    subseq = seqs[i:i+len(seq)]
    t1 = time.time()
    nout = nfe.eval(
        subseq,
        shl_open_left = shl_open_left,
        shl_open_right = shl_open_right,
        use_correction = True,
        )
    t2 = time.time()
    # print(f'Position {i}: {subseq} | Free Energy: {nout["F"]:.2f} kT | Time: {t2-t1:.4f} s')
    print(f'Position {i}: {subseq} | Enthalpy: {nout["F_enthalpy"]:.2f} kT  | Time: {t2-t1:.4f} s')
    # print(f'Position {i}: {subseq} | Free Energy: {nout["F"]:.2f} kT | Fluctuation: {nout["F_fluctuation"]:.2f} kT | Enthalpy: {nout["F_enthalpy"]:.2f} kT | Time: {t2-t1:.4f} s')

    # print(f'Position {i}: {subseq} | Enthalpy: {nout["F_enthalpy"]:.2f} kT')
    # # print(f'Position {i}: {subseq} | Free Energy: {nout["F"]:.2f} kT')
