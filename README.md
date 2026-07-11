# RBPNFE
A python module for the evaluation of nucleosome positioning free energies.

Clone with all submodules
```console
git clone --recurse-submodules -j8 https://github.com/eskoruppa/RBPNFE.git
```


## Basic Function

Free energy calculations for individual positioning sequences (147bp) can be accessed via the `NucFreeEnergy` object.

```python
import rbpnfe

params_model = 'MD'
hard_constraint = False

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
    use_correction = True
    )

print(f'Full Free Energy:         {nout["F"]:.2f} kT')
print(f'Fluctuation Contribution: {nout["F_fluctuation"]:.2f} kT')
print(f'Enthalpic Contribution:   {nout["F_enthalpy"]:.2f} kT')
```

A fully specified call, exposing every option, looks like this:

```python
nfe = rbpnfe.NucFreeEnergy(
    params_model     = 'cgna+',            # elastic model (see below)
    hardconstraint   = True,               # hard vs. soft constraint model
    rescale_factors  = [1,1,1,1,1,1],      # optional per-DOF stiffness rescaling (6 positive values)
    flanking         = 10,                 # flanking bp added when generating parameters
    cgnaplus_setname = 'curves_plus',      # cgNA+ parameter set (only used for cgna+)
    mode             = 'compse3',          # hard-constraint backend: 'compse3' or 'legacy'
    midstep_locations= None,               # override the 28 binding-site step indices
    triadfn          = None,               # override the nucleosome triad state file
    Kmat_file        = None,               # override the soft-constraint stiffness matrix
    )

nout = nfe.eval(
    seq,                                   # 147 bp sequence
    shl_open_left  = 0,                    # open superhelical locations from the left
    shl_open_right = 0,                    # open superhelical locations from the right
    use_correction = True,                 # apply translation correction
    )
```

### Constructor options: `rbpnfe.NucFreeEnergy(...)` <a name=args></a>

- `params_model` (str, default `'hybrid'`):
    Select the elastic model for the generation of stiffness and structure parameters:
    - `md` (alias `lankas`): Molecular Dynamics derived parameters from Lankas et al. \[[1](#lank03)\]
    - `crystal` (alias `olson`): Parameters from crystallographic data from Olson et al. \[[2](#olson98)\]
    - `cgna+` (aliases `cgnaplus`, `cgnap`): Parameters derived from cgNA+ via marginalization to rigid base pair model \[[3](#sharma23)\]
    - `hybrid`: md parameters for stiffness and crystal parameters for ground state.

- `hardconstraint` (bool, default `False`):
    Select nucleosome binding model
    - `True`: Use hard constraint model
    - `False`: Use soft constraint model

- `rescale_factors` (sequence of 6 floats or `None`, default `None`):
    Optional per-degree-of-freedom rescaling of the base-pair-step stiffness, ordered
    `[tilt, roll, twist, shift, slide, rise]`. Must contain exactly 6 strictly positive
    values (`1.0` leaves a DOF unchanged). `None` applies no rescaling.

- `flanking` (int, default `10`):
    Number of flanking base pairs added on each side of the sequence during parameter
    generation to reduce edge effects. For the `cgna+` model the flanks are stripped again
    after generation.

- `cgnaplus_setname` (str, default `'curves_plus'`):
    Name of the cgNA+ parameter set. Only used when `params_model` is `cgna+`.

- `mode` (str, default `'compse3'`):
    Evaluation backend for the **hard-constraint** model:
    - `compse3`: current CompSE3-based evaluation
    - `legacy`: previous implementation

    Ignored for the soft-constraint model.

- `midstep_locations` (list[int] or `None`, default `None`):
    Base-pair-step indices of the 28 binding sites. Defaults to the built-in nucleosome map.

- `triadfn` (str or `None`, default `None`):
    Path to the nucleosome triad state file. Defaults to the bundled `Parameters/Nucleosome.state`.

- `Kmat_file` (str or `None`, default `None`):
    Path to the stiffness (K) matrix `.npy` used by the soft-constraint model. Defaults to the bundled matrix.

### Evaluation options: `nfe.eval(seq, ...)`

- `seq` (str):
    The 147 bp positioning sequence to evaluate.

- `shl_open_left` / `shl_open_right` (int or `None`, default `None`):
    Number of open **superhelical locations** (SHL) counted from the left / right end.
    Each SHL corresponds to 2 binding sites. When set, they override `open_left`/`open_right`.

- `open_left` / `open_right` (int, default `0`):
    Number of open **binding sites** counted from the left / right end (used directly when the
    `shl_*` variants are `None`). `open_left + open_right` must not exceed 28.

- `use_correction` (bool, default `True`):
    Apply translation correction in a second iteration by expanding around compromise rotations
    deduced during the first iteration.

### Return value

`eval` returns a dictionary of free-energy components, all in units of kT:

| key | meaning |
|-----|---------|
| `F` | total free energy |
| `F_fluctuation` | fluctuation (entropic) contribution |
| `F_enthalpy` | enthalpic contribution |
| `F_jacob` | Jacobian contribution |
| `F_freedna` | reference free energy of the free (unbound) DNA |
| `dF` | binding free energy relative to free DNA (`F - F_freedna`) |


## Free Energy Landscape

`eval_landscape` slides a 147 bp window along a longer sequence and evaluates the free
energy at every position. It accepts all the same evaluation options as `eval`, plus
`ncores` and `verbose`. The example below mirrors `example_landscape.py`.

```python
#!/usr/bin/env python3
import os

# Pin the numerical libraries to a single thread BEFORE importing numpy, so each
# parallel worker process stays single-threaded. Parallelism is provided by
# `ncores` (worker processes); letting BLAS also multithread on top of that would
# oversubscribe the cores and slow everything down.
num_cores = 1
os.environ["OMP_NUM_THREADS"]        = f"{num_cores}"
os.environ["OPENBLAS_NUM_THREADS"]   = f"{num_cores}"
os.environ["MKL_NUM_THREADS"]        = f"{num_cores}"
os.environ["VECLIB_MAXIMUM_THREADS"] = f"{num_cores}"
os.environ["NUMEXPR_NUM_THREADS"]    = f"{num_cores}"

import numpy as np
import time
import rbpnfe

params_model    = 'cgnaplus'
hard_constraint = True
ncores          = 8                              # number of parallel worker processes

# optional per-DOF stiffness rescaling [tilt, roll, twist, shift, slide, rise]
factors = [0.6, 0.6, 0.70, 1.0, 1.0, 0.4]

nfe = rbpnfe.NucFreeEnergy(
    params_model    = params_model,
    hardconstraint  = hard_constraint,
    rescale_factors = factors,
    )

# any sequence of length >= 147; every 147 bp window is evaluated
nbp = 400
seq = ''.join(['ATCG'[np.random.randint(4)] for i in range(nbp)])

t1 = time.time()
fes = nfe.eval_landscape(
    seq,
    shl_open_left  = 0,
    shl_open_right = 0,
    use_correction = True,
    ncores         = ncores,
    verbose        = False,
    )
t2 = time.time()
print(f'Evaluated {len(fes)} positions in {t2-t1:.2f} s')

# fes is an (Nnucs, 3) array, Nnucs = len(seq) - 147 + 1
#   column 0: F              (total free energy)
#   column 1: F_fluctuation  (fluctuation contribution)
#   column 2: F_enthalpy     (enthalpic contribution)
best = int(np.argmin(fes[:, 0]))
print(f'Lowest free energy at position {best}: {fes[best, 0]:.2f} kT')
```

### Landscape-specific options: `nfe.eval_landscape(seq, ...)`

In addition to every `eval` option (`shl_open_left`, `shl_open_right`, `open_left`,
`open_right`, `use_correction`):

- `seq` (str):
    Sequence of length `>= 147`. Every 147 bp window is evaluated, yielding
    `Nnucs = len(seq) - 147 + 1` positions.

- `ncores` (int, default `1`):
    Number of parallel worker processes used for the position scan (`1` runs serially).
    Limiting numpy to a single core (see the BLAS environment variables in the example) is
    recommended to avoid oversubscribing the cores.

- `verbose` (bool, default `False`):
    Print a detailed line per position (free-energy components, timing). Takes precedence over
    `progress`. Under `ncores > 1` the lines arrive out of order; the returned array stays ordered.

- `progress` (bool, default `True`):
    Show a live progress bar with an ETA (suppressed when `verbose=True`).

### Return value

`eval_landscape` returns a NumPy array of shape `(Nnucs, 3)` with columns
`[F, F_fluctuation, F_enthalpy]` (in kT), one row per window position.


## References

\[1\] <a name="lank03"></a> F. Lankaš, Jiří Šponer, Jörg Langowski, Thomas E. Cheatham, III, DNA basepair step deformability inferred from molecular dynamics simulations, [Biophys. J, **85**, 2872 (2003)](https://doi.org/10.1016/S0006-3495(03)74710-9).

\[2\] <a name="olson98"></a> W. K. Olson, A. A. Gorin, X. Lu, L. M. Hock, and V. B. Zhurkin, DNA sequence-dependent deformability deduced from protein–DNA crystal complexes, [Proc. Natl. Acad. Sci. U.S.A. **95**, 11163 (1998).](https://doi.org/10.1073/pnas.95.19.11163).

\[3\] <a name="sharma23"></a> R. Sharma, A. S. Patelli, L. de Bruin, and J. H. Maddocks, cgNA+web: A visual interface to the cgNA+ sequence-dependent statistical mechanics model of double-stranded nucleic acids, [J. Mol. Biol. **435**,
167978 (2023).](http://dx.doi.org/10.1016/j.jmb.2023.167978).
