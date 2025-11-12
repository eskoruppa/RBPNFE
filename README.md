# RBPNFE
A python module for the evaluation of nucleosome positioning free energies.

Clone with all submodules
```console
git clone --recurse-submodules -j8 git@github.com:eskoruppa/RBPNFE.git
```


## Basic Function


```python
import rbpnfe


params_model = 'MD'
hard_constraint = False

nfe = rbpnfe.NucFreeEnergy(
    params_model = params_model,
    hardconstraint=hard_constraint,
    )

seq  = "ATCGAGAATCCCGGTGCCGAGGCCGCTCAATTGGTCGTAGACAGCTCTAGCACCGCTTAAACGCACGTACGCGCTGTCCCCCGCGTTTTAACCGCCAAGGGGATTACTCCCTAGTCTCCAGGCACGTGTCAGATATATACATCCGAT"

shl_open_left = 0
shl_open_right = 0

nout = nfe.eval_single(
    seq,
    shl_open_left = shl_open_left,
    shl_open_right = shl_open_right,
    use_correction = True
    )
```

### Arguments <a name=args></a>

- `params_model`:

    Select the elastic model for the generation of stiffness and structure parameters:

    - `md`: Molecular Dynamics derived parameters from [Lankas et al.](#lank03)
    - `md`: Molecular Dynamics derived parameters from Lankas et al. [1](#lank03)
    - `cgna+`: Molecular Dynamics derived parameters from Lankas et al. [1](#lank03)




1. <a name="lank03"></a> F. Lankaš, Jiří Šponer, Jörg Langowski, Thomas E. Cheatham, III, [DNA basepair step deformability inferred from molecular dynamics simulations](https://doi.org/10.1016/S0006-3495(03)74710-9). *Biophys. J*, 85, 2872 (2003).
