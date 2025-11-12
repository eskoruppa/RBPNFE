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

nout = nfe.eval(
    seq,
    shl_open_left = shl_open_left,
    shl_open_right = shl_open_right,
    use_correction = True
    )
```

### Arguments <a name=args></a>

- `params_model`:
    Select the elastic model for the generation of stiffness and structure parameters:
    - `md`: Molecular Dynamics derived parameters from Lankas et al. \[[1](#lank03)\]
    - `crystal`: Parameters from crystallographic data from Olson et al. \[[2](#olson98)\]
    - `cgna+`: Parameters derived from cgNA+ via marginalization to rigid base pair model \[[3](#sharma23)\]


\[1\] <a name="lank03"></a> F. Lankaš, Jiří Šponer, Jörg Langowski, Thomas E. Cheatham, III, DNA basepair step deformability inferred from molecular dynamics simulations, [Biophys. J, **85**, 2872 (2003)](https://doi.org/10.1016/S0006-3495(03)74710-9).

\[2\] <a name="olson98"></a> W. K. Olson, A. A. Gorin, X. Lu, L. M. Hock, and V. B. Zhurkin, DNA sequence-dependent deformability deduced from protein–DNA crystal complexes, [Proc. Natl. Acad. Sci. U.S.A. **95**, 11163 (1998).](https://doi.org/10.1073/pnas.95.19.11163).

\[3\] <a name="sharma23"></a> R. Sharma, A. S. Patelli, L. de Bruin, and J. H. Maddocks, cgNA+web: A visual interface to the cgNA+ sequence-dependent statistical mechanics model of double-stranded nucleic acids, [J. Mol. Biol. **435**,
167978 (2023).](http://dx.doi.org/10.1016/j.jmb.2023.167978).
