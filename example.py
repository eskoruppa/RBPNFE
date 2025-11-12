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

print(f'Full Free Energy:         {nout["F"]:.2f} kT')
print(f'Fluctuation Contribution: {nout["F_fluctuation"]:.2f} kT')
print(f'Enthalpic Contribution:   {nout["F_enthalpy"]:.2f} kT')


