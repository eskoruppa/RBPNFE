"""
Multiharmonic nucleosome free-energy model.

Combines a nucleosome parameter set with a separate free-DNA parameter set. For a
given breathing state the 147 bp is split at the outermost bound binding sites
into the wrapped region and the two dangling arms, and the free energies are
recombined as

    F             = F_nuc + F_arm
    F_freedna     = F_bound + F_arm
    F_fluctuation = F - F_enthalpy

The arm term uses the Schur complement of the wrapped region rather than a raw
slice of the stiffness matrix. For block-diagonal parameter sets (crystal, md,
hybrid) the correction vanishes identically and this reduces to a plain slice;
for cgNA+, whose stiffness couples base-pair steps at long range, it is what
makes ``F_bound + F_arm`` equal the whole-sequence free energy.

Two properties of the original production model are preserved deliberately:

* ``F = F_nuc + F_arm`` adds arm DNA on top of an ``F_nuc`` that already spans
  all 147 bp.
* When the two parameter sets differ, ``dF`` differences two force fields.

Neither is a bug introduced here; both match the pipeline implementation this
module replaces.

Run the worked examples at the foot of this file with::

    python -m rbpnfe.multiharmonic

from the repository root. This module uses relative imports, so launching it as
a plain script (``python rbpnfe/multiharmonic.py``) leaves it without a parent
package and every ``from .`` line fails. The same applies to ``free_energy.py``
and the other modules here.
"""

from __future__ import annotations

# Thread pinning for the numerical backends. The variables are actually set by
# rbpnfe/__init__.py, which imports this module on its first line — BLAS reads
# them only when it is first loaded, so by the time this file runs numpy is
# already up. Imported here so the dependency is visible at the point of use and
# so `active_settings()` is available for logging.
from . import env_settings  # noqa: F401

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as sla

from .free_energy import NucFreeEnergy

_LOG2PI = float(np.log(2.0 * np.pi))

_STYLES = ("b_index", "ph_index", "open_sites")


@dataclass(frozen=True)
class MultiharmonicResult:
    """
    Free-energy components in kT.

    Supports attribute access (``res.F``), mapping access (``res['F']``), and
    exposes ``F_entropy`` as an alias of ``F_fluctuation`` for callers written
    against the older pipeline naming.
    """

    F: float
    F_fluctuation: float
    F_enthalpy: float
    F_freedna: float
    dF: float
    id: Optional[str] = None
    subid: Optional[str] = None

    @property
    def F_entropy(self) -> float:
        """Alias of :attr:`F_fluctuation` (legacy pipeline name)."""
        return self.F_fluctuation

    def keys(self) -> Tuple[str, ...]:
        return (
            "F", "F_fluctuation", "F_entropy", "F_enthalpy",
            "F_freedna", "dF", "id", "subid",
        )

    def __getitem__(self, key: str):
        if key not in self.keys():
            raise KeyError(key)
        return getattr(self, key)


def total_states_index(length: int) -> List[Tuple[int, int]]:
    """
    All (left, right) index states with ``0 <= left <= right < length``.

    Used by the 'b_index' (length 14) and 'ph_index' (length 28) styles, where
    left and right are positions counted inwards from each end.
    """
    return [(left, right)
            for left in range(length)
            for right in range(left, length)]


def total_open_states(length: int) -> List[Tuple[int, int]]:
    """
    All (left, right) open-site counts with ``left + right <= length``.

    Used by the 'open_sites' style, where left and right count *open* sites.
    """
    return [(left, right)
            for left in range(length + 1)
            for right in range(length + 1)
            if left + right <= length]


def get_left_right_open(left: int, right: int, style: str) -> Tuple[int, int]:
    """Translate a style-specific (left, right) state into open-site counts."""
    if style == "b_index":
        return 2 * left, 28 - 2 * right - 2
    if style == "ph_index":
        return left, 28 - right - 1
    if style == "open_sites":
        return left, right
    raise ValueError(f'Unknown style "{style}". Use one of {_STYLES}.')


def style_states(style: str) -> List[Tuple[int, int]]:
    """Every state belonging to ``style``."""
    if style == "b_index":
        return total_states_index(length=14)
    if style == "ph_index":
        return total_states_index(length=28)
    if style == "open_sites":
        return total_open_states(length=28)
    raise ValueError(f'Unknown style "{style}". Use one of {_STYLES}.')


def _logdet_spd(cf) -> float:
    """log|det| of an SPD matrix from its ``scipy.linalg.cho_factor`` output."""
    return 2.0 * float(np.sum(np.log(np.abs(np.diag(cf[0])))))


def whole_free_energy(K: np.ndarray) -> float:
    """Free energy of the full matrix, ``½ logdet K - ½ n log 2π``."""
    cf = sla.cho_factor(K, check_finite=False)
    return 0.5 * _logdet_spd(cf) - 0.5 * K.shape[0] * _LOG2PI


def split_free_energy(K: np.ndarray, start: int, end: int) -> Tuple[float, float]:
    """
    Split a stiffness matrix into a bound block and the two flanking arms.

    ``A`` is the contiguous bound range ``[start, end)``; ``B`` is everything
    outside it, i.e. the two dangling arms ``[0, start) ∪ [end, N)``.

    The arms are the plain slice ``K[B, B]``; the bound block takes the Schur
    complement

        K[A, A] - K[A, B] K[B, B]⁻¹ K[B, A]

    i.e. the wrapped region with the arms *integrated out*. Together these are
    the exact chain-rule factorisation

        logdet K = logdet K[B, B] + logdet(schur_A)

    so the two returned energies always sum to :func:`whole_free_energy`.

    Why this way round, and not the other
    -------------------------------------
    Both assignments are exactly additive, so additivity alone cannot choose
    between them. What chooses is the nucleosome calculation itself: the binding
    constraints act only on the wrapped region ``A``, so the operation that
    factorises the constrained partition function is integrating out the
    *arms*. That leaves the arms as the plain slice and puts the Schur
    complement on the wrapped block. With this convention

        F_nuc(full 147 bp) - F_arm(slice) == F_wrapped(arms marginalised out)

    holds to machine precision for cgNA+ (measured: 2e-12 kT over 65
    sequence/state pairs spanning GC 0.20-0.80 and 8-26 bound sites). Putting
    the Schur complement on the arms instead breaks that identity by ~1.5 kT,
    state-dependently.

    Only one of the two blocks may take the Schur complement; applying it to
    both would subtract the coupling twice.

    For block-diagonal stiffness (crystal, md, hybrid) ``K[A, B]`` is zero, the
    correction term vanishes, and both blocks are exactly raw slices. The zero
    check below makes that path free as well as exact.

    Returns ``(F_bound, F_arm)`` in kT. ``F_arm`` is ``0.0`` when there are no
    arms, in which case ``F_bound`` carries the whole matrix.
    """
    n = K.shape[0]
    if not 0 <= start < end <= n:
        raise ValueError(
            f"Invalid bound range [{start}, {end}) for a matrix of size {n}"
        )

    arm_idx = np.r_[0:start, end:n]
    if arm_idx.size == 0:
        return whole_free_energy(K), 0.0

    K_BB = K[np.ix_(arm_idx, arm_idx)]
    cB = sla.cho_factor(K_BB, check_finite=False)
    f_arm = 0.5 * _logdet_spd(cB) - 0.5 * K_BB.shape[0] * _LOG2PI

    K_AA = K[start:end, start:end]
    K_AB = K[np.ix_(np.arange(start, end), arm_idx)]
    if np.any(K_AB):
        K_AA = K_AA - K_AB @ sla.cho_solve(cB, K_AB.T, check_finite=False)
        K_AA = 0.5 * (K_AA + K_AA.T)

    cA = sla.cho_factor(K_AA, check_finite=False)
    f_bound = 0.5 * _logdet_spd(cA) - 0.5 * K_AA.shape[0] * _LOG2PI
    return f_bound, f_arm


class NucleosomeBreath:
    """
    Nucleosome breathing free energies with an integrated multiharmonic
    free-DNA term.

    Composes two :class:`~rbpnfe.free_energy.NucFreeEnergy` instances: one for
    the nucleosome parameter set and one for the free-DNA reference. All
    parameter generation — cgNA+ flanking, ``cgnaplus_setname``,
    ``rescale_factors`` — is delegated to them, so every model rbpnfe supports
    works here.

    Set ``free_dna_method`` to enable the multiharmonic path. Left as ``None``
    the wrapper is a thin pass-through over ``NucFreeEnergy``.

    Parameters
    ----------
    nuc_method
        Elastic model for the nucleosome-bound DNA.
    free_dna_method
        Elastic model for the free-DNA reference. ``None`` disables the
        multiharmonic recombination.
    hardconstraint
        Default binding model. ``calculate_free_energy_soft`` and
        ``calculate_free_energy_hard`` override it per call.
    rescale_factors
        Optional per-DOF stiffness rescaling ``[tilt, roll, twist, shift,
        slide, rise]``, applied to the nucleosome parameters only.
    cgnaplus_setname
        cgNA+ parameter set name; used only by cgNA+ models.
    flanking
        Flanking base pairs added during parameter generation.
    """

    def __init__(
        self,
        nuc_method: str = "crystal",
        free_dna_method: Optional[str] = None,
        hardconstraint: bool = False,
        rescale_factors: Optional[Sequence[float]] = None,
        cgnaplus_setname: str = "curves_plus",
        flanking: int = 10,
    ):
        self.nuc_method = nuc_method
        self.free_dna_method = free_dna_method
        self.hardconstraint = hardconstraint

        self._nfe_nuc = NucFreeEnergy(
            params_model=nuc_method,
            hardconstraint=hardconstraint,
            rescale_factors=rescale_factors,
            cgnaplus_setname=cgnaplus_setname,
            flanking=flanking,
        )
        self._nfe_free = None
        if free_dna_method is not None:
            self._nfe_free = NucFreeEnergy(
                params_model=free_dna_method,
                cgnaplus_setname=cgnaplus_setname,
                flanking=flanking,
            )

        self.midstep_locations = np.asarray(
            self._nfe_nuc.midstep_locations, dtype=int
        )
        # kept pristine so kresc_factor never accumulates across calls
        self._Kmat_base = np.array(self._nfe_nuc.Kmat, copy=True)

        self._cached_seq: Optional[str] = None
        self._nuc_gs: Optional[np.ndarray] = None
        self._nuc_stiff: Optional[np.ndarray] = None
        self._free_stiff: Optional[np.ndarray] = None
        self._free_tables: Dict[str, Dict[Tuple[int, int], Tuple[float, float]]] = {}

    # ------------------------------------------------------------------ caches

    def _ensure_sequence_cache(self, seq: str) -> None:
        """Generate and cache stiffness parameters for ``seq`` if not current."""
        if seq == self._cached_seq:
            return
        if len(seq) != 147:
            raise ValueError(
                f"Sequence must be 147 bp long, got {len(seq)}"
            )

        gs, stiff = self._nfe_nuc.gen_params(seq, flanking=self._nfe_nuc.flanking)
        self._nuc_gs = gs
        self._nuc_stiff = np.asarray(stiff)

        self._free_stiff = None
        self._free_tables = {}
        if self._nfe_free is not None:
            _, fstiff = self._nfe_free.gen_params(
                seq, flanking=self._nfe_free.flanking
            )
            self._free_stiff = np.asarray(fstiff)

        self._cached_seq = seq

    def _state_bounds(self, l_open: int, r_open: int) -> Tuple[int, int]:
        """Row range of the bound region for the given open-site counts."""
        locs = self.midstep_locations[l_open:self.midstep_locations.size - r_open]
        if locs.size == 0:
            raise ValueError(
                f"No bound binding sites remain for open counts "
                f"({l_open}, {r_open})"
            )
        return 6 * int(locs[0]), 6 * (int(locs[-1]) + 1)

    def _free_dna_table(
        self, style: str
    ) -> Dict[Tuple[int, int], Tuple[float, float, float]]:
        """
        ``{(left, right): (F_arm_nuc, F_bound_free, F_arm_free)}``.

        Three terms per state, because the recombination needs the arms under
        *both* parameter sets: the nucleosome-parameter arms are subtracted off
        the full nucleosome calculation, and the free-DNA-parameter arms are
        added back in their place.
        """
        table = self._free_tables.get(style)
        if table is not None:
            return table
        if self._free_stiff is None:
            raise RuntimeError(
                "Free-DNA stiffness is not cached; call _ensure_sequence_cache first."
            )

        table = {}
        for left, right in style_states(style):
            l_open, r_open = get_left_right_open(left, right, style)
            if l_open + r_open >= self.midstep_locations.size:
                continue
            start, end = self._state_bounds(l_open, r_open)
            _, f_arm_nuc = split_free_energy(self._nuc_stiff, start, end)
            f_bound_free, f_arm_free = split_free_energy(self._free_stiff, start, end)
            table[(left, right)] = (f_arm_nuc, f_bound_free, f_arm_free)

        self._free_tables[style] = table
        return table

    # ------------------------------------------------------------- evaluation

    def _eval_nuc(
        self,
        l_open: int,
        r_open: int,
        hard: bool,
        kresc_factor: float,
    ) -> dict:
        """Run the underlying nucleosome model for one open-site configuration."""
        self._nfe_nuc.hardconstraint = hard
        self._nfe_nuc.Kmat = (
            self._Kmat_base if kresc_factor == 1.0
            else self._Kmat_base * kresc_factor
        )
        return self._nfe_nuc._eval_single(
            self._nuc_gs,
            self._nuc_stiff,
            open_left=l_open,
            open_right=r_open,
            use_correction=True,
        )

    @staticmethod
    def _model_freedna(fdict: dict) -> float:
        """
        Free-DNA reference from an underlying model result.

        The backends disagree on the key name: the soft-constraint model and the
        'legacy' hard-constraint backend return ``F_freedna``, while the
        'compse3' hard-constraint backend returns ``F_free``. Both name the same
        quantity.
        """
        if "F_freedna" in fdict:
            return fdict["F_freedna"]
        if "F_free" in fdict:
            return fdict["F_free"]
        raise KeyError(
            "Model result carries neither 'F_freedna' nor 'F_free'; "
            f"got keys {sorted(fdict)}"
        )

    def _combine(
        self,
        fdict: dict,
        left: int,
        right: int,
        style: str,
        id: Optional[str],
        subid: Optional[str],
    ) -> MultiharmonicResult:
        """Apply the multiharmonic recombination, or pass through without it."""
        if self._nfe_free is None:
            f_freedna = self._model_freedna(fdict)
            return MultiharmonicResult(
                F=fdict["F"],
                F_fluctuation=fdict["F_fluctuation"],
                F_enthalpy=fdict["F_enthalpy"],
                F_freedna=f_freedna,
                dF=fdict["F"] - f_freedna,
                id=id,
                subid=subid,
            )

        # Two parameter sets, one molecule. The nucleosome model has been run
        # over the whole 147 bp under `nuc_method`, so it already contains the
        # dangling arms -- but under the wrong parameter set. Swap them:
        #
        #     F = F_nuc(full, nuc) - F_arm(nuc) + F_arm(free)
        #
        # Because the binding constraints act only on the wrapped region, the
        # first subtraction is exact: F_nuc(full) - F_arm(slice) is precisely
        # the wrapped region with the arms integrated out. See
        # `split_free_energy` for why the arms must be the plain slice.
        #
        # The reference is the same molecule as free DNA under `free_method`,
        # split the same way, so F_freedna is state-independent by construction
        # (asserted in the tests).
        f_arm_nuc, f_bound_free, f_arm_free = self._free_dna_table(style)[(left, right)]
        f_total = fdict["F"] - f_arm_nuc + f_arm_free
        f_freedna = f_bound_free + f_arm_free
        f_enthalpy = fdict["F_enthalpy"]
        return MultiharmonicResult(
            F=f_total,
            F_fluctuation=f_total - f_enthalpy,
            F_enthalpy=f_enthalpy,
            F_freedna=f_freedna,
            dF=f_total - f_freedna,
            id=id,
            subid=subid,
        )

    def calculate_free_energy_soft(
        self,
        seq601: str,
        left: int,
        right: int,
        id: Optional[str] = None,
        subid: Optional[str] = None,
        kresc_factor: float = 1.0,
        style: str = "b_index",
    ) -> MultiharmonicResult:
        """
        Soft-constraint free energy for one breathing state.

        ``seq601`` is a 147 bp sequence; the name is retained from the pipeline
        API. ``left``/``right`` are interpreted according to ``style``.
        ``kresc_factor`` scales the binding-site stiffness matrix.
        """
        self._ensure_sequence_cache(seq601)
        l_open, r_open = get_left_right_open(left, right, style)
        fdict = self._eval_nuc(l_open, r_open, hard=False, kresc_factor=kresc_factor)
        return self._combine(fdict, left, right, style, id, subid)

    def calculate_free_energy_hard(
        self,
        seq147: str,
        left: int,
        right: int,
        id: Optional[str] = None,
        subid: Optional[str] = None,
        style: str = "b_index",
    ) -> MultiharmonicResult:
        """
        Hard-constraint free energy for one breathing state.

        ``seq147`` is a 147 bp sequence; the name is retained from the pipeline
        API. The same multiharmonic recombination is applied as in
        :meth:`calculate_free_energy_soft`. The hard-constraint model has no
        binding-site stiffness matrix, so there is no ``kresc_factor``.
        """
        self._ensure_sequence_cache(seq147)
        l_open, r_open = get_left_right_open(left, right, style)
        fdict = self._eval_nuc(l_open, r_open, hard=True, kresc_factor=1.0)
        return self._combine(fdict, left, right, style, id, subid)

    def calculate_free_energy_soft_batch(
        self,
        seq601: str,
        states: Sequence[Tuple[int, int]],
        id: Optional[str] = None,
        subid: Optional[str] = None,
        kresc_factor: float = 1.0,
        style: str = "b_index",
    ) -> List[MultiharmonicResult]:
        """
        Soft-constraint free energies for many breathing states of one sequence.

        Parameter generation and the free-DNA state table are computed once and
        reused across every state, which is the reason to prefer this over a
        loop of :meth:`calculate_free_energy_soft` calls.
        """
        if not states:
            return []

        self._ensure_sequence_cache(seq601)
        if self._nfe_free is not None:
            self._free_dna_table(style)   # build once, then hit the cache

        results: List[MultiharmonicResult] = []
        for left, right in states:
            l_open, r_open = get_left_right_open(left, right, style)
            fdict = self._eval_nuc(
                l_open, r_open, hard=False, kresc_factor=kresc_factor
            )
            results.append(self._combine(fdict, left, right, style, id, subid))
        return results


if __name__ == '__main__':

    seq601 = (
        "CTGGAGAATCCCGGTGCCGAGGCCGCTCAATTGGTCGTAGACAGCTCTAGCACCGCTTAAACGCACGT"
        "ACGCGCTGTCCCCCGCGTTTTAACCGCCAAGGGGATTACTCCCTAGTCTCCAGGCACGTGTCAGATAT"
        "ATACATCCTGT"
    )

    def show(title, res):
        print(f'  {title:<34} F = {res.F:10.2f}   dF = {res.dF:8.2f}   '
              f'F_freedna = {res.F_freedna:9.2f}')

    # ------------------------------------------------------------------
    # 1. Single parameter set. With free_dna_method left as None this is a
    #    thin pass-through over NucFreeEnergy -- the free-DNA reference is
    #    whatever the underlying model reports.
    # ------------------------------------------------------------------
    print('\n1. single parameter set (no multiharmonic recombination)')
    for model in ('crystal', 'md'):
        nb = NucleosomeBreath(nuc_method=model)
        show(f'nuc={model}', nb.calculate_free_energy_soft(seq601, 0, 13))

    # ------------------------------------------------------------------
    # 2. Two parameter sets. The nucleosome is evaluated under nuc_method;
    #    the dangling arms and the free-DNA reference come from
    #    free_dna_method. dF now compares two force fields, which is the
    #    point of the model -- it is not a single-force-field free energy
    #    difference.
    # ------------------------------------------------------------------
    print('\n2. multiharmonic, local parameter sets')
    for nuc, free in (('crystal', 'md'), ('md', 'crystal'), ('hybrid', 'md')):
        nb = NucleosomeBreath(nuc_method=nuc, free_dna_method=free)
        show(f'nuc={nuc}, free={free}',
             nb.calculate_free_energy_soft(seq601, 0, 13))

    # ------------------------------------------------------------------
    # 3. cgNA+ anywhere in the pair. cgNA+ couples base-pair steps out to
    #    ~30-40 steps, so the wrapped/arm split is done with a Schur
    #    complement rather than a raw slice. Slower: parameter generation
    #    dominates, so reuse one instance across states where possible.
    # ------------------------------------------------------------------
    print('\n3. multiharmonic with cgNA+ (slower)')
    for nuc, free in (('cgnaplus', 'md'), ('crystal', 'cgnaplus'),
                      ('cgnaplus', 'cgnaplus')):
        nb = NucleosomeBreath(nuc_method=nuc, free_dna_method=free)
        show(f'nuc={nuc}, free={free}',
             nb.calculate_free_energy_soft(seq601, 0, 13))

    # ------------------------------------------------------------------
    # 4. Breathing landscape. Use the batch call rather than a loop of
    #    single calls: parameter generation and the free-DNA state table
    #    are built once and reused across every state.
    # ------------------------------------------------------------------
    print('\n4. symmetric unwrapping, nuc=crystal free=md')
    nb = NucleosomeBreath(nuc_method='crystal', free_dna_method='md')
    states = [(i, 13 - i) for i in range(7)]
    for (left, right), res in zip(states,
                                  nb.calculate_free_energy_soft_batch(seq601, states)):
        n_bound = 28 - 2 * left - (28 - 2 * right - 2)
        print(f'  state {str((left, right)):<9} {n_bound:2d} bound sites   '
              f'F = {res.F:10.2f}   dF = {res.dF:7.2f}')

    # every state of the 'b_index' style in one call
    every = nb.calculate_free_energy_soft_batch(seq601, style_states('b_index'))
    print(f'  ...{len(every)} states total, '
          f'dF from {min(r.dF for r in every):.2f} to '
          f'{max(r.dF for r in every):.2f} kT')

    # ------------------------------------------------------------------
    # 5. Hard constraints, and the id/subid passthrough used downstream.
    # ------------------------------------------------------------------
    print('\n5. hard constraint, with record labels')
    nb = NucleosomeBreath(nuc_method='crystal', free_dna_method='md')
    res = nb.calculate_free_energy_hard(seq601, 0, 13, id='seq0001', subid='rep1')
    print(f'  id={res.id} subid={res.subid}   F = {res.F:.2f}   dF = {res.dF:.2f}')
    print(f'  mapping access: res["F_enthalpy"] = {res["F_enthalpy"]:.2f}, '
          f'F_entropy alias = {res.F_entropy:.2f}')

    # ------------------------------------------------------------------
    # 6. Sanity check. With both parameter sets identical the arm swap
    #    cancels term by term, so the result must reproduce the plain
    #    single-parameter model exactly.
    # ------------------------------------------------------------------
    print('\n6. degenerate limit (nuc == free) reproduces the plain model')
    for model in ('crystal', 'cgnaplus'):
        multi = NucleosomeBreath(nuc_method=model, free_dna_method=model)
        plain = NucleosomeBreath(nuc_method=model)
        worst = max(
            abs(a.F - b.F)
            for a, b in zip(
                multi.calculate_free_energy_soft_batch(seq601, states),
                plain.calculate_free_energy_soft_batch(seq601, states))
        )
        print(f'  {model:<10} max |multiharmonic - plain| = {worst:.2e} kT')
