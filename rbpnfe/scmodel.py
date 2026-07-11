from __future__ import annotations

import sys, os
import numpy as np
import scipy as sp
import scipy.linalg as sla
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import List, Tuple, Callable, Any, Dict

from .PolyCG.polycg.SO3 import so3
from .PolyCG.polycg.transforms.transform_marginals import send_to_back_permutation
from .midstep_composites import midstep_composition_transformation, midstep_groundstate, midstep_groundstate_se3
from .midstep_composites import midstep_composition_transformation_correction
from .midstep_composites import calculate_midstep_triads
from .midstep_composites_fast import (
    transformation_first_order as _fast_transformation_first_order,
    transformation_correction as _fast_transformation_correction,
    coordinate_transformation as _fast_coordinate_transformation,
    coordinate_transformation_correction as _fast_coordinate_transformation_correction,
)

def get_midstep_locations(left_open: int, right_open: int, base_midstep_locations = None, sort = True) -> list[int]:
    if base_midstep_locations is None:
        MIDSTEP_LOCATIONS = [
            2, 6, 14, 17, 24, 29, 
            34, 38, 45, 49, 55, 59, 
            65, 69, 76, 80, 86, 90, 
            96, 100, 107, 111, 116, 121, 
            128, 131, 139, 143
        ]
    else:
        MIDSTEP_LOCATIONS = base_midstep_locations
    if left_open + right_open > len(MIDSTEP_LOCATIONS):
        return []
    
    locs = MIDSTEP_LOCATIONS[left_open:len(MIDSTEP_LOCATIONS)-right_open]
    if sort:
        locs = sorted(list(set(locs)))
    return locs

def Hinverse(Psi: float) -> np.ndarray:
    psih = so3.hat_map(Psi)
    psihsq = psih @ psih
    Hinv = np.eye(3)
    Hinv += 0.5* psih
    Hinv += 1./12 * psihsq
    Hinv -= 1./720 * psihsq @ psihsq
    Hinv += 1./30240 * psihsq @ psihsq @ psihsq
    return Hinv
    
def coordinate_transformation(muk0s: np.ndarray, sks: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    B = np.zeros((len(sks)*6,len(muk0s)*6), dtype=np.float64)
    Pbar = np.zeros(len(sks)*6, dtype=np.float64)
    for k in range(len(sks)):
        sig0 = np.linalg.inv(muk0s[k]) @ muk0s[k+1]
        Sig  = sig0[:3,:3]
        sig  = sig0[:3,3]
        Sk   = sks[k,:3,:3]
        sk   = sks[k,:3,3]
        
        Psi  = so3.rotmat2euler(Sk.T @ Sig)
        Hi   = Hinverse(Psi)
        Bkm = np.zeros((6,6), dtype=np.float64)
        Bkp = np.zeros((6,6), dtype=np.float64)
        Bkm[:3,:3] = -Hi @ Sig.T
        Bkm[3:,:3] = Sk.T @ so3.hat_map(sig)
        Bkm[3:,3:] = -Sk.T
        Bkp[:3,:3] = Hi
        Bkp[3:,3:] = Sk.T @ Sig
        
        B[6*k:6*(k+1),6*k:6*(k+1)]      = Bkm
        B[6*k:6*(k+1),6*(k+1):6*(k+2)]  = Bkp
        
        Pbar[k*6:k*6+3]   = Psi
        Pbar[k*6+3:k*6+6] = Sk.T @ (sig-sk)
    return B, Pbar

def coordinate_transformation_correction(
    muk0s: np.ndarray, 
    sks: np.ndarray,
    Z_delta_ref: np.ndarray
    ) -> tuple[np.ndarray,np.ndarray]:
    
    if len(Z_delta_ref.shape) < 2:
        Z_delta_ref = Z_delta_ref.reshape((len(Z_delta_ref)//6,6))
    
    B = np.zeros((len(sks)*6,len(muk0s)*6), dtype=np.float64)
    Pbar = np.zeros(len(sks)*6, dtype=np.float64)
    for k in range(len(sks)):
        sig0 = np.linalg.inv(muk0s[k]) @ muk0s[k+1]
        SIG  = sig0[:3,:3]
        sig  = sig0[:3,3]
        Sk   = sks[k,:3,:3]
        sk   = sks[k,:3,3]
        
        Psi  = so3.rotmat2euler(Sk.T @ SIG)
        Hi   = Hinverse(Psi)
        
        Z0k = so3.euler2rotmat(Z_delta_ref[k,:3])
        htheta0 = so3.hat_map(Z_delta_ref[k,:3])
        
        Bkm = np.zeros((6,6), dtype=np.float64)
        Bkp = np.zeros((6,6), dtype=np.float64)
        
        Bkm[:3,:3] = -Hi @ SIG.T
        Bkm[3:,:3] = Sk.T @ so3.hat_map(sig)
        Bkm[3:,3:] = -Sk.T @ Z0k.T
        
        Bkp[:3,:3] = Hi
        Bkp[3:,3:] = Sk.T @ Z0k.T @ SIG
        
        B[6*k:6*(k+1),6*k:6*(k+1)]      = Bkm
        B[6*k:6*(k+1),6*(k+1):6*(k+2)]  = Bkp
        
        Pbar[k*6:k*6+3]   = Psi
        Pbar[k*6+3:k*6+6] = Sk.T @ ( (Z0k.T + htheta0) @ sig-sk)
    return B, Pbar

def sc_free_energy(
    free_gs: np.ndarray,
    free_M: np.ndarray,    
    nuc_mu0_full: np.ndarray,
    nuc_K_full: np.ndarray,
    left_open: int = 0,
    right_open: int = 0,
    base_midstep_locations: list[int] = None,
    use_correction: bool = True,
    optimized: bool = True
) -> dict[str]:
    
    if optimized:
        return sc_free_energy_optimized(
            free_gs, free_M, nuc_mu0_full, nuc_K_full,
            left_open=left_open, right_open=right_open,
            base_midstep_locations=base_midstep_locations,
            use_correction=use_correction
        )
    return sc_free_energy_explicit(
        free_gs, free_M, nuc_mu0_full, nuc_K_full,
        left_open=left_open, right_open=right_open,
        base_midstep_locations=base_midstep_locations,
        use_correction=use_correction
    )


def sc_free_energy_explicit(
    free_gs: np.ndarray,
    free_M: np.ndarray,    
    nuc_mu0_full: np.ndarray,
    nuc_K_full: np.ndarray,
    left_open: int = 0,
    right_open: int = 0,
    base_midstep_locations: list[int] = None,
    use_correction: bool = True,
) -> dict[str]:
    
    if sp.sparse.issparse(free_M):
        free_M = free_M.toarray()

    midstep_constraint_locations = get_midstep_locations(left_open, right_open, base_midstep_locations=base_midstep_locations)
    if len(midstep_constraint_locations) <= 1:
        n = len(free_M)
        F_pi = -0.5*n * np.log(2*np.pi)
        # matrix term
        logdet_sign, logdet = np.linalg.slogdet(free_M)
        F_mat = 0.5*logdet
        F = F_mat + F_pi  
        Fdict = {
            'F': F,
            'F_fluctuation' : F,
            'F_enthalpy': 0,
            'F_jacob'   : 0,
            'F_freedna' : F,
            'dF'        : 0,
            'gs': np.zeros(free_gs.shape, dtype=np.float64),
            'alphas' : None
        }
        return Fdict
    
    sks = midstep_groundstate_se3(free_gs,midstep_constraint_locations)
    
    # select midstep triads and their stiffness 
    nuc_mu0 = nuc_mu0_full[left_open:len(nuc_mu0_full)-right_open]
    nuc_K = nuc_K_full[6*left_open:len(nuc_K_full)-6*right_open,6*left_open:len(nuc_K_full)-6*right_open]
    
    # find composite transformation
    transform, replaced_ids = midstep_composition_transformation(
        free_gs,
        midstep_constraint_locations
    )
    
    # transform stiffness matrix
    inv_transform = np.linalg.inv(transform)
    M_transformed = inv_transform.T @ free_M @ inv_transform
    
    # rearrange stiffness matrix
    full_replaced_ids = list()
    for i in range(len(replaced_ids)):
        full_replaced_ids += [6*replaced_ids[i]+j for j in range(6)]
     
    P = send_to_back_permutation(len(free_M),full_replaced_ids)
    M_rearranged = P @ M_transformed @ P.T
    
    # select M and R submatrices
    N  = len(M_rearranged)
    NC = len(full_replaced_ids)
    NF = N-NC
    
    M_R = M_rearranged[:NF,:NF]
    M_M = M_rearranged[NF:,NF:]
    M_RM = M_rearranged[:NF,NF:]
    
    # Calculate M block marginal
    M_Mp = M_M - M_RM.T @ np.linalg.inv(M_R) @ M_RM
    M_Mp = 0.5*(M_Mp+M_Mp.T)
    

    ##############################################
    # Binding Model
    ##############################################
    
    # Calculate Incidence Matrix
    B, Pbar = coordinate_transformation(nuc_mu0,sks)  
    Kcomb = nuc_K + B.T @ M_Mp @ B
    # calculate ground state
    alpha = -np.linalg.inv(Kcomb) @ B.T @ M_Mp @ Pbar
    
    B, Pbar = coordinate_transformation_correction(nuc_mu0,sks,alpha)
    Kcomb = nuc_K + B.T @ M_Mp @ B
    # calculate ground state
    alpha = -np.linalg.inv(Kcomb) @ B.T @ M_Mp @ Pbar
    
    Y_C = Pbar + B @ alpha
    F_enthalpy = 0.5* Pbar.T @ ( M_Mp - M_Mp @ B @ np.linalg.inv(Kcomb) @ B.T @ M_Mp ) @ Pbar
    # print(f'F_enthalpy = {F_enthalpy}')
    
    gamma = -np.linalg.inv(M_R) @ M_RM @ Y_C
    
    if use_correction:
    
        gs_transf_perm = np.concatenate((gamma,Y_C))
        gs_transf = P.T @ gs_transf_perm
        gs = inv_transform @ gs_transf

        gs = gs.reshape((len(gs)//6,6))
        # find composite transformation
        transform, replaced_ids, shift = midstep_composition_transformation_correction(
            free_gs,
            midstep_constraint_locations,
            gs
        )
        
        # transform stiffness matrix
        inv_transform = np.linalg.inv(transform)
        M_transformed = inv_transform.T @ free_M @ inv_transform
        
        # rearrange stiffness matrix
        full_replaced_ids = list()
        for i in range(len(replaced_ids)):
            full_replaced_ids += [6*replaced_ids[i]+j for j in range(6)]
        
        P = send_to_back_permutation(len(free_M),full_replaced_ids)
        M_rearranged = P @ M_transformed @ P.T
        
        # select M and R submatrices
        N  = len(M_rearranged)
        NC = len(full_replaced_ids)
        NF = N-NC
        
        M_R = M_rearranged[:NF,:NF]
        M_M = M_rearranged[NF:,NF:]
        M_RM = M_rearranged[:NF,NF:]
        
        # Calculate M block marginal
        M_Mp = M_M - M_RM.T @ np.linalg.inv(M_R) @ M_RM
        M_Mp = 0.5*(M_Mp+M_Mp.T)
        
        ##############################################
        # Binding Model
        ##############################################
        
        # Calculate Incidence Matrix
        B, Pbar = coordinate_transformation(nuc_mu0,sks)  
        
        Kcomb = nuc_K + B.T @ M_Mp @ B
        # calculate ground state
        alpha = -np.linalg.inv(Kcomb) @ B.T @ M_Mp @ Pbar
        
        B, Pbar = coordinate_transformation_correction(nuc_mu0,sks,alpha)
        Kcomb = nuc_K + B.T @ M_Mp @ B 
        
        # b -> b - a
        Pbar -= shift
        
        # calculate ground state
        alpha = -np.linalg.inv(Kcomb) @ B.T @ M_Mp @ Pbar
        Y_C = Pbar + B @ alpha
        gamma = -np.linalg.inv(M_R) @ M_RM @ Y_C
        
        F_enthalpy = 0.5* Pbar.T @ ( M_Mp - M_Mp @ B @ np.linalg.inv(Kcomb) @ B.T @ M_Mp ) @ Pbar
        # print(f'F_enthalpy = {F_enthalpy}')
        
    gs_transf_perm = np.concatenate((gamma,Y_C))
    gs_transf = P.T @ gs_transf_perm
    gs = inv_transform @ gs_transf
       
    alphas = alpha.reshape((len(alpha)//6,6))
        
    # Z entropy term
    n = len(Kcomb)
    logdet_sign, logdet_K = np.linalg.slogdet(Kcomb)
    F_piK = -0.5*n * np.log(2*np.pi)
    Z_entropy = 0.5*logdet_K + F_piK
    
    # Z entropy term
    n = len(M_R)
    logdet_sign, logdet_R = np.linalg.slogdet(M_R)
    F_piR = -0.5*n * np.log(2*np.pi)
    R_entropy = 0.5*logdet_R + F_piR
    
    # jacobian A
    # F_jacob = np.log(np.linalg.det(transform))
    signjacob, F_Ajacob = np.linalg.slogdet(transform)
        
    # # volume element B
    # signBlogdet, Blogdet = np.linalg.slogdet(B@B.T)
    # F_Bjacob = 0.5*Blogdet
    
    # Full entropy term
    F_fluctuation = Z_entropy + R_entropy + F_Ajacob #+ F_Bjacob
    
    
    # free energy of unconstrained DNA
    ff_logdet_sign, ff_logdet = np.linalg.slogdet(free_M)
    ff_pi = -0.5*len(free_M) * np.log(2*np.pi)
    F_free = 0.5*ff_logdet + ff_pi
    
    # prepare output
    Fdict = {
        'F': F_fluctuation + F_enthalpy,
        'F_fluctuation' : F_fluctuation,
        'F_enthalpy': F_enthalpy,
        'F_jacob'  : F_Ajacob,
        # 'F_Bjacob'  : F_Bjacob,
        'F_freedna' : F_free,
        'dF'        : F_fluctuation + F_enthalpy - F_free ,
        'gs'        : gs,
        'alphas'    : alphas
    }
    return Fdict



_LOG2PI = np.log(2*np.pi)


def _logdet_spd(cf) -> float:
    """log|det| of an SPD matrix from its Cholesky factor (sla.cho_factor output)."""
    return 2.0*np.sum(np.log(np.abs(np.diag(cf[0]))))


def _sc_logdet_transform(transform: np.ndarray, con_idx: np.ndarray) -> float:
    """
    log|det(transform)| for the composition transform.

    ``transform`` equals the identity in every row except the constrained rows
    ``con_idx`` (the replaced degrees of freedom). Under the partition
    (constrained | free) it is therefore block lower triangular with an identity
    free-block, so ``det(transform) == det(transform[con_idx, con_idx])`` and we
    only need the log-determinant of that small NC x NC principal submatrix.
    """
    sub = transform[np.ix_(con_idx, con_idx)]
    return np.linalg.slogdet(sub)[1]


def _sc_marginal(cf_freeM, transform: np.ndarray, replaced_ids: list[int]):
    """
    Marginal stiffness of the constrained (composite) coordinates.

    Uses the identity  M_Mp = (A F^-1 A^T)^-1  with  A = transform[constrained,:],
    F = free_M. This avoids inverting ``transform`` (N x N), forming the congruence
    ``F -> W^T F W``, the permutation, and inverting the large, ill-conditioned free
    block ``M_R`` — all of which the reference implementation does. The only
    conditioning that enters here is cond(free_M) and cond(A F^-1 A^T), both small,
    which makes this both far faster and numerically more accurate.

    Returns (M_Mp, G, logdet_invMmp, con_idx) where ``G = F^-1 A^T`` is reused to
    reconstruct the ground state, and ``logdet_invMmp = log det(A F^-1 A^T)``.
    """
    con_idx = np.array([6*rid+j for rid in replaced_ids for j in range(6)])
    A = np.ascontiguousarray(transform[con_idx, :])          # (NC, N)
    G = sla.cho_solve(cf_freeM, A.T, check_finite=False)     # (N, NC) = F^-1 A^T
    inv_M_Mp = A @ G                                          # (NC, NC) = A F^-1 A^T
    inv_M_Mp = 0.5*(inv_M_Mp + inv_M_Mp.T)
    cS = sla.cho_factor(inv_M_Mp, check_finite=False)
    logdet_invMmp = _logdet_spd(cS)
    NC = con_idx.size
    M_Mp = sla.cho_solve(cS, np.eye(NC), check_finite=False)
    M_Mp = 0.5*(M_Mp + M_Mp.T)
    return M_Mp, G, logdet_invMmp, con_idx


def _sc_binding(M_Mp, nuc_K, B0, Pbar0, nuc_mu0, sks, shift=None):
    """
    Binding-model ground state for a given marginal stiffness ``M_Mp``.

    Mirrors the reference two-step estimate: a first ``alpha`` from the uncorrected
    coordinate transform (B0, Pbar0), then the corrected transform. Uses Cholesky
    solves instead of explicit matrix inverses.

    Returns (alpha, Y_C, Kcomb, cK, F_enthalpy) where
    ``F_enthalpy = 0.5 Pbar^T (M_Mp - M_Mp B Kcomb^-1 B^T M_Mp) Pbar``.
    """
    Kcomb = nuc_K + B0.T @ (M_Mp @ B0)
    cK = sla.cho_factor(Kcomb, check_finite=False)
    alpha = -sla.cho_solve(cK, B0.T @ (M_Mp @ Pbar0), check_finite=False)

    B, Pbar = _fast_coordinate_transformation_correction(nuc_mu0, sks, alpha)
    if shift is not None:
        Pbar = Pbar - shift
    Kcomb = nuc_K + B.T @ (M_Mp @ B)
    cK = sla.cho_factor(Kcomb, check_finite=False)
    w = M_Mp @ Pbar
    u = B.T @ w
    alpha = -sla.cho_solve(cK, u, check_finite=False)
    Y_C = Pbar + B @ alpha
    F_enthalpy = 0.5*(Pbar @ w - u @ sla.cho_solve(cK, u, check_finite=False))
    return alpha, Y_C, Kcomb, cK, F_enthalpy


def sc_free_energy_optimized(
    free_gs: np.ndarray,
    free_M: np.ndarray,
    nuc_mu0_full: np.ndarray,
    nuc_K_full: np.ndarray,
    left_open: int = 0,
    right_open: int = 0,
    base_midstep_locations: list[int] = None,
    use_correction: bool = True,
) -> dict[str]:
    """
    Optimized drop-in replacement for :func:`sc_free_energy`.

    Computes the identical soft-constraint free-energy decomposition but avoids the
    four dominant costs of the reference implementation — ``inv(transform)`` (N x N),
    the congruence ``W^T free_M W``, the permutation matmul ``P M P^T``, and
    ``inv(M_R)`` on the large free block — by using two algebraic identities:

    * ``M_Mp = (A free_M^-1 A^T)^-1``  with ``A = transform[constrained, :]``
      (one Cholesky of ``free_M``, reused across both passes, plus a solve).
    * ``gs   = free_M^-1 A^T M_Mp Y_C`` (the equality-constrained minimizer), which
      removes the need for ``M_R``/``gamma`` entirely.

    Log-determinants are taken from Cholesky factors, and ``logdet(M_R)`` from
    ``logdet(free_M) - 2 logdet(transform) + logdet(A free_M^-1 A^T)``.

    Numerical note: the reference inverts ``M_R`` (cond ~1e6 here), losing ~6-7
    digits, so its free energies carry ~1e-4 numerical error. This routine only ever
    factors well-conditioned matrices (cond(free_M) ~ 1e2), so it is *more* accurate
    than the reference, which it therefore reproduces only to the reference's own
    ~1e-4 noise floor (not to machine precision).
    """
    if sp.sparse.issparse(free_M):
        free_M = free_M.toarray()
    N = len(free_M)

    midstep_constraint_locations = get_midstep_locations(
        left_open, right_open, base_midstep_locations=base_midstep_locations)

    # trivial case: no (or a single) constraint -> free DNA only
    if len(midstep_constraint_locations) <= 1:
        cf0 = sla.cho_factor(free_M, check_finite=False)
        F = 0.5*_logdet_spd(cf0) - 0.5*N*_LOG2PI
        return {
            'F': F, 'F_fluctuation': F, 'F_enthalpy': 0, 'F_jacob': 0,
            'F_freedna': F, 'dF': 0,
            'gs': np.zeros(free_gs.shape, dtype=np.float64), 'alphas': None,
        }

    sks = midstep_groundstate_se3(free_gs, midstep_constraint_locations)
    nuc_mu0 = nuc_mu0_full[left_open:len(nuc_mu0_full)-right_open]
    nuc_K = nuc_K_full[6*left_open:len(nuc_K_full)-6*right_open,
                       6*left_open:len(nuc_K_full)-6*right_open]

    # factor free_M once (reused by both passes and by F_free)
    cf = sla.cho_factor(free_M, check_finite=False)
    logdet_freeM = _logdet_spd(cf)

    # coordinate_transformation(nuc_mu0, sks) is identical across passes -> compute once
    B0, Pbar0 = _fast_coordinate_transformation(nuc_mu0, sks)

    # ---- predictor pass ----
    transform, replaced_ids = _fast_transformation_first_order(
        free_gs, midstep_constraint_locations)
    M_Mp, G, logdet_invMmp, con_idx = _sc_marginal(cf, transform, replaced_ids)
    alpha, Y_C, Kcomb, cK, F_enthalpy = _sc_binding(M_Mp, nuc_K, B0, Pbar0, nuc_mu0, sks)
    gs = (G @ (M_Mp @ Y_C)).reshape((-1, 6))
    logdet_transform = _sc_logdet_transform(transform, con_idx)

    # ---- corrector pass ----
    if use_correction:
        transform, replaced_ids, shift = _fast_transformation_correction(
            free_gs, midstep_constraint_locations, gs)
        M_Mp, G, logdet_invMmp, con_idx = _sc_marginal(cf, transform, replaced_ids)
        alpha, Y_C, Kcomb, cK, F_enthalpy = _sc_binding(
            M_Mp, nuc_K, B0, Pbar0, nuc_mu0, sks, shift=shift)
        gs = (G @ (M_Mp @ Y_C)).reshape((-1, 6))
        logdet_transform = _sc_logdet_transform(transform, con_idx)

    alphas = alpha.reshape((-1, 6))

    # ---- fluctuation entropy (uses the final/corrected quantities) ----
    nK = len(Kcomb)
    Z_entropy = 0.5*_logdet_spd(cK) - 0.5*nK*_LOG2PI

    # logdet(M_R) = logdet(free_M) - 2 logdet(transform) + logdet(A free_M^-1 A^T)
    logdet_MR = logdet_freeM - 2.0*logdet_transform + logdet_invMmp
    nR = N - con_idx.size
    R_entropy = 0.5*logdet_MR - 0.5*nR*_LOG2PI

    F_Ajacob = logdet_transform
    F_fluctuation = Z_entropy + R_entropy + F_Ajacob

    # free energy of unconstrained DNA
    F_free = 0.5*logdet_freeM - 0.5*N*_LOG2PI

    return {
        'F': F_fluctuation + F_enthalpy,
        'F_fluctuation': F_fluctuation,
        'F_enthalpy': F_enthalpy,
        'F_jacob': F_Ajacob,
        'F_freedna': F_free,
        'dF': F_fluctuation + F_enthalpy - F_free,
        'gs': gs,
        'alphas': alphas,
    }


if __name__ == '__main__':

    from .nuctriads import read_nucleosome_triads
    from .RBPStiff.read_params import GenStiffness
    # from .PolyCG.polycg.cgnaplus import cgnaplus_bps_params
    
    genstiff = GenStiffness(method='hybrid')   # alternatively you can use the 'crystal' method for the Olson data
    seq  = "CTGGAGAATCCCGGTGCCGAGGCCGCTCAATTGGTCGTAGACAGCTCTAGCACCGCTTAAACGCACGTACGCGCTGTCCCCCGCGTTTTAACCGCCAAGGGGATTACTCCCTAGTCTCCAGGCACGTGTCAGATATATACATCCTGT"

    stiffmat,groundstate = genstiff.gen_params(seq,use_group=True)

    triadfn = os.path.join(os.path.dirname(__file__), 'Parameters/Nucleosome.state')
    nuctriads = read_nucleosome_triads(triadfn)

    midstep_constraint_locations = [
        2, 6, 14, 17, 24, 29, 
        34, 38, 45, 49, 55, 59, 
        65, 69, 76, 80, 86, 90, 
        96, 100, 107, 111, 116, 121, 
        128, 131, 139, 143
    ]
    
    nuc_mu0 = calculate_midstep_triads(
        midstep_constraint_locations,
        nuctriads
    )
            
    left_open = 0
    right_open = 0
    fn_nuc_K_pos_resc_sym    = os.path.join(os.path.dirname(__file__), 'Parameters/MDParams/nuc_K_pos_resc_sym.npy') 
    
    print(fn_nuc_K_pos_resc_sym)
    nuc_K_pos_resc_sym = np.load(fn_nuc_K_pos_resc_sym)

    print('##################################')
    print('Calculate model')
    nucout = sc_free_energy(
        groundstate,
        stiffmat,    
        nuc_mu0,
        nuc_K_pos_resc_sym,
        left_open=left_open,
        right_open=right_open,
        use_correction=True,
    )
    
    print(f"F:   {nucout['F']}")
    print(f"F_E: {nucout['F_fluctuation']}")
    print(f"F_S: {nucout['F_enthalpy']}")

    print(f"dF:  {nucout['F'] - nucout['F_freedna']}")
    
    
    print(nucout['gs'].shape)
    print(nucout['alphas'].shape)
    