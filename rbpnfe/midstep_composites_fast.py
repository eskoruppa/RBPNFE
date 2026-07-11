"""
numba-JIT'd reimplementations of the two per-block builders used to assemble the
midstep composition transformation:

    midstep_composition_block_first_order   (predictor transform)
    midstep_composition_block_correction    (corrector transform)

These are exact transcriptions of the reference implementations in
``midstep_composites.py`` with the ``so3`` calls replaced by their njit
single-vector kernels, so they compile to machine code and eliminate the
thousands of tiny Python/numba-dispatch calls that dominate the transform
assembly. They are verified to reproduce the reference block matrices to ~1e-13.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from .PolyCG.polycg.SO3.so3.Euler import _euler2rotmat_sv, _rotmat2euler_sv
from .PolyCG.polycg.SO3.so3.generators import _hat_map_sv
from .PolyCG.polycg.SO3.so3.conversions import (
    _splittransform_algebra2group_sv,
    _splittransform_group2algebra_sv,
)


@njit(cache=True)
def _Hinverse(Psi):
    psih = _hat_map_sv(Psi)
    psihsq = psih @ psih
    Hinv = np.eye(3)
    Hinv += 0.5 * psih
    Hinv += 1.0/12 * psihsq
    Hinv -= 1.0/720 * psihsq @ psihsq
    Hinv += 1.0/30240 * psihsq @ psihsq @ psihsq
    return Hinv


@njit(cache=True)
def coordinate_transformation(muk0s, sks):
    K = sks.shape[0]
    M = muk0s.shape[0]
    B = np.zeros((K * 6, M * 6))
    Pbar = np.zeros(K * 6)
    for k in range(K):
        sig0 = np.linalg.inv(muk0s[k]) @ muk0s[k + 1]
        Sig = sig0[:3, :3]
        sig = sig0[:3, 3]
        Sk = sks[k, :3, :3]
        sk = sks[k, :3, 3]

        Psi = _rotmat2euler_sv(Sk.T @ Sig)
        Hi = _Hinverse(Psi)
        Bkm = np.zeros((6, 6))
        Bkp = np.zeros((6, 6))
        Bkm[:3, :3] = -Hi @ Sig.T
        Bkm[3:, :3] = Sk.T @ _hat_map_sv(sig)
        Bkm[3:, 3:] = -Sk.T
        Bkp[:3, :3] = Hi
        Bkp[3:, 3:] = Sk.T @ Sig

        B[6 * k:6 * (k + 1), 6 * k:6 * (k + 1)] = Bkm
        B[6 * k:6 * (k + 1), 6 * (k + 1):6 * (k + 2)] = Bkp

        Pbar[k * 6:k * 6 + 3] = Psi
        Pbar[k * 6 + 3:k * 6 + 6] = Sk.T @ (sig - sk)
    return B, Pbar


@njit(cache=True)
def coordinate_transformation_correction(muk0s, sks, Z_delta_ref_flat):
    # Z_delta_ref_flat is the flat (K*6,) correction vector.
    K = sks.shape[0]
    M = muk0s.shape[0]
    Zref = Z_delta_ref_flat.reshape(Z_delta_ref_flat.shape[0] // 6, 6)
    B = np.zeros((K * 6, M * 6))
    Pbar = np.zeros(K * 6)
    for k in range(K):
        sig0 = np.linalg.inv(muk0s[k]) @ muk0s[k + 1]
        SIG = sig0[:3, :3]
        sig = sig0[:3, 3]
        Sk = sks[k, :3, :3]
        sk = sks[k, :3, 3]

        Psi = _rotmat2euler_sv(Sk.T @ SIG)
        Hi = _Hinverse(Psi)

        Z0k = _euler2rotmat_sv(np.ascontiguousarray(Zref[k, :3]))
        htheta0 = _hat_map_sv(np.ascontiguousarray(Zref[k, :3]))

        Bkm = np.zeros((6, 6))
        Bkp = np.zeros((6, 6))
        Bkm[:3, :3] = -Hi @ SIG.T
        Bkm[3:, :3] = Sk.T @ _hat_map_sv(sig)
        Bkm[3:, 3:] = -Sk.T @ Z0k.T
        Bkp[:3, :3] = Hi
        Bkp[3:, 3:] = Sk.T @ Z0k.T @ SIG

        B[6 * k:6 * (k + 1), 6 * k:6 * (k + 1)] = Bkm
        B[6 * k:6 * (k + 1), 6 * (k + 1):6 * (k + 2)] = Bkp

        Pbar[k * 6:k * 6 + 3] = Psi
        Pbar[k * 6 + 3:k * 6 + 6] = Sk.T @ ((Z0k.T + htheta0) @ sig - sk)
    return B, Pbar


@njit(cache=True)
def _rot_accu(rots, i, j):
    raccu = np.eye(3)
    for k in range(i, j + 1):
        raccu = raccu @ rots[k]
    return raccu


@njit(cache=True)
def block_first_order(groundstate):
    Phi0s = groundstate[:, :3]
    N = groundstate.shape[0]

    srots = np.zeros((N, 3, 3))
    srots[0] = _euler2rotmat_sv(0.5 * Phi0s[0])
    srots[N - 1] = _euler2rotmat_sv(0.5 * Phi0s[N - 1])
    for l in range(1, N - 1):
        srots[l] = _euler2rotmat_sv(Phi0s[l])

    trans = groundstate[:, 3:].copy()
    trans[0] = 0.5 * srots[0].T @ trans[0]
    trans[N - 1] = 0.5 * trans[N - 1]

    ndims = 6
    i = 0
    j = N - 1
    comp_block = np.zeros((ndims, N * ndims))

    # middle blocks (i <= k <= j)
    for k in range(i, j + 1):
        Saccu = _rot_accu(srots, k + 1, j)
        comp_block[:3, k * 6:k * 6 + 3] = Saccu.T
        comp_block[3:, k * 6 + 3:k * 6 + 6] = Saccu.T

        coup = np.zeros((3, 3))
        for l in range(k + 1, j + 1):
            coup += _hat_map_sv(-_rot_accu(srots, l, j).T @ trans[l])
        coup = coup @ Saccu.T
        comp_block[3:, k * 6:k * 6 + 3] = coup

    # first block (i)
    Saccu = _rot_accu(srots, i + 1, j)
    Phi_0 = Phi0s[0]
    H_half = _splittransform_algebra2group_sv(0.5 * Phi_0)
    Hinv = _splittransform_group2algebra_sv(Phi_0)
    Hprod = H_half @ Hinv

    comp_block[:3, :3] = 0.5 * Saccu.T @ Hprod
    comp_block[3:, 3:6] = 0.5 * Saccu.T

    coup = np.zeros((3, 3))
    for l in range(1, j + 1):
        coup += _hat_map_sv(-_rot_accu(srots, l, j).T @ trans[l])
    coup = coup @ Saccu.T
    coup += Saccu.T @ srots[i].T @ _hat_map_sv(trans[i])
    coup = 0.5 * coup @ Hprod
    comp_block[3:, :3] = coup

    # last block (j)
    Phi_0 = Phi0s[N - 1]
    H_half = _splittransform_algebra2group_sv(0.5 * Phi_0)
    Hinv = _splittransform_group2algebra_sv(Phi_0)
    Hprod = H_half @ Hinv

    comp_block[:3, j * 6:j * 6 + 3] = 0.5 * Hprod
    comp_block[3:, j * 6 + 3:j * 6 + 6] = 0.5 * srots[N - 1]

    return comp_block


@njit(cache=True)
def block_correction(groundstate, deformations):
    N = groundstate.shape[0]
    ndims = 6
    i = 0
    j = N - 1

    # ---- groundstate components ----
    Phi0s = groundstate[:, :3]
    srots = np.zeros((N, 3, 3))
    srots[0] = _euler2rotmat_sv(0.5 * Phi0s[0])
    srots[N - 1] = _euler2rotmat_sv(0.5 * Phi0s[N - 1])
    for l in range(1, N - 1):
        srots[l] = _euler2rotmat_sv(Phi0s[l])

    strans = groundstate[:, 3:].copy()
    strans[0] = 0.5 * srots[0].T @ strans[0]
    strans[N - 1] = 0.5 * strans[N - 1]

    # ---- deformation components ----
    Phid0 = deformations[:, :3]
    drots = np.zeros((N, 3, 3))
    Phi_0 = Phi0s[0]
    H_half = _splittransform_algebra2group_sv(0.5 * Phi_0)
    Hinv = _splittransform_group2algebra_sv(Phi_0)
    Hprod = H_half @ Hinv
    drots[0] = _euler2rotmat_sv(0.5 * Hprod @ Phid0[0])
    Phi_0 = Phi0s[N - 1]
    H_half = _splittransform_algebra2group_sv(0.5 * Phi_0)
    Hinv = _splittransform_group2algebra_sv(Phi_0)
    Hprod = H_half @ Hinv
    drots[N - 1] = _euler2rotmat_sv(0.5 * Hprod @ Phid0[N - 1])
    for l in range(1, N - 1):
        drots[l] = _euler2rotmat_sv(Phid0[l])

    # ---- repeated products ----
    Rrots = np.zeros((N, 3, 3))
    for l in range(N):
        Rrots[l] = srots[l] @ drots[l]

    # S_{[l,j]}
    S_lj = np.zeros((N + 1, 3, 3))
    curr = np.eye(3)
    S_lj[N] = curr
    for k in range(N):
        curr = srots[N - 1 - k] @ curr
        S_lj[N - 1 - k] = curr

    # s_{(l,j)}
    s_lj = np.zeros((N + 1, 3))
    for l in range(N):
        scomp = np.zeros(3)
        for k in range(l, N):
            scomp += _rot_accu(srots, l, k - 1) @ strans[k]
        s_lj[l] = scomp

    # lambda_k
    lambdak = np.zeros((N, 3))
    for k in range(N):
        lambsum = np.zeros(3)
        for l in range(k + 1, N):
            lambsum += _rot_accu(Rrots, k + 1, l - 1) @ srots[l] @ (drots[l] - np.eye(3)) @ s_lj[l + 1]
        lambdak[k] = lambsum

    # ---- compose block ----
    comp_block = np.zeros((ndims, N * ndims))
    const = np.zeros(6)

    for l in range(i, j + 1):
        prefac_trans = S_lj[i].T @ _rot_accu(Rrots, i, l - 1) @ srots[l]
        prefac_coup = S_lj[i].T @ _rot_accu(Rrots, i, l)

        if l == i:
            Phi_0 = Phi0s[0]
            H_half = _splittransform_algebra2group_sv(0.5 * Phi_0)
            Hinv = _splittransform_group2algebra_sv(Phi_0)
            Hprod = H_half @ Hinv

            comp_block[:3, l * 6:l * 6 + 3] = 0.5 * S_lj[l + 1].T @ Hprod
            comp_block[3:, l * 6 + 3:l * 6 + 6] = 0.5 * prefac_trans

            phid0_i = 0.5 * Hprod @ Phid0[0]
            Hmat = _splittransform_algebra2group_sv(phid0_i)
            hspdlamHmat = _hat_map_sv(lambdak[l]) + _hat_map_sv(s_lj[l + 1])

            coup_corr = -0.5 * prefac_coup @ hspdlamHmat @ Hmat @ Hprod
            v_full = groundstate[0, 3:]
            S_full_T = (srots[0] @ srots[0]).T
            coup_rh = 0.25 * prefac_trans @ S_full_T @ _hat_map_sv(v_full) @ srots[0] @ Hprod
            comp_block[3:, l * 6:l * 6 + 3] = coup_corr + coup_rh

            const[3:] += prefac_trans @ (drots[l] - np.eye(3)) @ s_lj[l + 1] + prefac_coup @ hspdlamHmat @ Hmat @ phid0_i

        elif l == j:
            Phi_0 = Phi0s[N - 1]
            H_half = _splittransform_algebra2group_sv(0.5 * Phi_0)
            Hinv = _splittransform_group2algebra_sv(Phi_0)
            Hprod = H_half @ Hinv
            comp_block[:3, l * 6:l * 6 + 3] = 0.5 * Hprod
            comp_block[3:, l * 6 + 3:l * 6 + 6] = prefac_trans @ (0.5 * srots[N - 1])

        else:
            comp_block[:3, l * 6:l * 6 + 3] = S_lj[l + 1].T
            comp_block[3:, l * 6 + 3:l * 6 + 6] = prefac_trans

            Hmat = _splittransform_algebra2group_sv(Phid0[l])
            hspdlamHmat = _hat_map_sv(lambdak[l]) + _hat_map_sv(s_lj[l + 1])
            comp_block[3:, l * 6:l * 6 + 3] = -prefac_coup @ hspdlamHmat @ Hmat
            const[3:] += prefac_trans @ (drots[l] - np.eye(3)) @ s_lj[l + 1] + prefac_coup @ hspdlamHmat @ Hmat @ Phid0[l]

    return comp_block, const


def transformation_first_order(intrinsic_groundstate, midstep_constraint_locations):
    """Assemble the full predictor transform using the JIT'd block builder."""
    N = len(intrinsic_groundstate)
    mat = np.eye(N * 6)
    replaced_ids = []
    for i in range(len(midstep_constraint_locations) - 1):
        id1 = midstep_constraint_locations[i]
        id2 = midstep_constraint_locations[i + 1]
        partial_gs = np.ascontiguousarray(intrinsic_groundstate[id1:id2 + 1])
        block = block_first_order(partial_gs)
        mat[id2 * 6:id2 * 6 + 6, id1 * 6:id2 * 6 + 6] = block
        replaced_ids.append(id2)
    return mat, replaced_ids


def transformation_correction(intrinsic_groundstate, midstep_constraint_locations, first_order_compromise):
    """Assemble the full corrector transform using the JIT'd block builder."""
    N = len(intrinsic_groundstate)
    mat = np.eye(N * 6)
    replaced_ids = []
    shifts = []
    for i in range(len(midstep_constraint_locations) - 1):
        id1 = midstep_constraint_locations[i]
        id2 = midstep_constraint_locations[i + 1]
        partial_gs = np.ascontiguousarray(intrinsic_groundstate[id1:id2 + 1])
        partial_compromise = np.ascontiguousarray(first_order_compromise[id1:id2 + 1])
        block, shift = block_correction(partial_gs, partial_compromise)
        shifts.append(shift)
        mat[id2 * 6:id2 * 6 + 6, id1 * 6:id2 * 6 + 6] = block
        replaced_ids.append(id2)
    shifts = np.array(shifts).flatten()
    return mat, replaced_ids, shifts
