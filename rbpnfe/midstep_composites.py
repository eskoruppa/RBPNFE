import sys, os
import numpy as np
from typing import List, Tuple, Callable, Any, Dict

from .PolyCG.polycg.SO3 import so3
from .PolyCG.polycg.transforms.transform_SO3 import euler2rotmat_so3
from .PolyCG.polycg.transforms.transform_marginals import send_to_back_permutation


def calculate_midstep_triads(
    triad_ids: List[int],  # index of the lower (left-hand) triad neighboring the constraint midstep-triad
    nucleosome_triads: np.ndarray
) -> np.ndarray:
    midstep_triads = np.zeros((len(triad_ids),4,4))
    for i,id in enumerate(triad_ids):
        T1 = nucleosome_triads[id]
        T2 = nucleosome_triads[id+1]
        midstep_triads[i,:3,:3] = T1[:3,:3] @ so3.euler2rotmat(0.5*so3.rotmat2euler(T1[:3,:3].T @ T2[:3,:3]))
        midstep_triads[i,:3,3]  = 0.5* (T1[:3,3]+T2[:3,3])
        midstep_triads[i,3,3]   = 1
    return midstep_triads


def midstep_composition_excess(
    groundstate: np.ndarray,
    triad1: np.ndarray,
    triad2: np.ndarray
) -> np.ndarray:
    g_ij = np.linalg.inv(triad1) @ triad2
    Smats = midstep_se3_groundstate(groundstate)
    s_ij = np.eye(4)
    for Smat in Smats:
        s_ij = s_ij @ Smat
    d_ij = np.linalg.inv(s_ij) @ g_ij
    X = so3.se3_rotmat2euler(d_ij)
    return X


def midstep_excess_vals(
    groundstate: np.ndarray,
    midstep_constraint_locations: List[int],
    midstep_triads: np.ndarray  
):
    num = len(midstep_constraint_locations)-1
    excess_vals = np.zeros((num,6))
    for i in range(num):
        id1 = midstep_constraint_locations[i]
        id2 = midstep_constraint_locations[i+1]
        triad1 = midstep_triads[i]
        triad2 = midstep_triads[i+1]
        partial_gs = groundstate[id1:id2+1] 
        excess_vals[i] = midstep_composition_excess(partial_gs,triad1,triad2) 
    return excess_vals


def midstep_groundstate_se3(
    gs: np.ndarray, 
    midstep_locs: List[int]
    ) -> np.ndarray:
    num = len(midstep_locs)-1
    sks = np.zeros((num,4,4))
    for i in range(num):
        id1 = midstep_locs[i]
        id2 = midstep_locs[i+1]
        partial_gs = gs[id1:id2+1] 
        
        Smats = midstep_se3_groundstate(partial_gs)
        s_ij = np.eye(4)
        for Smat in Smats:
            s_ij = s_ij @ Smat
        sks[i] = s_ij
    return sks


def rot_accu(rots: np.ndarray,i,j) -> np.ndarray:
    raccu = np.eye(3)
    for k in range(i,j+1):
        raccu = raccu @ rots[k]
    return raccu


def midstep_groundstate(
    gs: np.ndarray ,
    midstep_locs: List[int]
    ) -> np.ndarray:
    num = len(midstep_locs)-1
    mid_gs = np.zeros((num,6))
    for i in range(num):
        id1 = midstep_locs[i]
        id2 = midstep_locs[i+1]
        partial_gs = gs[id1:id2+1] 
        
        Smats = midstep_se3_groundstate(partial_gs)
        s_ij = np.eye(4)
        for Smat in Smats:
            s_ij = s_ij @ Smat
        mid_gs[i] = so3.se3_rotmat2euler(s_ij)
    return mid_gs

    
def midstep_se3_groundstate(groundstate: np.ndarray) -> np.ndarray:
    Phi0s = groundstate[:,:3]
    N = len(groundstate)
    # assign static rotation matrices
    srots = np.zeros((N,3,3))
    srots[0]  = so3.euler2rotmat(0.5*Phi0s[0])    
    srots[-1] = so3.euler2rotmat(0.5*Phi0s[-1])    
    for l in range(1,len(srots)-1):
        srots[l] = so3.euler2rotmat(Phi0s[l])   
    # assign translation vectors
    trans = np.copy(groundstate[:,3:])
    trans[0] = 0.5*trans[0]
    trans[-1] = 0.5* srots[-1].T @ trans[-1]
    
    Smats = np.zeros((N,4,4))
    for i in range(N):
        S = np.zeros((4,4))
        S[:3,:3] = srots[i]
        S[:3,3]  = trans[i]
        S[3,3]   = 1
        Smats[i] = S
    return Smats


def midstep_composition_transformation(
    intrinsic_groundstate: np.ndarray,
    midstep_constraint_locations: List[int],
) -> Tuple[np.ndarray,List[int]]:
    N = len(intrinsic_groundstate)
    mat = np.eye(N*6)
    replaced_ids = []
    for i in range(len(midstep_constraint_locations)-1):
        id1 = midstep_constraint_locations[i]
        id2 = midstep_constraint_locations[i+1]
        replace_id = id2
        partial_gs = intrinsic_groundstate[id1:id2+1]
        midstep_comp_block = midstep_composition_block_first_order(partial_gs)
        mat[replace_id*6:replace_id*6+6,id1*6:id2*6+6] = midstep_comp_block
        replaced_ids.append(replace_id)
    return mat, replaced_ids

def midstep_composition_block_first_order(
    groundstate: np.ndarray
) -> np.ndarray:
    if len(groundstate) < 2:
        raise ValueError(f'midstep_composition_block: grounstate needs to contain at least two elements. {len(groundstate)} provided.')
    
    Phi0s = groundstate[:,:3]
    # ss    = groundstate[:,3:]
    
    N = len(groundstate)
    # assign static rotation matrices
    srots = np.zeros((N,3,3))
    srots[0]  = so3.euler2rotmat(0.5*Phi0s[0])    
    srots[-1] = so3.euler2rotmat(0.5*Phi0s[-1])    
    for l in range(1,len(srots)-1):
        srots[l] = so3.euler2rotmat(Phi0s[l])    
    
    # assign translation vectors
    trans = np.copy(groundstate[:,3:])
    trans[0] = 0.5*trans[0]
    trans[-1] = 0.5* srots[-1].T @ trans[-1]
    
    ndims = 6
    N = len(groundstate)
    i = 0
    j = N-1
    comp_block  = np.zeros((ndims,N*ndims))
    
    ################################  
    # set middle blocks (i < k < j)
    for k in range(i,j+1):
        Saccu = rot_accu(srots,k+1,j)
        comp_block[:3,k*6:k*6+3]   = Saccu.T
        comp_block[3:,k*6+3:k*6+6] = Saccu.T
        
        coup = np.zeros((3,3))
        for l in range(k+1,j+1):
            coup += so3.hat_map(-rot_accu(srots,l,j).T @ trans[l])
        coup = coup @ Saccu.T
        comp_block[3:,k*6:k*6+3] = coup
    
    ################################  
    # set first block (i)
    Saccu = rot_accu(srots,i+1,j)
    Phi_0 = Phi0s[0]
    H_half = so3.splittransform_algebra2group(0.5*Phi_0)
    Hinv   = so3.splittransform_group2algebra(Phi_0)
    Hprod  = H_half @ Hinv
    
    # assign diagonal blocks
    comp_block[:3,:3] = 0.5 * Saccu.T @ Hprod
    comp_block[3:,3:6] = 0.5 * Saccu.T
    
    coup = np.zeros((3,3))
    # first term
    for l in range(1,j+1):
        coup += so3.hat_map(-rot_accu(srots,l,j).T @ trans[l])
    coup = coup @ Saccu.T
    # second term
    coup += Saccu.T @ srots[i].T @ so3.hat_map(trans[i])
    # multiply everything with 0.5 * Hprod
    coup = 0.5 * coup @ Hprod
    # assign coupling term
    comp_block[3:,:3] = coup
    
    ################################  
    # set last block (j)
    Phi_0 = Phi0s[-1]
    H_half = so3.splittransform_algebra2group(0.5*Phi_0)
    Hinv   = so3.splittransform_group2algebra(Phi_0)
    Hprod  = H_half @ Hinv
    
    # assign diagonal blocks
    comp_block[:3,j*6:j*6+3]   = 0.5 * Hprod
    comp_block[3:,j*6+3:j*6+6] = 0.5 * srots[-1]
        
    return comp_block


def midstep_composition_transformation_correction(
    intrinsic_groundstate: np.ndarray,
    midstep_constraint_locations: List[int],
    first_order_compromise: np.ndarray
) -> np.ndarray:
    N = len(intrinsic_groundstate)
    mat = np.eye(N*6)
    replaced_ids = []
    shifts = []
    for i in range(len(midstep_constraint_locations)-1):
        id1 = midstep_constraint_locations[i]
        id2 = midstep_constraint_locations[i+1]
        replace_id = id2
        partial_gs = intrinsic_groundstate[id1:id2+1]
        partial_compromise = first_order_compromise[id1:id2+1]
        midstep_comp_block,shift = midstep_composition_block_correction(partial_gs,partial_compromise)
        shifts.append(shift)        
        mat[replace_id*6:replace_id*6+6,id1*6:id2*6+6] = midstep_comp_block
        replaced_ids.append(replace_id)
    shifts = np.array(shifts).flatten()
    return mat, replaced_ids, shifts


def midstep_composition_block_correction(
    groundstate: np.ndarray,
    deformations: np.ndarray
) -> np.ndarray:
    if len(groundstate) < 2:
        raise ValueError(f'midstep_composition_block: groundstate needs to contain at least two elements. {len(groundstate)} provided.')
     
    if len(groundstate) != len(deformations):
        raise ValueError('Dimsional mismatch between groundstate and deformation')
    
    N = len(groundstate)
    ndims = 6
    N = len(groundstate)
    i = 0
    j = N-1
    
    ################################################
    ################################################
    # assign groundstate components
    
    # Euler vectors
    Phi0s = groundstate[:,:3]
    # static rotation matrices
    srots = np.zeros((N,3,3))
    # left half-step
    srots[0]  = so3.euler2rotmat(0.5*Phi0s[0])    
    # right half-step
    srots[-1] = so3.euler2rotmat(0.5*Phi0s[-1])    
    # bulf steps
    for l in range(1,len(srots)-1):
        srots[l] = so3.euler2rotmat(Phi0s[l])
    
    # assign translation vectors
    strans = np.copy(groundstate[:,3:])
    strans[0] = 0.5* srots[0].T @ strans[0] 
    strans[-1] = 0.5*strans[-1]
    
    ################################################
    ################################################
    # assign deformation components
    # D^0
    # Phi^0
    # R^0
    
    # Euler Vectors
    Phid0 = deformations[:,:3]
    
    # dynamic rotation matrices
    drots = np.zeros((N,3,3))
    # left half-step
    Phi_0 = Phi0s[0]
    H_half = so3.splittransform_algebra2group(0.5*Phi_0)
    Hinv   = so3.splittransform_group2algebra(Phi_0)
    Hprod  = H_half @ Hinv
    drots[0]  = so3.euler2rotmat(0.5*Hprod @ Phid0[0]) 
    # right half-step
    Phi_0 = Phi0s[-1]
    H_half = so3.splittransform_algebra2group(0.5*Phi_0)
    Hinv   = so3.splittransform_group2algebra(Phi_0)
    Hprod  = H_half @ Hinv
    drots[-1]  = so3.euler2rotmat(0.5*Hprod @ Phid0[-1]) 
    # bulk steps
    for l in range(1,len(drots)-1):
        drots[l] = so3.euler2rotmat(Phid0[l])
        
    ################################################
    ################################################
    # pre compute repeatedly occuring products
    
    Rrots = np.zeros(srots.shape)
    for l in range(len(drots)):
        Rrots[l] = srots[l] @ drots[l]
    
    ################################################
    # products of static rotation matrices
    # S_{[l,j]}
    S_lj = np.zeros((N+1,3,3))
    curr = np.eye(3)
    S_lj[N] = curr
    for k in range(N):
        curr = srots[N-1-k] @ curr
        S_lj[N-1-k] = curr
        
    ################################################
    # translational component of composites
    # s_{(l,j)}
    s_lj = np.zeros((N+1,3))
    for l in range(N):
        scomp = np.zeros(3)
        for k in range(l,N):
            scomp += rot_accu(srots,l,k-1) @ strans[k]
        s_lj[l] = scomp

    ################################################
    # lambda_k

    lambdak = np.zeros((N,3))
    for k in range(N):
        lambsum = np.zeros(3)
        # j+1 = N
        for l in range(k+1,N):
            lambsum += rot_accu(Rrots,k+1,l-1) @ srots[l] @ (drots[l] - np.eye(3)) @ s_lj[l+1]
        lambdak[k] = lambsum

    ################################################
    ################################################
    # compose composite block
    comp_block = np.zeros((ndims,N*ndims))
    const      = np.zeros(6)

    ################################  
    # set middle blocks (i < k < j)
    for l in range(i,j+1):
        prefac = S_lj[i].T @ rot_accu(Rrots,i,l-1) @ srots[l]

        if l == i:
            Phi_0 = Phi0s[0]
            H_half = so3.splittransform_algebra2group(0.5*Phi_0)
            Hinv   = so3.splittransform_group2algebra(Phi_0)
            Hprod  = H_half @ Hinv
            
            # rot
            comp_block[:3,l*6:l*6+3] = 0.5 * S_lj[l+1].T @ Hprod
            # trans
            comp_block[3:,l*6+3:l*6+6] = 0.5 * prefac
            # coupling and constant
            phid0_i = 0.5 * Hprod @ Phid0[0]            
            Hmat    = so3.splittransform_algebra2group(phid0_i)
            hspdlamHmat = so3.hat_map(s_lj[l+1]) + drots[l] @ so3.hat_map(lambdak[l]) @ Hmat
            # rot-trans coupling
            comp_block[3:,l*6:l*6+3]   = -0.5 * prefac @ hspdlamHmat @ Hprod
            # const
            const[3:] += prefac @ ( (drots[l] - np.eye(3)) @ s_lj[l+1] +  hspdlamHmat @ phid0_i ) 
                        
        elif l == j:
            Phi_0 = Phi0s[-1]
            H_half = so3.splittransform_algebra2group(0.5*Phi_0)
            Hinv   = so3.splittransform_group2algebra(Phi_0)
            Hprod  = H_half @ Hinv
            # rot
            comp_block[:3,l*6:l*6+3] = 0.5 * Hprod
            # trans
            comp_block[3:,l*6+3:l*6+6] = 0.5 * prefac
            # no constant
        else:
            # rot
            comp_block[:3,l*6:l*6+3]   = S_lj[l+1].T
            # trans
            comp_block[3:,l*6+3:l*6+6] = prefac
            # coupling and constant
            Hmat = so3.splittransform_algebra2group(Phid0[l])
            hspdlamHmat = so3.hat_map(s_lj[l+1]) + drots[l] @ so3.hat_map(lambdak[l]) @ Hmat
            # rot-trans coupling
            comp_block[3:,l*6:l*6+3] = -prefac @ hspdlamHmat
            # const
            const[3:] += prefac @ ( (drots[l] - np.eye(3)) @ s_lj[l+1] +  hspdlamHmat @ Phid0[l] ) 
    
    return comp_block,const



