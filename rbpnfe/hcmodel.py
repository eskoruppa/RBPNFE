import sys, os
import numpy as np
import scipy as sp
from typing import List, Tuple, Callable, Any, Dict

from .PolyCG.polycg.SO3 import so3
from .PolyCG.polycg.transforms.transform_marginals import send_to_back_permutation
from .midstep_composites import midstep_composition_transformation
from .midstep_composites import midstep_composition_transformation_correction
from .midstep_composites import calculate_midstep_triads, midstep_excess_vals


def hc_free_energy(
    intrinsic_groundstate: np.ndarray,
    stiffmat: np.ndarray,
    midstep_constraint_locations: List[int],  # index of the lower (left-hand) triad neighboring the constraint midstep-triad
    nucleosome_triads: np.ndarray,
    use_correction: bool = False,
) -> np.ndarray:
    
    if len(midstep_constraint_locations) == 0:
        n = len(stiffmat)
        F_pi = -0.5*n * np.log(2*np.pi)
        # matrix term
        logdet_sign, logdet = np.linalg.slogdet(stiffmat)
        F_mat = 0.5*logdet
        F = F_mat + F_pi  
    
        Fdict = {
            'F': F,
            'F_entropy'  : F,
            'F_enthalpy' : 0,
            'F_jacob'    : 0,
            'F_freedna'  : F,
            'gs'         : np.zeros(n)
        }
        return Fdict
    
    
    midstep_constraint_locations = sorted(list(set(midstep_constraint_locations)))

    midstep_triads = calculate_midstep_triads(
        midstep_constraint_locations,
        nucleosome_triads
    )

    # find contraint excess values
    excess_vals = midstep_excess_vals(
        intrinsic_groundstate,
        midstep_constraint_locations,
        midstep_triads
    )  
    C = excess_vals.flatten()
        
    # find composite transformation
    transform, replaced_ids = midstep_composition_transformation(
        intrinsic_groundstate,
        midstep_constraint_locations
    )
    
    # transform stiffness matrix
    inv_transform = np.linalg.inv(transform)
    stiffmat_transformed = inv_transform.T @ stiffmat @ inv_transform
    
    # rearrange stiffness matrix
    full_replaced_ids = list()
    for i in range(len(replaced_ids)):
        full_replaced_ids += [6*replaced_ids[i]+j for j in range(6)]
     
    P = send_to_back_permutation(len(stiffmat),full_replaced_ids)
    stiffmat_rearranged = P @ stiffmat_transformed @ P.T

    # select fluctuating, constraint and coupling part of matrix
    N  = len(stiffmat)
    NC = len(full_replaced_ids)
    NF = N-NC
    
    MF = stiffmat_rearranged[:NF,:NF]
    MC = stiffmat_rearranged[NF:,NF:]
    MM = stiffmat_rearranged[NF:,:NF]
    
    MFi = np.linalg.inv(MF)
    b = MM.T @ C
    
    ########################################
    ########################################
    if use_correction:
        
        alpha = -MFi @ b
        gs_transf_perm = np.concatenate((alpha,C))
        gs_transf = P.T @ gs_transf_perm
        gs = inv_transform @ gs_transf
    
        gs = gs.reshape((len(gs)//6,6))
        # find composite transformation
        transform, replaced_ids, shift = midstep_composition_transformation_correction(
            intrinsic_groundstate,
            midstep_constraint_locations,
            gs
        )
        
        # transform stiffness matrix
        inv_transform = np.linalg.inv(transform)
        stiffmat_transformed = inv_transform.T @ stiffmat @ inv_transform
        
        # rearrange stiffness matrix
        stiffmat_rearranged = P @ stiffmat_transformed @ P.T

        # select fluctuating, constraint and coupling part of matrix
        N  = len(stiffmat)
        NC = len(full_replaced_ids)
        NF = N-NC
        
        MF = stiffmat_rearranged[:NF,:NF]
        MC = stiffmat_rearranged[NF:,NF:]
        MM = stiffmat_rearranged[NF:,:NF]
        
        C = C - shift
        MFi = np.linalg.inv(MF)
        b = MM.T @ C
        
    # Calculate ground state 
    alpha = -MFi @ b
    gs_transf_perm = np.concatenate((alpha,C))
    gs_transf = P.T @ gs_transf_perm
    gs = inv_transform @ gs_transf
    # # gs = gs.reshape((len(gs)//6,6))
    
    # constant energies
    F_const_C =  0.5 * C.T @ MC @ C
    F_const_b = -0.5 * b.T @ MFi @ b
    
    # entropy term
    n = len(MF)
    logdet_sign, logdet = np.linalg.slogdet(MF)
    F_pi = -0.5*n * np.log(2*np.pi)
    # matrix term
    F_mat = 0.5*logdet
    F_entropy = F_pi + F_mat
    F_jacob = np.log(np.linalg.det(transform))
    
    # free energy of unconstrained DNA
    ff_logdet_sign, ff_logdet = np.linalg.slogdet(stiffmat)
    ff_pi = -0.5*len(stiffmat) * np.log(2*np.pi)
    F_free = 0.5*ff_logdet + ff_pi
     
    # prepare output
    Fdict = {
        'F': F_entropy + F_jacob + F_const_C + F_const_b,
        'F_entropy'  : F_entropy + F_jacob,
        'F_enthalpy' : F_const_C + F_const_b,
        'F_jacob'    : F_jacob,
        'F_freedna'  : F_free,
        'gs'         : gs
    }
    return Fdict



if __name__ == '__main__':
    
    from .nuctriads import read_nucleosome_triads
    from .RBPStiff.read_params import GenStiffness
    # from .PolyCG.polycg.cgnaplus import cgnaplus_bps_params
        
    np.set_printoptions(linewidth=250,precision=3,suppress=True)
    
    genstiff = GenStiffness(method='MD')
    
    randseq = ''.join(['ATCG'[np.random.randint(4)] for i in range(147)])
    seq601 = "ATCGAGAATCCCGGTGCCGAGGCCGCTCAATTGGTCGTAGACAGCTCTAGCACCGCTTAAACGCACGTACGCGCTGTCCCCCGCGTTTTAACCGCCAAGGGGATTACTCCCTAGTCTCCAGGCACGTGTCAGATATATACATCCGAT"
    
    # seq = randseq
    seq = seq601
    
    beta = 1./4.114
    stiff,gs = genstiff.gen_params(seq)
    
    triadfn = os.path.join(os.path.dirname(__file__), 'Parameters/Nucleosome.state')
    nuctriads = read_nucleosome_triads(triadfn)

    midstep_constraint_locations = [
        2, 6, 14, 17, 24, 29, 
        34, 38, 45, 49, 55, 59, 
        65, 69, 76, 80, 86, 90, 
        96, 100, 107, 111, 116, 121, 
        128, 131, 139, 143
    ]
        
    Fdict  = hc_free_energy(
        gs,
        stiff,
        midstep_constraint_locations, 
        nuctriads
    )
    
    for key in Fdict:
        print(f'{key} = {Fdict[key]}')
