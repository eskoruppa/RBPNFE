import sys, os
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import List, Tuple, Callable, Any, Dict

from .hcmodel import hc_free_energy
from .scmodel import sc_free_energy

from .midstep_composites import calculate_midstep_triads
from .nuctriads import read_nucleosome_triads
from .RBPStiff.read_params import GenStiffness
from .PolyCG.polycg.cgnaplus import cgnaplus_bps_params


class NucFreeEnergy:
    
    cgnaplus_names = ['cgnap','cgnaplus','cgna+']
    genstiff_names = ['hybrid','crystal','olson','md','lankas']
    
    midstep_locations = [
        2, 6, 14, 17, 24, 29, 
        34, 38, 45, 49, 55, 59, 
        65, 69, 76, 80, 86, 90, 
        96, 100, 107, 111, 116, 121, 
        128, 131, 139, 143
    ]
    
    Kmat_file = os.path.join(os.path.dirname(__file__), 'Parameters/MDParams/nuc_K_pos_resc_sym.npy') 
    
    def __init__(
        self,
        params_model : str = 'hybrid',
        hardconstraint: bool = False,
        midstep_locations: List[int] = None,
        triadfn: str = None,
        Kmat_file: str = None,
        flanking: int = 10
        ):
        
        # parameter config        
        self.params_model = params_model.lower()
        if self.params_model in self.genstiff_names:
            self.genstiff = GenStiffness(method=params_model)
        elif self.params_model not in self.cgnaplus_names:
            raise ValueError(f'Unknown params_method "{params_model}"')
        else:
            self.genstiff = None
        
        # load K matrix
        if Kmat_file is not None:
            self.Kmat_file = Kmat_file
        self.Kmat = np.load(self.Kmat_file)
        
        # set defines
        self.hardconstraint = hardconstraint
        self.flanking = flanking
        if midstep_locations is not None:
            self.midstep_locations = midstep_locations
        
        if triadfn is None:
            triadfn = os.path.join(os.path.dirname(__file__), 'Parameters/Nucleosome.state')
        self.nuctriads = read_nucleosome_triads(triadfn)
    
        self.nuc_mu0 = calculate_midstep_triads(
            self.midstep_locations,
            self.nuctriads
        )
    

    def eval(
        self, 
        seq: str, 
        shl_open_left:  int = 0,
        shl_open_right: int = 0,
        use_correction: bool = True
        ):
        
        if shl_open_left + shl_open_right > 14:
            raise ValueError('The number of open superhelcial locations cannot exceed 14')
        
        gs,stiff = self.gen_params(seq,flanking=self.flanking)
        
        if self.hardconstraint:
            midloc = self.midstep_locations[shl_open_left*2:len(self.midstep_locations)-2*shl_open_right]
            nucout  = hc_free_energy(
                gs,
                stiff,
                midloc, 
                self.nuctriads,
                use_correction=use_correction
            )
            
        else:
            nucout = sc_free_energy(
                gs,
                stiff,    
                self.nuc_mu0,
                self.Kmat,
                left_open=shl_open_left*2,
                right_open=shl_open_right*2,
                base_midstep_locations=self.midstep_locations,
                use_correction=use_correction
            )       
        return nucout
       
        
    def gen_params(self,seq: str,flanking: int=10):
        if self.params_model in self.cgnaplus_names:
            if flanking > 0:
                flank = ('CG' * int(np.ceil(flanking / 2)))[:flanking]
                fseq = flank + seq + flank
                gs,stiff = cgnaplus_bps_params(fseq,group_split=True)
                # stiff *= 0.75
                stiff = stiff[6*flanking:-6*flanking,6*flanking:-6*flanking]
                gs = gs[flanking:-flanking]
            else:
                gs,stiff = cgnaplus_bps_params(seq,group_split=True)     
        else:
            prms = self.genstiff.gen_params(seq,use_group=True)
            gs    = prms['groundstate']
            stiff = prms['stiffness']
        return gs,stiff




if __name__ == '__main__':
    
    
    params_model = 'cgna+'
    # params_model = 'MD'
    hard_constraint = False
    
    nfe = NucFreeEnergy(
        params_model = params_model,
        hardconstraint=hard_constraint,
        )
    # nfe.gen_params(seq,flanking=9)

    seq  = "CTGGAGAATCCCGGTGCCGAGGCCGCTCAATTGGTCGTAGACAGCTCTAGCACCGCTTAAACGCACGTACGCGCTGTCCCCCGCGTTTTAACCGCCAAGGGGATTACTCCCTAGTCTCCAGGCACGTGTCAGATATATACATCCTGT"
    seq  = "ATCGAGAATCCCGGTGCCGAGGCCGCTCAATTGGTCGTAGACAGCTCTAGCACCGCTTAAACGCACGTACGCGCTGTCCCCCGCGTTTTAACCGCCAAGGGGATTACTCCCTAGTCTCCAGGCACGTGTCAGATATATACATCCGAT"
    
    
    shl_open_left = 0
    shl_open_right = 0
    
    nout = nfe.eval_single(
        seq,
        shl_open_left = shl_open_left,
        shl_open_right = shl_open_right,
        use_correction = True
        )
    
    print(nout['dF'])
    print(nout['F'])
    print(nout['F_freedna'])