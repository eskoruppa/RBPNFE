import sys, os
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import List, Tuple, Callable, Any, Dict

from .hcmodel import hc_free_energy
from .scmodel import sc_free_energy

from .nuctriads import read_nucleosome_triads
from .RBPStiff.read_params import GenStiffness
from .PolyCG.polycg.cgnaplus import cgnaplus_bps_params


class NucFreeEnergy:
    
    cgnaplus_names = ['cgnap','cgnaplus','cgna+']
    genstiff_names = ['hybrid','crystal','olson','md','lankas']
    
    def __init__(
        self,
        params_model : str = 'hybrid',
        hardconstraint: bool = False,
        midstep_constraint_locations: List[int] = None,
        triadfn: str = None,
        Kmat_file: str = None
        ):
        
        self.params_model = params_model.lower()
        if self.params_model in self.genstiff_names:
            self.genstiff = GenStiffness(method=params_model)
        elif self.params_model not in self.cgnaplus_names:
            raise ValueError(f'Unknown params_method "{params_model}"')
        else:
            self.genstiff = None
            
        
        
        
    def gen_params(self,seq: str,flanking: int=10):
        if self.params_model in self.cgnaplus_names:
            if len(seq) < 167:
                flank = ('CG' * int(np.ceil(flanking / 2)))[:flanking]
                fseq = flank + seq + flank
                gs,stiff = cgnaplus_bps_params(fseq,group_split=True)

                print(gs.shape)
                print(stiff.shape)
                
                sys.exit()
                
                stiff = stiff[6*flanking:-6*flanking,6*flanking:-6*flanking]
                gs = gs[flanking:-flanking]
            
                print(gs.shape)
                print(stiff.shape)
            else:
                fs,stiff = cgnaplus_bps_params(seq,group_split=True)
                
        else:
            gs,stiff = self.genstiff.gen_params(seq,use_group=True)
        return gs,stiff


    





# def free_energy(
#     sequence: str,
#     model : str = 'hybrid',
#     shl_open_left:  int = 0,
#     shl_open_right: int = 0,
#     hardconstraint: bool = False,
#     midstep_constraint_locations: List[int] = None,
#     triadfn: str = None,
#     Kmat_file: str = None
# ):
    
    


if __name__ == '__main__':
    
    
    nfe = NucFreeEnergy(
        params_model = 'cgna+',
        hardconstraint=True,
        )

    
    seq  = "CTGGAGAATCCCGGTGCCGAGGCCGCTCAATTGGTCGTAGACAGCTCTAGCACCGCTTAAACGCACGTACGCGCTGTCCCCCGCGTTTTAACCGCCAAGGGGATTACTCCCTAGTCTCCAGGCACGTGTCAGATATATACATCCTGT"
    
    nfe.gen_params(seq,flanking=9)