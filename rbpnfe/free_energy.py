from __future__ import annotations

import sys, os, time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import List, Tuple, Callable, Any, Dict, Sequence

from .hcmodel import hc_free_energy, hc_fe_compse3
from .scmodel import sc_free_energy

from .midstep_composites import calculate_midstep_triads
from .nuctriads import read_nucleosome_triads
from .RBPStiff.read_params import GenStiffness
from .PolyCG.polycg.cgnaplus import cgnaplus_bps_params
from .PolyCG.polycg.gen_params import gen_params as cgnap_gen_params
from .utils.rescale_stiffness import rescale_stiff_dofs
from .PolyCG.polycg.utils.console_output import ProgressBar


_LANDSCAPE_CTX: dict = {}
def _landscape_worker(i: int) -> tuple[int, float, float, float]:
    ctx = _LANDSCAPE_CTX
    nfe     = ctx['nfe']
    gs      = ctx['gs']
    stiff   = ctx['stiff']
    verbose = ctx['verbose']
    if verbose:
        t1 = time.time()
    pgs    = gs[i:i+147]
    pstiff = stiff[6*i:6*(i+147), 6*i:6*(i+147)]
    nfe_out = nfe._eval_single(
        pgs,
        pstiff,
        open_left=ctx['open_left'],
        open_right=ctx['open_right'],
        shl_open_left=ctx['shl_open_left'],
        shl_open_right=ctx['shl_open_right'],
        use_correction=ctx['use_correction'],
    )
    if verbose:
        t2 = time.time()
        seq = ctx['seq']
        print(f'Position {i}: {seq[i:i+147]} | Free Energy: {nfe_out["F"]:.2f} kT | Enthalpy: {nfe_out["F_enthalpy"]:.2f} kT | Time: {t2-t1:.4f} s', flush=True)
    return i, nfe_out['F'], nfe_out['F_fluctuation'], nfe_out['F_enthalpy']


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
        flanking: int = 10,
        mode: str = 'compse3',
        cgnaplus_setname: str = 'curves_plus',
        rescale_factors: Sequence[float] | None = None,
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

        self.eval_mode = mode.lower()
        self.cgnaplus_setname = cgnaplus_setname
        self.rescale_factors = rescale_factors

        if self.rescale_factors is not None:
            rescale_arr = np.asarray(self.rescale_factors, dtype=float)
            if rescale_arr.ndim != 1 or rescale_arr.shape[0] != 6:
                raise ValueError(
                    f'rescale_factors must be a sequence of length 6, '
                    f'got {self.rescale_factors}'
                )
            if not np.all(rescale_arr > 0):
                raise ValueError(
                    f'rescale_factors must contain only positive entries, '
                    f'got {self.rescale_factors}'
                )


    
    def _eval_single(
        self,
        gs: np.ndarray,
        stiff: np.ndarray,
        open_left: int = 0,
        open_right: int = 0,
        shl_open_left:  int | None = None,
        shl_open_right: int | None = None,
        use_correction: bool = True
    ) -> dict[str]:
        
        if len(gs) < 146:
            raise ValueError(f'Provided sequence needs to be of length 147. Provided sequence has length {len(gs)+1}')
        
        if stiff.shape[0] != 6*len(gs) or stiff.shape[1] != 6*len(gs):
            raise ValueError(f'Provided stiffness matrix needs to be of shape (6*len(seq),6*len(seq)). Provided stiffness matrix has shape {stiff.shape}')


        if shl_open_left is not None:
            open_left  = shl_open_left * 2
        if shl_open_right is not None:
            open_right = shl_open_right * 2
            
        if open_left + open_right > 28:
            raise ValueError('The number of open binding sites cannot exceed 28')
        
        if self.hardconstraint:
            midloc = self.midstep_locations[open_left:len(self.midstep_locations)-open_right]
            
            if self.eval_mode == 'compse3':
                nucout = hc_fe_compse3(
                    gs,
                    stiff,
                    midloc, 
                    self.nuctriads,
                    use_correction=use_correction
                )
            elif self.eval_mode == 'legacy':
                nucout  = hc_free_energy(
                    gs,
                    stiff,
                    midloc, 
                    self.nuctriads,
                    use_correction=use_correction
                )
            else:
                raise ValueError(f'Unknown eval_mode for hard constraint "{self.eval_mode}"')
            
        else:
            nucout = sc_free_energy(
                gs,
                stiff,    
                self.nuc_mu0,
                self.Kmat,
                left_open=open_left,
                right_open=open_right,
                base_midstep_locations=self.midstep_locations,
                use_correction=use_correction
            )       
        return nucout
        
    

    def eval(
        self, 
        seq: str,
        open_left: int = 0,
        open_right: int = 0, 
        shl_open_left:  int | None = None,
        shl_open_right: int | None = None,
        use_correction: bool = True
        ) -> dict[str]:
        """
        Evaluate nucleosome free energy for a given sequence and open binding sites.

        Args:
            seq (str): DNA sequence of length 147 bp.
            open_left (int, optional): Number of open binding sites on the left. Defaults to 0.
            open_right (int, optional): Number of open binding sites on the right. Defaults to 0.
            shl_open_left (int | None, optional): Number of open superhelical locations on the left. Defaults to None.
            shl_open_right (int | None, optional): Number of open superhelical locations on the right. Defaults to None.
            use_correction (bool, optional): Whether to use correction in free energy calculation. Defaults to True.

        Returns:
            dict[str]: Dictionary containing free energy components.
        """
        
        if shl_open_left is not None:
            open_left  = shl_open_left * 2
        if shl_open_right is not None:
            open_right = shl_open_right * 2
            
        if open_left + open_right > 28:
            raise ValueError('The number of open binding sites cannot exceed 28')
        
        if len(seq) != 147:
            raise ValueError(f'Provided sequence needs to be of length 147. Provided sequence has length {len(seq)}')
        
        gs,stiff = self.gen_params(seq,flanking=self.flanking)
        return self._eval_single(
            gs,
            stiff,
            open_left=open_left,
            open_right=open_right,
            shl_open_left=shl_open_left,
            shl_open_right=shl_open_right,
            use_correction=use_correction
        )
    

    def eval_landscape(
        self, 
        seq: str,
        open_left: int = 0,
        open_right: int = 0, 
        shl_open_left:  int | None = None,
        shl_open_right: int | None = None,
        use_correction: bool = True,
        ncores: int = 1,
        verbose: bool = False,
        progress: bool = True
        ) -> dict[str]:
        
        if shl_open_left is not None:
            open_left  = shl_open_left * 2
        if shl_open_right is not None:
            open_right = shl_open_right * 2
            
        if open_left + open_right > 28:
            raise ValueError('The number of open binding sites cannot exceed 28')
        
        gs,stiff = self.gen_params(seq,flanking=self.flanking)
        Nnucs = len(seq) - 147 + 1
        if Nnucs < 1:
            raise ValueError(f'Provided sequence needs to be of length at least 147. Provided sequence has length {len(seq)}')
        
        fes = np.zeros((Nnucs,3), dtype=np.float64)

        if ncores < 1:
            raise ValueError(f'ncores must be a positive integer, got {ncores}')

        def _eval_position(i: int) -> tuple[int, float, float, float]:
            if verbose: t1 = time.time()
            pgs = gs[i:i+147]
            pstiff = stiff[6*i:6*(i+147),6*i:6*(i+147)]
            nfe = self._eval_single(
                    pgs,
                    pstiff,
                    open_left=open_left,
                    open_right=open_right,
                    shl_open_left=shl_open_left,
                    shl_open_right=shl_open_right,
                    use_correction=use_correction
                )
            if verbose:
                t2 = time.time()
                print(f'Position {i}: {seq[i:i+147]} | Free Energy: {nfe["F"]:.2f} kT | Enthalpy: {nfe["F_enthalpy"]:.2f} kT | Time: {t2-t1:.4f} s')
            return i, nfe['F'], nfe['F_fluctuation'], nfe['F_enthalpy']

        show_bar = progress and not verbose
        bar = None
        if show_bar:
            # flush=True and an immediate 0% frame so the header and bar appear
            # right away, even when the first position is slow to evaluate (e.g.
            # the soft-constraint model) or stdout is block-buffered.
            print(f'Computing Free Energy Landscape for {len(seq)} base pair sequence ({Nnucs} positions)', flush=True)
            bar = ProgressBar(Nnucs, prefix='Progress:', show_eta=True)
            bar.update(0, suffix=f'Position 0/{Nnucs}')

        def _store(done: int, res: tuple[int, float, float, float]) -> None:
            i, F, F_fluctuation, F_enthalpy = res
            fes[i, 0] = F
            fes[i, 1] = F_fluctuation
            fes[i, 2] = F_enthalpy
            if bar is not None:
                bar.update(done, suffix=f'Position {done}/{Nnucs}')

        if ncores > 1:
            _LANDSCAPE_CTX.update(
                nfe=self,
                gs=gs,
                stiff=stiff,
                seq=seq,
                open_left=open_left,
                open_right=open_right,
                shl_open_left=shl_open_left,
                shl_open_right=shl_open_right,
                use_correction=use_correction,
                verbose=verbose,
            )
            mp_context = mp.get_context('fork')
            try:
                with ProcessPoolExecutor(max_workers=min(ncores, Nnucs),
                                         mp_context=mp_context) as executor:
                    futures = [executor.submit(_landscape_worker, i)
                               for i in range(Nnucs)]
                    for done, future in enumerate(as_completed(futures), start=1):
                        _store(done, future.result())
            finally:
                _LANDSCAPE_CTX.clear()
        else:
            for done, i in enumerate(range(Nnucs), start=1):
                _store(done, _eval_position(i))

        return fes

    
        
    def gen_params(self,seq: str,flanking: int=10) -> tuple[np.ndarray,np.ndarray]:
        if self.params_model in self.cgnaplus_names:
            if flanking > 0:
                flank = ('CG' * int(np.ceil(flanking / 2)))[:flanking]
                fseq = flank + seq + flank
            else:
                fseq = seq

            if len(seq) > 200: 
                gs, stiff = cgnap_gen_params(
                    'cgnaplus',
                    fseq, 
                    cgnap_setname=self.cgnaplus_setname,
                    verbose=True).get_params()

            else:
                gs,stiff = cgnaplus_bps_params(
                    fseq,
                    group_split=True,
                    parameter_set_name = self.cgnaplus_setname)
                
            if flanking > 0:
                stiff = stiff[6*flanking:-6*flanking,6*flanking:-6*flanking]
                gs = gs[flanking:-flanking]

        else:
            prms = self.genstiff.gen_params(seq,use_group=True)
            gs    = prms['groundstate']
            stiff = prms['stiffness']

        if self.rescale_factors is not None:
            stiff = rescale_stiff_dofs(stiff, self.rescale_factors)
        
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
    
    nout = nfe.eval(
        seq,
        shl_open_left = shl_open_left,
        shl_open_right = shl_open_right,
        use_correction = True
        )
    
    print(nout['dF'])
    print(nout['F'])
    print(nout['F_freedna'])