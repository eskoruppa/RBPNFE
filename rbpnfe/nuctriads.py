import numpy as np
from .SO3 import so3

def read_nucleosome_triads(fn: str) -> np.ndarray:
    data = np.loadtxt(fn)
    N = len(data) // 12
    nuctriads = np.zeros((N,4,4))
    for i in range(N):
        tau = np.eye(4)
        pos   = data[i*12:i*12+3] / 10
        triad = data[i*12+3:i*12+12].reshape((3,3))
        triad = so3.euler2rotmat(so3.rotmat2euler(triad))
        tau[:3,:3] = triad
        tau[:3,3]  = pos
        nuctriads[i] = tau
    return nuctriads