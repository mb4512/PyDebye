import sys, os

import numpy as np

from lib.computespec import ComputeSpectrum
from lib.readfile import ReadFile

# template to replace MPI functionality for single threaded use
class MPI_to_serial():
    def bcast(self, *args, **kwargs):
        return args[0]
    def barrier(self):
        return 0

# try running in parallel, otherwise single thread
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    me = comm.Get_rank()
    nprocs = comm.Get_size()
    mode = 'MPI'
except:
    me = 0
    nprocs = 1
    comm = MPI_to_serial()
    mode = 'serial'

def mpiprint(*arg):
    if me == 0:
        print(*arg)
        sys.stdout.flush()
    return 0



def main():
    
    fpath = 'lammpsfiles/vacanneal_slow_9_1800.relax'

    filedat = ReadFile(fpath, filetype="restart")
    filedat.load(shuffle=True)

    cspec = ComputeSpectrum(filedat, rpartition=10., pbc=True, rcut=30.0)
    cspec.build_histogram(dr=0.001)

    spectrum = cspec.build_debyescherrer(.3, 1.2, 200, damp=True, ccorrection=True)

    if (me == 0):
        np.savetxt("spectrum.out", spectrum)

    return 0

if __name__ == "__main__":
    main()
    if mode == 'MPI':
        MPI.Finalize()



