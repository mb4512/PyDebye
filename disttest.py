import sys, time, math, cmath
import numpy as np


from mpi4py import MPI
from mpi4py_fft import PFFT, newDistArray, DistArray

comm = MPI.COMM_WORLD
me = comm.Get_rank()
nprocs = comm.Get_size()

from lib.distarray import DistributedTensor 

def mpiprint(*arg, **kwargs):
    if me == 0:
        print(*arg, **kwargs)
        sys.stdout.flush()
    return 0

class Store:
    def __init__(self):
        pass


def fftDist(shape, distaxis=0):
 
    # create distributed array along axis 0
    ngroups = [1]*len(shape)
    ngroups[distaxis] = 0

    uvox = DistArray(shape, ngroups, dtype=float)
    uvox[:] = me 
    print ("pre", me, uvox.shape)
    sys.stdout.flush()

    # gather slab thicknesses into one list
    indexlist_ini = comm.allgather(uvox.shape[distaxis])
    mpiprint ("Voxel grid was distributed into slabs along axis {} of length: {}".format(distaxis, indexlist_ini))

    # align along the previously distributed axis 
    uvox = uvox.redistribute(distaxis)

    print ("post", me, uvox.shape)
    sys.stdout.flush()
 
    # return thicknesses of slabs for the two distributions so we can initialise 
    # our slabs with the bespoke method with consistent dimensions
    return uvox, indexlist_ini


def main():
    #shape = (200,236,261)
    shape = (52,23,73,261)

    # if storage is initialised, the two resulting arrays will be compared
    # care: disk access for large voxel arrays (500^3+) and hence inaccurate timings
    storage = Store()
    #storage = False

    # initial axis of distribution
    axinitial = 1

    # final axis of distribution
    axfinal = 0

    mpiprint ("Voxel shape:", shape) 

    gbsize = 8*np.product(shape)*1e-9/nprocs
    mpiprint("\nPredicted size per rank: %.4f GB" % gbsize)

    mpiprint("\nMPI4PY-FTW distribution:")
    mpiprint("==================\n")

    clock = time.time()
    uvox, indexlist_ini = fftDist(shape, distaxis=axinitial)
    clock = time.time() - clock

    # gather slab thicknesses into one list
    indexlist_fin = comm.allgather(uvox.shape[axfinal])
    mpiprint ("Voxel grid was distributed into slabs along axis {} of length: {}".format(axfinal, indexlist_fin))

    if storage:
        Store.uvox1 = np.copy(uvox)

    time.sleep(0.01)
    comm.barrier()
    sys.stdout.flush()
    mpiprint ("\nTime for MPI4PY-FFT redistribution: %.5fs" % clock)
   
    mpiprint("\nModular distribution:")
    mpiprint("=====================\n")
    comm.barrier()
    sys.stdout.flush()

    clock = time.time()
    uvoxdist = DistributedTensor(shape, distaxis=axinitial, slabwidths=indexlist_ini, value=me)
    print ("pre", me, uvoxdist.array.shape)
    sys.stdout.flush()

    uvoxdist.redistribute(axfinal, slabwidths=indexlist_fin)
    print ("post", me, uvoxdist.array.shape)
    sys.stdout.flush()

    clock = time.time() - clock

    time.sleep(0.01)
    comm.barrier()
    sys.stdout.flush()
    mpiprint ("\nTime for modular redistribution: %.5fs" % clock)

    # raise error if local mpi4py-fft arrays and bespoke arrays are different
    if storage:
        np.testing.assert_equal(storage.uvox1.flatten(), uvoxdist.array.flatten())
        comm.barrier()
        mpiprint ("\nThe two redistributed arrays are equivalent.")
    else:
        mpiprint ("\nStorage was set to false, the resulting arrays are not compared.")


    mpiprint ()
    MPI.Finalize()
    return 0

if __name__=="__main__":
    main ()


