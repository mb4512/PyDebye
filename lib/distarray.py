import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
me = comm.Get_rank()
nprocs = comm.Get_size()


class DistributedTensor:
    def __init__(self, shape, slabwidths=None, distaxis=0, numpytype=float, value=None):
        """
        Initialise a multidimensional tensor distributed over MPI threads with the specified properties.

        Parameters
        ----------
        shape : tuple, list, or numpy array of ints
            The global shape of the tensor, e.g. (100, 50, 700).
        slabwidths : list or tuple or numpy array of ints, optional, e.g. (100, 40, 20, ...).
            The width of each slab along the distribution axis. If not provided, slabs will be of approximately even width.
        distaxis : int, optional
            The index of the axis along which to distribute the tensor. The default is 0.
        numpytype : numpy.dtype, optional
            The numpy data type of the tensor elements. The default is float.
        value : scalar, optional
            The initial value of the tensor elements. If not provided, the tensor is initialised with empty elements.

        Returns
        -------
        None

        Properties
        ----------
        self.array : Reference to the local array on the current MPI thread.
        self.global_shape : The global shape of the tensor.
        self.dim : The number of dimensions of the tensor
        self.distaxis : The index of the axis along with the tensor is distributed.
        self.nspans : List of indices deliminating the local slab positions in the global array. 
        self.nwidths : The widths of the slabs along the distribution axis.
        self.local_shape : The local shape of the tensor on this current MPI thread.

        Raises
        ------
        ValueError
            If the supplied slab widths do not add up to the length of the distribution axis.
            If the number of supplied slab are not equal to the number of MPI threads. 
        """

        # save input as properties 
        self.global_shape = shape
        self.dim = len(shape)

        # initially supplied slabwidths are kept consistent through
        # the lifetime of the array for internal consistency    
        # TODO: Not yet implemented, only needed once distribution along other axis is supported
        self.slabwidths = slabwidths

        # self.distaxis stores the current axis of distribution, i.e. it is updated after redistribution
        self.distaxis = distaxis

        # plan out slab widths and global indices spanning the slabs along the distribution axis 
        self.nspans, self.nwidths = self._plan_shape(slabwidths, distaxis)

        # allocate local arrays distributed along specified axis
        self.local_shape = np.copy(shape)
        self.local_shape[distaxis] = self.nwidths[me]

        # initialise empty local arrays
        if value is not None:
            self.array = value*np.ones(self.local_shape, dtype=numpytype)
        else:
            self.array = np.empty(self.local_shape, dtype=numpytype)

        # gather slab thicknesses into one list for easy access
        self.indexlist = comm.allgather(self.array.shape[distaxis])


    def redistribute(self, new_distaxis, slabwidths=None):
        """
        Redistribute the tensor along a new distribution axis. Currently only supports redistribution from any axis to last axis.

        Parameters
        ----------
        new_distaxis : int
            The index of the new distribution axis.
        slabwidths : list or tuple of ints, optional
            The width of each slab along the distribution axis. If not provided, slabs will be of approximately even width.

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            If the distribution axis is not the last axis.
        """

        # currently supports only redistribution to last axis 
        if new_distaxis == self.dim-1:
            new_distaxis = -1 
        assert new_distaxis == -1, "Currently only redistribution to last axis (-1, i.e. {} here) is supported.".format(self.dim-1)

        # skip redistribution if initial and final disributed axes are the same 
        if self.distaxis == new_distaxis:
            return None 

        # plan out new slab widths and global indices spanning the slabs along the distribution axis 
        nspans, nwidths = self._plan_shape(slabwidths, new_distaxis)

        # initialise receive buffer
        new_local_shape = np.copy(self.global_shape)
        new_local_shape[new_distaxis] = nwidths[me]

        nrecv = np.product(new_local_shape)
        recv = np.empty(nrecv, dtype=self.array.dtype)
        reqarray = [0]*nprocs

        # if we are distributing from an axis that is not the first axis, transpose such that it is the case
        if self.distaxis != 0:
            permute = list(range(self.dim))
            permute[0], permute[self.distaxis] = permute[self.distaxis], permute[0]
            self.array = np.transpose(self.array, permute) 

        # let every rank send their data between the index spans of the distributed axis into rank pi
        for pi in range(nprocs):

            # sending from all ranks to rank pi
            ni,nis = nspans[pi],nspans[pi+1]

            send_shapes = np.r_[[np.copy(self.global_shape) for pi in range(nprocs)]]
            send_shapes[:,self.distaxis] = self.nwidths
            send_shapes[:,new_distaxis]  = nwidths[pi]
            nsend = np.product(send_shapes, axis=1)

            # nonblocking gather
            # could be faster if we had a contiguous 1D view of the multidimensional slice without copying (how??)
            reqarray[pi] = comm.Igatherv(sendbuf=self.array[...,ni:nis].flatten(), recvbuf=(recv, nsend), root=pi)

        # make sure each rank has received all their data
        MPI.Request.waitall(reqarray)

        # reshape the receive buffer to the new slab dimensions (in-place)
        recv_Nd = recv.view()

        new_local_shape = np.copy(self.global_shape)
        new_local_shape[new_distaxis]  = nwidths[me]

        # if we distributed from an axis that was not the first axis, our local shape now must also be transposed 
        if self.distaxis != 0:
            new_local_shape[0], new_local_shape[self.distaxis] = new_local_shape[self.distaxis], new_local_shape[0]

        # reshape received arrays into the right slab shape
        recv_Nd.shape = new_local_shape 

        # if we transposed the local arrays earlier, transform them back now to the right shape 
        if self.distaxis != 0:
            recv_Nd = np.transpose(recv_Nd, permute) 

        # delete reference to buffered voxel grid, enabling garbage collection
        self.array, recv_Nd = recv_Nd, None

        # set new shapes and distribution axis
        self.distaxis = new_distaxis
        self.slabwidths = nwidths
        self.nspans = nspans
        self.nwidths = nwidths 
        self.local_shape = self.array.shape

        return 0
      

    def _plan_shape(self, slabwidths, distaxis):
        nglobal = self.global_shape[distaxis]
        if slabwidths is not None:

            if (len(slabwidths) != nprocs): 
                raise ValueError("Supplied index list is of inequal length ({}) as the number of MPI threads ({})!".format(len(slabwidths), nprocs))

            if (np.sum(slabwidths) != nglobal): 
                raise ValueError("Supplied index list has cumulative dimension {} inconsistent " \
                                 "with global grid dimension {} along distributed axis {}.".format(np.sum(slabwidths), nglobal, distaxis))

            nspans = np.r_[0, np.cumsum(slabwidths)]
            nwidths = slabwidths 
        else:
            nspans = ((nglobal/nprocs)*np.arange(0,nprocs+1)).astype(int) 
            nwidths = nspans[1:] - nspans[:-1]
 
        return nspans, nwidths

def main ():
    '''To test this, run mpirun -n X python distarray.py, where X is number of MPI threads.'''
    import sys, time
    
    shape = (400,436,461)
    
    # initialise distributed array
    clock = time.time()
    uvoxdist = DistributedTensor(shape, distaxis=0, value=me)
    clock_ini = time.time() - clock    

    sys.stdout.flush()
    print ("initialised local array on rank {} with shape {}.".format(me, uvoxdist.array.shape))

    # redistribute array
    clock = time.time()
    uvoxdist.redistribute(-1)
    clock_dis = time.time() - clock    

    sys.stdout.flush()
    print ("redistributed local array on rank {} with shape {}.".format(me, uvoxdist.array.shape))

    sys.stdout.flush()
    comm.barrier()

    if (me == 0):
        print ()
        print ("initialisation time: %.3f seconds" % clock_ini)
        print ("distribution time:   %.3f seconds" % clock_dis)
        print ()

    MPI.Finalize()
    return 0


if __name__=="__main__":
    main ()
