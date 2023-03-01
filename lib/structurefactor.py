import sys, time
import numpy as np
from numba import jit

from scipy.spatial import cKDTree

# try importing parallel FFTW library
try:
    from mpi4py_fft import PFFT, newDistArray
    IMPORTED = "successful"
except ImportError:
    IMPORTED = ImportError("Could not load mpi4py_fft library. Is it installed?")

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

# global blocking size (in number of elements) used throughout this code
BSIZE = 10000

@jit(nopython=True, fastmath=True)
def jdisgauss(xyz, _r, sigma):
    _dr = (xyz-_r).T
    gaussdensity = np.sum(np.exp(-(_dr[0]*_dr[0] + _dr[1]*_dr[1] + _dr[2]*_dr[2])/(2.*sigma*sigma)))
    return gaussdensity


#@jit(nopython=True, cache=True, fastmath=True)
@jit(nopython=True, fastmath=True)
def jpbc_wrap(xyz, box):
    '''Wraps xyz coordinates back into periodic box so KDTree with PBC on does not complain.'''
    for i in range(len(xyz)): 
        xyz[i] = xyz[i]-box*np.floor(xyz[i]/box)


class StructureFactor:
    def __init__(self, readfile, pbc=False, precision="double", dx=0.3):

        # check if library available
        if IMPORTED != "successful":
            raise IMPORTED

        # store reference to readfile instance
        self.readfile = readfile 
 
        # if pbc, wrap atoms back into box
        if pbc:
            mpiprint ("Rewrapping atoms back into periodic box...")
            # compile first with small test set, then rewrap all atoms
            ncomp = min([10,self.readfile.natoms])
            jpbc_wrap(self.readfile.xyz[:ncomp], self.readfile.box) 
            jpbc_wrap(self.readfile.xyz, self.readfile.box) 

        comm.barrier()
        mpiprint ("done.\n")
       
        self.dx = dx # voxel spacing in Angstrom 
        self.pbc = pbc

        # change precision if single precision mode is requested 
        mpiprint ("Constructing voxel field with dx=%.3f voxel spacing and computing FFT.\n" % self.dx)

        ''' # TODO: test whether large grids need long float precision
        if precision == "single":
            self.single = True
            mpiprint ("Constructing voxel field and computing FFT in single precision mode.\n")
        else:
            self.single = False
            mpiprint ("Constructing voxel field and computing FFT in double precision mode.\n")

        if self.single:
            self.readfile.xyz = self.readfile.xyz.astype(np.single)
        '''

        comm.barrier() 
 
    def prepare_voxels(self):
        '''Set dimensions of the 3D voxel field. Valid for both orthogonal and nonorthogonal cells.'''

        xyz = self.readfile.xyz
        cmat = self.readfile.cmat

        # gaussian standard-deviation in angstrom
        self.sig = self.dx/2

        # gaussian cutoff is 4 standard deviations
        # note that the higher the cutoff, the more accurate the summation
        # density = -exp(-.5*q^2) sqrt(2/pi)*q + erf(q/sqrt(2))
        # 3 std: 97.07% density
        # 4 std: 99.88% density
        # 5 std: 99.9985% density
        self.rc = 4*self.sig

        # we need a regular and evenly spaced mesh in a bounding box enclosing all atoms in the system
        bmin = np.min(xyz, axis=0)
        bmax = np.max(xyz, axis=0)

        # include a buffer region to make sure we do not cut off any broadened atomic densities
        #rbuffer = 1.5*self.rc
        rbuffer = 0.0

        self.bmin = bmin - rbuffer
        self.bmax = bmax + rbuffer

        # voxel mesh has dimensions of (N_vox, 3)
        self.v0 = np.arange(self.bmin[0], self.bmax[0], self.dx)
        self.v1 = np.arange(self.bmin[1], self.bmax[1], self.dx)
        self.v2 = np.arange(self.bmin[2], self.bmax[2], self.dx)

        # ensure voxel array dimensions are even for easier handling of fft frequencies
        if len(self.v0)%2 == 1:
            self.v0 += 1
        if len(self.v1)%2 == 1:
            self.v1 += 1
        if len(self.v2)%2 == 1:
            self.v2 += 1

        # dimensions of 3D voxel array 
        self.vdim = np.array([len(self.v0), len(self.v1), len(self.v2)], dtype=int) 

 
    def create_voxels(self, slabindex):
        mpiprint ("\nBuilding Gaussian density representation on a voxel grid, distributed over threads.")

        clock = time.time()       
 
        xyz = self.readfile.xyz
        cmat = self.readfile.cmat

        # start and end indices along 1st axis of the voxel tensor 
        # note: index0 and indexE are different on each thread, this is where the parallel distribution occurs
        myshape = self.uvox.shape
        nvox = np.product(self.uvox.global_shape)
        index0 = slabindex[me] 
        indexE = slabindex[me+1]

        # build a KD-tree of all atoms 
        self.xyztree = cKDTree(xyz)

        # normalisation prefactor 
        pref = self.dx**3 * 1/(np.power(2.*np.pi*self.sig**2, 3/2))

        # create a line of voxel coordinates: going along all y indices (2nd axis) while x and y are fixed
        voxelline = np.array([[1., j*self.dx, 1.] for j in range(myshape[1])])
        voxelxyz = np.copy(voxelline)

        # compile gaussian distance function
        mpiprint("Compiling...")
        jdisgauss(xyz[:20], voxelxyz[0], self.sig)
        mpiprint("done.\n")

        # benchmark first
        if (me == 0):
            clock = time.time()
            ilocal = 0
            for i in range(index0, indexE):
                k = 0
                voxelxyz = np.copy(voxelline)
                voxelxyz[:,0] *= i*self.dx
                voxelxyz[:,2] *= k*self.dx
                voxelxyz += self.bmin
                _ixlist = self.xyztree.query_ball_point(voxelxyz, self.rc)
                _nvox = len(voxelxyz)
                self.uvox[ilocal,:,k] = [jdisgauss(xyz[_ixlist[_n]], voxelxyz[_n], self.sig) for _n in range(_nvox)]
                ilocal += 1

            print (ilocal, _nvox, nvox)
            clock = time.time() - clock
            looptime  = clock/(ilocal*_nvox)
            totaltime = looptime * nvox/nprocs # total est. runtime in seconds

            print ("Estimated runtime per voxel: %.1fµs" % (1e6*looptime))
            print ("Estimated total runtime:     %.3fs"  % totaltime)
            print ()
            self.uvox *= 0

        comm.barrier()
        clock = time.time()
    
        # populate voxels with gaussian atomic density
        clock = time.time()
        ilocal = 0
        for i in range(index0, indexE):
            for k in range(myshape[2]):
                voxelxyz = np.copy(voxelline)
                voxelxyz[:,0] *= i*self.dx
                voxelxyz[:,2] *= k*self.dx
                voxelxyz += self.bmin
                _ixlist = self.xyztree.query_ball_point(voxelxyz, self.rc)
                _nvox = len(voxelxyz)
                self.uvox[ilocal,:,k] = [jdisgauss(xyz[_ixlist[_n]], voxelxyz[_n], self.sig) for _n in range(_nvox)]
            ilocal += 1

        # apply prefactor
        self.uvox *= pref

        # get integrated density in this slab
        nlocal = np.sum(self.uvox)

        # here all processes need to finish
        comm.barrier()

        clock = time.time() - clock
        mpiprint ("Populated %d voxels in %.3f seconds.\n" % (nvox, clock))

        # gather all densities
        nlocalarray = comm.allgather(nlocal)
        
        mpiprint ("Total number of atoms:    %d" % len(xyz))
        mpiprint ("Integrated voxel density: %.2f" % np.sum(nlocalarray))
        mpiprint ("Error: %.2f %%" % ((len(xyz)-np.sum(nlocalarray))/len(xyz)*100))
        mpiprint ()

        # renormalise voxeldensities to conserve number of atoms
        self.uvox = self.uvox*len(xyz)/np.sum(nlocalarray)

        return 0


    def build_structurefactor_fftw(self):
 
        # prepare dimensions of the voxel grid
        self.prepare_voxels() 
      
        # initialise distributed tensor on all cores in slab distribution: 
        # the 1st axis of the tensor is distributed among the cores 
        fft = PFFT(MPI.COMM_WORLD, self.vdim, axes=(0, 1, 2), dtype=float, grid=(-1,))
        self.uvox = newDistArray(fft, False)
        myshape = self.uvox.shape
        globalshape = self.uvox.global_shape

        # gather the slab thicknesses into one list
        indexlist = comm.allgather(myshape[0])
        
        mpiprint ("Global shape of the voxel grid:", self.uvox.global_shape, "with total number of voxels:", np.product(self.uvox.global_shape))
        mpiprint ("Voxel grid is distributed into slabs along axis 0 of size:", indexlist)

        # list of voxel slab indices to be constructed by each thread 
        slabindex = np.r_[0, np.cumsum(indexlist)]

        # populate tensor with atomic densities
        self.create_voxels(slabindex)

        mpiprint ()
        mpiprint ("Applying FFT...")

        # obtain structure factor from fourier transform
        self.psi_k = fft.forward(self.uvox)

        #print ("resulting tensor shape on thread %d:" % me, self.psi_k.shape)
        #sys.stdout.flush()

        # define diffraction intensity
        self.int_k = self.psi_k.real*self.psi_k.real + self.psi_k.imag*self.psi_k.imag
        newshape = self.int_k.shape

        # note: after the transformation, the tensor is distributed over axis 1
        newindexlist = comm.allgather(newshape[1])
        newslabindex = np.r_[0, np.cumsum(newindexlist)]

        # kx, ky, kz space resolution
        dk = 1./(self.vdim*self.dx)
        self.kmax = np.linalg.norm(self.uvox.global_shape * dk)

        mpiprint("k-space spacing:", dk)
        mpiprint("maximum k norm:", self.kmax)

        # set some k space resolution for binning
        kres = np.linalg.norm(dk) 
        _spectrum = np.zeros(int(self.kmax/kres)+1)


        # bin diffraction intensity over all k-directions
        for i in range(newshape[0]): 
            for j in range(newshape[1]): 
                for k in range(newshape[2]):

                    # the tensor is distributed over axis 1, hence offset j index appropriately 
                    iv, jv, kv = i, j + newslabindex[me], k

                    # in the other axes, frequencies are ordered as {0, 1, 2, ..., Nx/2, -Nx/2 + 1, ..., -2, -1}.
                    if iv > globalshape[0]/2:
                        iv -= globalshape[0]
                    if jv > globalshape[1]/2:
                        jv -= globalshape[1]
                    if kv > globalshape[2]/2:
                        kv -= globalshape[2]

                    # norm of the k vector belonging to this voxel
                    knorm = np.linalg.norm([dk[0]*iv, dk[1]*jv, dk[2]*kv])

                    # as the input signal is real, we need to double-count all except for the kz=0 value.
                    d3kelement = 4.*np.pi*knorm*knorm*kres

                    if knorm > 0.0:
                        _pref = np.sqrt((dk[0]*iv)*(dk[0]*iv) + (dk[1]*jv)*(dk[1]*jv))/knorm/d3kelement
                    else:
                        _pref = 1.0
                    if k == 0: 
                        _spectrum[int(knorm/kres)] += self.int_k[i,j,k] * _pref 
                    else:
                        _spectrum[int(knorm/kres)] += 2*self.int_k[i,j,k] * _pref 


        # collate results from all threads into one 
        spectrum = np.zeros_like(_spectrum)
        comm.Allreduce([_spectrum, MPI.DOUBLE], [spectrum, MPI.DOUBLE], op=MPI.SUM)

        # normalise
        spectrum *= np.product(globalshape)

        # define diffraction spectrum
        krange = .5*kres + kres*np.arange(len(spectrum)) # bin mid-points
        self.spectrum = np.c_[krange, spectrum]

        return 0

   
