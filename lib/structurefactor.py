import sys, time, math
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

        mpiprint ("Constructing voxel field with dx=%.3f voxel spacing and computing FFT.\n" % self.dx)

        comm.barrier() 


    def SOAS_build_structurefactor_fftw(self):

        # prepare dimensions of the voxel grid
        cellnorms = np.linalg.norm(self.readfile.cmat, axis=1)
        global_shape = 1 + (cellnorms/self.dx).astype(int)
        self.global_shape = global_shape 
        nvox = np.product(global_shape)

        self.dxres = cellnorms/global_shape
        mpiprint ("Spacing along cell vectors in Angstrom:", self.dxres)

        # initialise distributed tensor on all cores in slab distribution: 
        # the 1st axis of the tensor is distributed among the cores 
        fft = PFFT(MPI.COMM_WORLD, self.global_shape, axes=(0, 1, 2), dtype=float, grid=(-1,))
        self.uvox = newDistArray(fft, False)
        myshape = self.uvox.shape

        # gather the slab thicknesses into one list
        indexlist = comm.allgather(myshape[0])
        
        mpiprint ("Global shape of the voxel grid:", global_shape, "with total number of voxels:", nvox)
        mpiprint ("Voxel grid is distributed into slabs along axis 0 of size:", indexlist)
        mpiprint ()
        sys.stdout.flush()

        # list of voxel slab indices to be constructed by each thread 
        slabindex = np.r_[0, np.cumsum(indexlist)]

        # create voxel smoothing kernel
        mpiprint ("Building voxel kernel...")
        kernel, iacell = self.SOAS_kernel()
        mpiprint ("done.\n")

        # populate voxel tensor with atomic densities
        mpiprint ("Compiling voxelisation routine...")
        jSOAS_voxel(kernel, self.readfile.xyz[:10], iacell, self.uvox, global_shape, slabindex[me])
        mpiprint ("done.\n")

        clock = time.time()
        mpiprint ("Voxelising...")
        jSOAS_voxel(kernel, self.readfile.xyz, iacell, self.uvox, global_shape, slabindex[me])
        mpiprint ("done.\n")
        clock = time.time() - clock

        mpiprint ("Populated %d voxels in %.3f seconds." % (nvox, clock))
        mpiprint ("Performance: %.3f ns/vox.\n" % (1e9*clock/nvox))
        mpiprint ()


        # obtain structure factor from fourier transform
        clock = time.time()
        mpiprint ("Applying FFT...")
        self.psi_k = fft.forward(self.uvox)
        mpiprint ("done.\n")
        clock = time.time() - clock

        mpiprint ("Transformed %d voxels in %.3f seconds." % (nvox, clock))
        mpiprint ("Performance: %.3f ns/vox.\n" % (1e9*clock/nvox))
   
        # define diffraction intensity
        self.int_k = self.psi_k.real*self.psi_k.real + self.psi_k.imag*self.psi_k.imag
        newshape = self.int_k.shape

        # note: after the transformation, the tensor is distributed over axis 1
        newindexlist = comm.allgather(newshape[1])
        newslabindex = np.r_[0, np.cumsum(newindexlist)]

        # construct reciprocal vectors
        a0,a1,a2 = self.readfile.cmat        
        ivol = 1./np.dot(a0, np.cross(a1, a2))
        b0 = ivol * np.cross(a1, a2) 
        b1 = ivol * np.cross(a2, a0) 
        b2 = ivol * np.cross(a0, a1) 

        mpiprint ("\nReciprocal space vectors for voxel grid:")
        mpiprint (b0)
        mpiprint (b1)
        mpiprint (b2)

        # set some k space resolution for binning
        kres = np.linalg.norm(b0+b1+b2)
        
        self.kmax = np.linalg.norm(global_shape[0]*b0 + global_shape[1]*b1 + global_shape[2]*b2)
        mpiprint ("maximum k norm:", self.kmax)
        mpiprint ()

        # bin diffraction intensity over all k-directions 
        _spectrum = np.zeros(int(self.kmax/kres)+1)

        mpiprint ("Compiling binning routine...")
        _slice = self.int_k[:min(3, newshape[0]), :min(3, newshape[1]), :min(3, newshape[2])]
        jSOAS_spectrum(_spectrum, kres, b0, b1, b2, _slice, newshape, global_shape, slabindex[me])
        _spectrum *= 0
        mpiprint ("done.\n")

        mpiprint ("Binning...")
        clock = time.time()
        jSOAS_spectrum(_spectrum, kres, b0, b1, b2, self.int_k, newshape, global_shape, slabindex[me])
        mpiprint ("done.\n")
        clock = time.time() - clock

        mpiprint ("Binned all k-points in %.3f seconds." % clock)
        mpiprint ("Performance: %.3f ns/vox.\n" % (1e9*clock/nvox))
        mpiprint ()
 
        # collate results from all threads into one 
        spectrum = np.zeros_like(_spectrum)
        comm.Allreduce([_spectrum, MPI.DOUBLE], [spectrum, MPI.DOUBLE], op=MPI.SUM)

        # normalise
        spectrum *= np.product(global_shape)

        # define diffraction spectrum
        krange = .5*kres + kres*np.arange(len(spectrum)) # bin mid-points
        self.spectrum = np.c_[krange, spectrum]

        return 0



    def SOAS_kernel(self):
    
        nx,ny,nz = self.global_shape 

        nfine = 10
        infine = 1./nfine

        kernel = np.zeros((nfine,nfine,nfine, 4,4,4))

        # find the unit cell for the voxels and for the fine-mesh subdivision of the voxels
        # assume rows of acell are the cell vectors
        asuper = self.readfile.cmat 
        acell = np.copy(self.readfile.cmat) 
        acell[0] = asuper[0]/nx
        acell[1] = asuper[1]/ny
        acell[2] = asuper[2]/nz

        # inverse voxel cell 
        iacell = np.linalg.inv(acell)

        afine = acell*infine
        iafine = iacell*nfine

        sigma = .5*np.min(np.linalg.norm(acell, axis=0))

        # construct the kernel
        i2s2 = 1./(2.*sigma*sigma) 

        for ix in range(nfine):
            for iy in range(nfine):
                for iz in range(nfine):
                    # construct position of point in fine cell
                    xx = afine[0,0]*(ix+.5) + afine[1,0]*(iy+.5) + afine[2,0]*(iz+.5)
                    yy = afine[0,1]*(ix+.5) + afine[1,1]*(iy+.5) + afine[2,1]*(iz+.5)
                    zz = afine[0,2]*(ix+.5) + afine[1,2]*(iy+.5) + afine[2,2]*(iz+.5)

                    ddsum = 0.0

                    for jx in range(-1,3):
                        for jy in range(-1,3):
                            for jz in range(-1,3):
                                # construct separation of node from point in fine cell
                                dx = acell[0,0]*jx + acell[1,0]*jy + acell[2,0]*jz - xx
                                dy = acell[0,1]*jx + acell[1,1]*jy + acell[2,1]*jz - yy
                                dz = acell[0,2]*jx + acell[1,2]*jy + acell[2,2]*jz - zz

                                dd = dx*dx + dy*dy + dz*dz
                                dd = dd*i2s2

                                if dd > 4.5: # 3*sigma range -> (3*sigma)^2/(2*sigma^2) = 9/2
                                    dd = 0.
                                else:
                                    dd = math.exp(-dd)
                               
                                kernel[ix,iy,iz, 1+jx,1+jy,1+jz] = dd
                                ddsum += dd

                    # normalise
                    ddsum = 1./ddsum
                    kernel[ix,iy,iz, :,:,:] *= ddsum

        return kernel, iacell



@jit(nopython=True, fastmath=True)
def jSOAS_spectrum(spectrum, kres, b0, b1, b2, int_k, new_shape, global_shape, slabindex):

    # bin diffraction intensity over all k-directions
    for i in range(new_shape[0]): 
        for j in range(new_shape[1]): 
            for k in range(new_shape[2]):

                # the tensor is distributed over axis 1, hence offset j index appropriately 
                iv, jv, kv = i, j + slabindex, k

                # in the other axes, frequencies are ordered as {0, 1, 2, ..., Nx/2, -Nx/2 + 1, ..., -2, -1}.
                if iv > global_shape[0]/2:
                    iv -= global_shape[0]
                if jv > global_shape[1]/2:
                    jv -= global_shape[1]
                if kv > global_shape[2]/2:
                    kv -= global_shape[2]

                kx = iv*b0[0]+jv*b1[0]+kv*b2[0] 
                ky = iv*b0[1]+jv*b1[1]+kv*b2[1] 
                kz = iv*b0[2]+jv*b1[2]+kv*b2[2] 

                # norm of the k vector belonging to this voxel
                knorm = math.sqrt(kx*kx + ky*ky + kz*kz)

                # as the input signal is real, we need to double-count all except for the kz=0 value.
                d3kelement = 4.*np.pi*knorm*knorm*kres

                if knorm > 0.0:
                    _pref = math.sqrt(kx*kx + ky*ky)/knorm/d3kelement
                else:
                    _pref = 1.0
                if k == 0: 
                    spectrum[int(knorm/kres)] += int_k[i,j,k] * _pref 
                else:
                    spectrum[int(knorm/kres)] += 2*int_k[i,j,k] * _pref 

    return 0

 
@jit(nopython=True, fastmath=True)
def jSOAS_voxel(kernel, xyz, iacell, rho, global_shape, slabindex):

    nx_global,ny_global,nz_global = global_shape
    nx,ny,nz = rho.shape # shape of distributed tensor 
 
    natoms = len(xyz)
    nfine = 10

    # loop through each atom    
    rho *= 0        
    for ii in range(natoms):
        # find the position of the atom in global voxel space
        # should be 0:nx_global-1 etc., but the atom may be outside the periodic supercell bounds. Will fix below
        xx = iacell[0,0]*xyz[ii,0] + iacell[1,0]*xyz[ii,1] + iacell[2,0]*xyz[ii,2]
        yy = iacell[0,1]*xyz[ii,1] + iacell[1,1]*xyz[ii,1] + iacell[2,1]*xyz[ii,2]
        zz = iacell[0,2]*xyz[ii,2] + iacell[1,2]*xyz[ii,1] + iacell[2,2]*xyz[ii,2]

        # find which global voxel the atom sits in. Note: haven't yet made sure this cell is within range...
        jx = int(xx) # eg xx = 12.3456 -> jx = 12
        jy = int(yy)
        jz = int(zz)

        # ignore atoms that lie beyond a few slices outside the distributed domain
        lx = jx%nx_global - slabindex # now this is within 1st periodic replica
        if lx > nx+3 or lx < -3:
            continue

        # now find which fine kernel voxel the atom sits in.
        ix = int((xx - jx)*nfine)  # xx = 12.3456 -> ix=3   (for NFINE = 10)
        iy = int((yy - jy)*nfine) 
        iz = int((zz - jz)*nfine) 

        # now can add kernel
        for kx in range(-1,3):
            lx = (jx + kx)%nx_global - slabindex # now this is within 1st periodic replica
            if lx < 0 or lx >= nx:
                continue

            for ky in range(-1,3):
                ly = (jy + ky)%ny_global
                for kz in range(-1,3):
                    lz = (jz + kz)%nz_global

                    rho[lx,ly,lz] += kernel[ix,iy,iz, 1+kx,1+ky,1+kz]

    return 0
