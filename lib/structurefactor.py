import sys, time, math, cmath
import numpy as np
from numba import jit

import scipy.stats
import scipy.fft

# try importing parallel FFTW library
try:
    from mpi4py_fft import PFFT, newDistArray, DistArray
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

def mpiprint(*arg, **kwargs):
    if me == 0:
        print(*arg, **kwargs)
        sys.stdout.flush()
    return 0


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
            mpiprint ("Rewrapping atoms back into periodic box... ", end="")
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

    def SOAS_build_structurefactor_fftw(self, fftmode="FFTW", nsobol=-1, nres=10):

        # prepare dimensions of the voxel grid
        cellnorms = np.linalg.norm(self.readfile.cmat, axis=1)
        global_shape = 1 + (cellnorms/self.dx).astype(int)
        global_shape[global_shape%2==1] += 1 # set even voxel dimensions (easier fft freq handling)

        self.global_shape = global_shape 
        nvox = np.product(global_shape)
        self.dxres = cellnorms/global_shape

        mpiprint ("Spacing along cell vectors in Angstrom:", self.dxres)
        mpiprint ("Global shape of the voxel grid:", global_shape, "with total number of voxels:", nvox)
        mpiprint ("Total tensor size: %.3f GB, per thread: %.3f GB" % (1e-9*16*nvox, 1e-9*16*nvox/nprocs)) # 16 bytes per complex num
        mpiprint ()

        # construct reciprocal vectors
        a0,a1,a2 = self.readfile.cmat        
        ivol = 1./np.dot(a0, np.cross(a1, a2))
        b0 = ivol * np.cross(a1, a2) 
        b1 = ivol * np.cross(a2, a0) 
        b2 = ivol * np.cross(a0, a1) 

        mpiprint ("\nReciprocal vectors for supercell:")
        mpiprint (b0)
        mpiprint (b1)
        mpiprint (b2)

        # determine required number of sobol points
        if nsobol == -1:
            mpiprint ("Number of Sobol points to resolve size-broadened peak with a %d^3 k-pt grid: " % nres) 
            delk = 6./(2.*np.pi*np.min(cellnorms)) # FWHM size broadening
            npts = nres*nres*nres                  # target number of kpts inside broadened peak
            vol = 1/ivol                           # volume of simulation cell (=inverse volume of a reciprocal cell) 
            nsobol = 1+int(npts/(vol*(2*delk)**3))
            kres = np.linalg.norm(b0+b1+b2)/nres
            mpiprint ("%d points with a k-resolution of %f.\n" % (nsobol, kres))
        else:
            kres = np.linalg.norm(b0+b1+b2)
 
        if nsobol > 0:
            mpiprint ("Subsampling each reciprocal unit cell with %d Sobol-generated k-points." % nsobol)
            sampler = scipy.stats.qmc.Sobol(3, seed=59210939)
            if (me == 0):
                fshifts = sampler.random(nsobol) # shifts in fractional coordinates w.r.t reciprocal unit cell
            else:
                fshifts = None
            comm.barrier()
            fshifts = comm.bcast(fshifts, root=0)
            fshifts = np.r_[[[0.,0.,0.]], fshifts]
        else:
            fshifts = np.r_[[[0.,0.,0.]]]
            mpiprint ("No subsampling enabled, using one k-point per reciprocal unit cell.\n")

        # set some k space resolution for binning
        #kres = 0.7*np.linalg.norm(b0+b1+b2)*np.power(nsobol, -1/3.)
        #kres = 0.00095047523 
 
        self.kmax = np.linalg.norm(global_shape[0]*b0 + global_shape[1]*b1 + global_shape[2]*b2)
        mpiprint ("Maximum k norm:", self.kmax)
        mpiprint ()

        # initialise distributed tensor on all cores in slab distribution: 
        # the 1st axis of the tensor is distributed among the cores 
        if fftmode == "FFTW":
            mpiprint ("Constructing FFTW plan... ", end="")
            fft = PFFT(MPI.COMM_WORLD, self.global_shape, axes=(0, 1, 2), dtype=complex, grid=(-1,))
            self.uvox = newDistArray(fft, False)
            mpiprint ("done.")

        elif fftmode == "SCIPY":
            self.uvox = DistArray(global_shape, [0, 1, 1], dtype=complex)
            self.uvox[:] *= 0.0


        # gather the slab thicknesses into one list
        myshape = self.uvox.shape
        indexlist = comm.allgather(myshape[0])
        mpiprint ("Voxel grid is distributed into slabs along axis 0 of length:", indexlist)
        mpiprint ()       
 
        # list of voxel slab indices to be constructed by each thread 
        slabindex = np.r_[0, np.cumsum(indexlist)]

        # create voxel smoothing kernel
        mpiprint ("Building voxel kernel... ", end="")
        #kernel, iacell = self.SOAS_kernel()
        kernel, iacell = self.SOAS_kernel(tri=True)
        mpiprint ("done.")

        # populate voxel tensor with atomic densities
        mpiprint ("Compiling voxelisation routine... ", end="")
        #jSOAS_voxel(kernel, self.readfile.xyz[:10], iacell, self.uvox, global_shape, slabindex[me])
        jSOAS_voxel_tri(kernel, self.readfile.xyz[:10], iacell, self.uvox, global_shape, slabindex[me])
        mpiprint ("done.")

        # histogram for binning k-points over all k-directions 
        _spectrum = np.zeros(int(self.kmax/kres)+1)
        _counts = np.zeros(int(self.kmax/kres)+1, dtype=int)

        mpiprint ("Compiling binning routine... ", end="")
        _slice = self.uvox[:min(3, myshape[0]), :min(3, myshape[1]), :min(3, myshape[2])]
        kshift = np.r_[0.,0.,0.]
        
        jSOAS_spectrum_inplace(_spectrum, _counts, kres, b0, b1, b2, kshift, _slice, myshape, global_shape, 0, 0, 0) 
        _spectrum *= 0
        _counts *= 0
        mpiprint ("done.")

        if nsobol > 0:
            # get voxel mesh basis set
            nx,ny,nz = global_shape 
            asuper = self.readfile.cmat 
            acell = np.copy(self.readfile.cmat) 
            acell[0] = asuper[0]/nx
            acell[1] = asuper[1]/ny
            acell[2] = asuper[2]/nz

            mpiprint ("Compiling complex modulation routine... ", end="")
            jSOAS_modulate_voxels(self.uvox, kshift, self.uvox.shape, global_shape, acell[0], acell[1], acell[2], slabindex[me])
            mpiprint ("done.")


        for _isobol,_fshift in enumerate(fshifts):

            if _isobol > 0:
                # realign along axis=2
                self.uvox = self.uvox.redistribute(2)

            if nsobol > 0:
                mpiprint ("\n==============================================")
                mpiprint ("Sampling k-point number %d out of %d." % (1+_isobol, len(fshifts))) 
                mpiprint ("==============================================\n")

            clock = time.time()
            mpiprint ("Voxelising... ", end="")
            jSOAS_voxel_tri(kernel, self.readfile.xyz, iacell, self.uvox, global_shape, slabindex[me])
            mpiprint ("done.")
            clock = time.time() - clock

            mpiprint ("Populated %d voxels in %.3f seconds." % (nvox, clock), end=" ")
            mpiprint ("Performance: %.3f ns/vox.\n" % (1e9*clock/nvox))

            if nsobol > 0:
                # 1st point is the 0,0,0 origin of reciprocal unit cell
                if _isobol > 0:
                    # get k-point coordinates in reciprocal space from fractional coordinates
                    kshift = b0*_fshift[0] + b1*_fshift[1] + b2*_fshift[2]
                    #jSOAS_modulate_voxels(self.uvox, kshift, self.uvox.shape, global_shape, acell[0], acell[1], acell[2], slabindex[me])
                    jSOAS_modulate_voxels(self.uvox, _fshift, self.uvox.shape, global_shape, acell[0], acell[1], acell[2], slabindex[me])
                else:
                    kshift = np.r_[0.,0.,0.]
            else:
                kshift = np.r_[0.,0.,0.]

            # obtain structure factor from fourier transform
            clock = time.time()
            if fftmode == "FFTW":
                mpiprint ("Applying FFT... ", end="")
                self.uvox = fft.forward(self.uvox)
            elif fftmode == "SCIPY":
                mpiprint ("Doing scipy complex FFT along axis 2... ", end="")
                self.uvox[:] = scipy.fft.fft(self.uvox, axis=2, norm="backward")

                #mpiprint ("Doing scipy complex FFT along axis 1...")
                mpiprint ("along axis 1... ", end="")
                self.uvox[:] = scipy.fft.fft(self.uvox, axis=1, norm="backward")
                
                # align along axis=0
                self.uvox = self.uvox.redistribute(0)

                #mpiprint ("Doing scipy complex FFT along axis 0...")
                mpiprint ("along axis 0... ", end="")
                self.uvox[:] = scipy.fft.fft(self.uvox, axis=0, norm="backward")
            
            mpiprint ("done.")

            clock = time.time() - clock
            mpiprint ("Transformed %d voxels in %.3f seconds." % (nvox, clock), end=" ")
            mpiprint ("Performance: %.3f ns/vox.\n" % (1e9*clock/nvox))

            # define diffraction intensity
            newshape = self.uvox.shape

            # note: after the transformation, the tensor is distributed along a different axis
            if fftmode == "FFTW":
                newindexlist = comm.allgather(newshape[1]) # fftw version
            elif fftmode == "SCIPY":
                newindexlist = comm.allgather(newshape[2]) # scipy version

            newslabindex = np.r_[0, np.cumsum(newindexlist)]

            if fftmode == "FFTW":
                slabi,slabj,slabk = 0,newslabindex[me],0 
            elif fftmode == "SCIPY":
                slabi,slabj,slabk = 0,0,newslabindex[me]

            mpiprint ("Binning... ", end="")
            clock = time.time()
            jSOAS_spectrum_inplace(_spectrum, _counts, kres, b0, b1, b2, kshift, self.uvox, newshape, global_shape, slabi, slabj, slabk)
            mpiprint ("done.")
            clock = time.time() - clock

            mpiprint ("Binned all k-points in %.3f seconds." % clock, end= " ")
            mpiprint ("Performance: %.3f ns/vox.\n" % (1e9*clock/nvox))

        if nsobol > 1:
            mpiprint ("\nFinished subsampling.\n")
     
        # collate results from all threads into one 
        spectrum = np.zeros_like(_spectrum)
        counts = np.zeros_like(_counts)
        comm.Allreduce([_spectrum, MPI.DOUBLE], [spectrum, MPI.DOUBLE], op=MPI.SUM)
        comm.Allreduce([_counts, MPI.INT], [counts, MPI.INT], op=MPI.SUM)

        # normalise
        #spectrum *= np.product(global_shape)#/len(fshifts)

        # define diffraction spectrum
        krange = .5*kres + kres*np.arange(len(spectrum))    # bin mid-points

        # remove window in k-space due to gaussian smoothing
        acell = np.copy(self.readfile.cmat) 
        acell[0] = acell[0]/nx
        acell[1] = acell[1]/ny
        acell[2] = acell[2]/nz
        sigma = .5*np.min(np.linalg.norm(acell, axis=0))
        spectrum *= np.exp(krange*krange*4*np.pi*np.pi*sigma*sigma)

        # normalisation: divide by number of kpts per shell
        spectrum[counts>0] *= 1/counts[counts>0]            

        self.spectrum = np.c_[krange, spectrum, counts]
        
        comm.barrier()
        return 0


    def SOAS_kernel(self, tri=False):
    
        nx,ny,nz = self.global_shape 

        nfine = 10
        infine = 1./nfine

        if tri:
            nfine += 1
        kernel = np.zeros((nfine,nfine,nfine, 4,4,4))

        # find the unit cell for the voxels and for the fine-mesh subdivision of the voxels
        # assume rows of acell are the cell vectors
        asuper = self.readfile.cmat 
        acell = np.copy(self.readfile.cmat) 
        acell[0] = asuper[0]/nx
        acell[1] = asuper[1]/ny
        acell[2] = asuper[2]/nz

        # inverse voxel cell = reciprocal voxel cell (convention for spectrum: no 2pi) 
        iacell = np.linalg.inv(acell)

        afine = acell*infine
        iafine = iacell*nfine

        sigma = .5*np.min(np.linalg.norm(acell, axis=0))

        # construct the kernel
        i2s2 = 1./(2.*sigma*sigma) 


        for ix in range(nfine):
            ixv = ix
            for iy in range(nfine):
                iyv = iy
                for iz in range(nfine):
                    izv = iz

                    # if no trilinear interpolation, assume atom is in centre of fine voxel mesh 
                    if not tri:
                        ixv += 0.5
                        iyv += 0.5
                        izv += 0.5

                    # construct position of point in fine cell
                    xx = afine[0,0]*ixv + afine[1,0]*iyv + afine[2,0]*izv
                    yy = afine[0,1]*ixv + afine[1,1]*iyv + afine[2,1]*izv
                    zz = afine[0,2]*ixv + afine[1,2]*iyv + afine[2,2]*izv

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
def jSOAS_modulate_voxels(uvox, kshift, local_shape, global_shape, acell0, acell1, acell2, slabindex):

    #'''
    # faster algoirhtm exploiting orthonormality of lattice to reciprocal vectors
    nx,ny,nz = global_shape
    for i in range(local_shape[0]):
        iv = i + slabindex
        for j in range(local_shape[1]): 
            for k in range(local_shape[2]):
                uvox[i,j,k] *= cmath.exp(2j*np.pi*(iv/nx*kshift[0] + j/ny*kshift[1] + k/nz*kshift[2]))
    #'''

    '''
    for i in range(local_shape[0]):
        v0 = (i+slabindex)*acell0
        for j in range(local_shape[1]): 
            v1 = j*acell1
            for k in range(local_shape[2]):
                v2 = k*acell2
                rvox = v0+v1+v2
                uvox[i,j,k] *= cmath.exp(2j*np.pi*(rvox[0]*kshift[0] + rvox[1]*kshift[1] + rvox[2]*kshift[2]))
    '''

    return 0


@jit(nopython=True, fastmath=True)
def jSOAS_spectrum_inplace(spectrum, counts, kres, b0, b1, b2, kshift, psi_k, new_shape, global_shape, slabi, slabj, slabk):

    # bin diffraction intensity over all k-directions
    for i in range(new_shape[0]): 
        for j in range(new_shape[1]): 
            for k in range(new_shape[2]):

                # the tensor is distributed over axis 1, hence offset j index appropriately 
                iv, jv, kv = i + slabi, j + slabj, k + slabk

                # in the other axes, frequencies are ordered as {0, 1, 2, ..., Nx/2, -Nx/2 + 1, ..., -2, -1}.
                if iv >= global_shape[0]/2:
                    iv -= global_shape[0]
                if jv >= global_shape[1]/2:
                    jv -= global_shape[1]
                if kv >= global_shape[2]/2:
                    kv -= global_shape[2]

                kx = iv*b0[0]+jv*b1[0]+kv*b2[0] - kshift[0]
                ky = iv*b0[1]+jv*b1[1]+kv*b2[1] - kshift[1]
                kz = iv*b0[2]+jv*b1[2]+kv*b2[2] - kshift[2]

                # norm of the k vector belonging to this voxel
                knorm = math.sqrt(kx*kx + ky*ky + kz*kz)

                # if the input signal is real, we need to double-count all except for the kz=0 value.
                _psi_re = psi_k[i,j,k].real 
                _psi_im = psi_k[i,j,k].imag
                _int_k = _psi_re*_psi_re + _psi_im*_psi_im

                spectrum[int(knorm/kres)] += _int_k 
                counts[int(knorm/kres)] +=  1
    return 0

@jit(nopython=True, fastmath=True)
def jSOAS_voxel_tri(kernel, xyz, iacell, rho, global_shape, slabindex):

    nx_global,ny_global,nz_global = global_shape
    nx,ny,nz = rho.shape # shape of distributed tensor 
 
    natoms = len(xyz)
    nfine = 10

    kernel_linint = np.zeros((4,4,4), dtype=float) 

    # loop through each atom    
    rho *= 0        
    for ii in range(natoms):
        # find the position of the atom in global voxel space
        # should be 0:nx_global-1 etc., but the atom may be outside the periodic supercell bounds. Will fix below
        xx = iacell[0,0]*xyz[ii,0] + iacell[1,0]*xyz[ii,1] + iacell[2,0]*xyz[ii,2]
        yy = iacell[0,1]*xyz[ii,1] + iacell[1,1]*xyz[ii,1] + iacell[2,1]*xyz[ii,2]
        zz = iacell[0,2]*xyz[ii,2] + iacell[1,2]*xyz[ii,1] + iacell[2,2]*xyz[ii,2]

        # find which global voxel the atom sits in. Note: haven't yet made sure this cell is within range...
        jx = int(xx) # int acts as floor, eg xx = 12.3456 -> jx = 12
        jy = int(yy)
        jz = int(zz)

        # ignore atoms that lie beyond a few slices outside the distributed domain
        #lx = jx%nx_global - slabindex # now this is within 1st periodic replica
        #if lx > nx+3 or lx < -3:
        #    continue

        # now find which fine kernel voxel the atom sits in.
        ix = int((xx - jx)*nfine)  # xx = 12.3456 -> ix=3   (for NFINE = 10)
        iy = int((yy - jy)*nfine) 
        iz = int((zz - jz)*nfine) 

        # find the weighting on each and construct a tri-linear interpolation of the kernel
        xx = (xx - jx)*nfine - ix  # xx = 12.3456 -> xx = 0.456
        yy = (yy - jy)*nfine - iy
        zz = (zz - jz)*nfine - iz

        kernel_linint[:,:,:] = kernel[ix  ,iy  ,iz  ]*(1-xx)*(1-yy)*(1-zz) +\
                               kernel[ix+1,iy  ,iz  ]*(  xx)*(1-yy)*(1-zz) +\
                               kernel[ix  ,iy+1,iz  ]*(1-xx)*(  yy)*(1-zz) +\
                               kernel[ix+1,iy+1,iz  ]*(  xx)*(  yy)*(1-zz) +\
                               kernel[ix  ,iy  ,iz+1]*(1-xx)*(1-yy)*(  zz) +\
                               kernel[ix+1,iy  ,iz+1]*(  xx)*(1-yy)*(  zz) +\
                               kernel[ix  ,iy+1,iz+1]*(1-xx)*(  yy)*(  zz) +\
                               kernel[ix+1,iy+1,iz+1]*(  xx)*(  yy)*(  zz)

        # now can add kernel
        for kx in range(-1,3):
            lx = (jx + kx)%nx_global - slabindex # now this is within 1st periodic replica
            
            if lx < 0 or lx >= nx: # only consider voxels inside the local slice 
                continue

            for ky in range(-1,3):
                ly = (jy + ky)%ny_global
                for kz in range(-1,3):
                    lz = (jz + kz)%nz_global

                    rho[lx,ly,lz] += kernel_linint[1+kx,1+ky,1+kz]

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
        #lx = jx%nx_global - slabindex # now this is within 1st periodic replica
        #if lx > nx+3 or lx < -3:
        #    continue

        # now find which fine kernel voxel the atom sits in.
        ix = int((xx - jx)*nfine)  # xx = 12.3456 -> ix=3   (for NFINE = 10)
        iy = int((yy - jy)*nfine) 
        iz = int((zz - jz)*nfine) 

        # now can add kernel
        for kx in range(-1,3):
            lx = (jx + kx)%nx_global - slabindex # now this is within 1st periodic replica
            
            if lx < 0 or lx >= nx: # only consider voxels inside the local slice 
                continue

            for ky in range(-1,3):
                ly = (jy + ky)%ny_global
                for kz in range(-1,3):
                    lz = (jz + kz)%nz_global

                    rho[lx,ly,lz] += kernel[ix,iy,iz, 1+kx,1+ky,1+kz]

    return 0
