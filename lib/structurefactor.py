import sys, time, math, cmath
import numpy as np
from numba import jit

from lib.distarray import DistributedTensor

import scipy.stats
import scipy.fft

# try importing parallel FFTW library
try:
    from mpi4py_fft import PFFT, newDistArray, DistArray
    IMPORTED = "successful"
except ImportError:
    IMPORTED = ImportError("FFTW mode was set but could not load mpi4py_fft library. Is it installed?")

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


class StructureFactor:
    def __init__(self, readfile, dx=0.3, fftmode="SCIPY"):

        # complain if mpi4py-fft mode was enabled but library could not be imported 
        if IMPORTED != "successful" and fftmode == "FFTW":
            raise IMPORTED

        # store reference to readfile instance
        self.readfile = readfile

        self.dx = dx 
        self.fftmode = fftmode

        mpiprint ("Constructing voxel field with dx=%.3f voxel spacing and computing FFT.\n" % self.dx)

    def construct_basis_sets(self, cmat, global_shape):

        # note we use row convention: c0 = cmat[0], c1 = cmat[1], c2 = cmat[3]

        # copy supercell vectors from input
        self.asuper = np.copy(cmat)
    
        # volume and inverse volume
        self.vol = np.linalg.det(self.asuper)
        self.ivol = 1./self.vol

        # reciprocal basis set (k-pts basis set)
        # note definition (shown here in column convention): [b1 b2 b3]^T = [c1 c2 c3]^-1
        self.bmat = np.linalg.inv(self.asuper).T 

        # voxel cell basis set 
        self.acell = np.copy(cmat)
        self.acell[0] *= 1./global_shape[0]
        self.acell[1] *= 1./global_shape[1]
        self.acell[2] *= 1./global_shape[2]

        # inverse voxel cell (for reduced coordinates in voxel space)
        self.iacell = np.linalg.inv(self.acell) 
   
        return 0
 

    def SOAS_build_structurefactor_fftw(self, nsobol=-1, nres=10, dexport=False, dkmin=0.2, dkmax=0.7, int_threshold=0, dsparsity=0.1):

        # catch non-sensical input
        assert nres > 0, "-nr NRESOLUTION, --nresolution: Target number of k-pts per reciprocal space direction must be larger than 0."

        fftmode = self.fftmode

        clock_complete = time.time()
        clock_startup = time.time()

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

        # construct voxel and reciprocal basis sets and store them as attributes in self
        self.construct_basis_sets(self.readfile.cmat, global_shape)

        mpiprint ("\nReciprocal vectors for supercell:")
        mpiprint (self.bmat)
        mpiprint ()

        # Estimate of maximum k norm sampled here (not accounting for FFT frequency shift) 
        self.kmax = np.linalg.norm(global_shape[0]*self.bmat[0] + global_shape[1]*self.bmat[1] + global_shape[2]*self.bmat[2])
        mpiprint ("Maximum k norm (1/Ang): {}\n".format(self.kmax))

        # determine required number of sobol points
        if nsobol == -1:
            mpiprint ("Number of Sobol points to resolve size-broadened peak with a %d^3 k-pt grid: " % nres) 
            delk = 6./(2.*np.pi*np.min(cellnorms)) # FWHM size broadening along the shortest dimension
            npts = nres*nres*nres                  # target number of kpts inside broadened peak
            nsobol = 1+int(npts/(self.vol*(2*delk)**3))

            # set k space resolution for binning
            kres = np.linalg.norm(self.bmat[0] + self.bmat[1] + self.bmat[2])/nres
            mpiprint ("%d points with a k-resolution of %f.\n" % (nsobol, kres))
        else:
            kres = np.linalg.norm(self.bmat[0] + self.bmat[1] + self.bmat[2])
 
        # generate construct Sobol subsampling points
        fshifts = np.zeros((1,3))
        if nsobol > 0:
            mpiprint ("Subsampling each reciprocal unit cell with %d Sobol-generated k-points." % nsobol)
            sampler = scipy.stats.qmc.Sobol(3, seed=59210939) # fixed seed for reproducibility
            if (me == 0):
                sobol_fshifts = sampler.random(nsobol) # shifts in reduced coordinates in reciprocal unit cell
            else:
                sobol_fshifts = None
            comm.barrier()
            sobol_fshifts = comm.bcast(sobol_fshifts, root=0)
            fshifts = np.r_[fshifts, sobol_fshifts]
        else:
            mpiprint ("No subsampling enabled, using one k-point per reciprocal unit cell.\n")

        # initialise distributed tensor on all cores in slab distribution: 
        # the first axis of the tensor is distributed over all MPI processes 
        if fftmode == "FFTW":
            mpiprint ("Constructing FFTW plan... ", end="")
            fft = PFFT(MPI.COMM_WORLD, self.global_shape, axes=(0, 1, 2), dtype=complex, grid=(-1,))
            self.uvox = newDistArray(fft, False)
            mpiprint ("done.")

        elif fftmode == "SCIPY":
            uvoxdist = DistributedTensor(global_shape, distaxis=0, value=0.0, numpytype=complex)
            self.uvox = uvoxdist.array

        # gather the slab thicknesses into one list
        indexlist = comm.allgather(self.uvox.shape[0])
        mpiprint ("Voxel grid is distributed into slabs along axis 0 of length: {}\n".format(indexlist))
 
        # list of voxel slab indices to be constructed by each thread 
        slabindex = np.r_[0, np.cumsum(indexlist)]

        # create voxel smoothing kernel
        mpiprint ("Building voxel kernel... ", end="")
        kernel = self.SOAS_kernel(self.acell, tri=True)
        mpiprint ("done.")

        # transform atom xyz coordinates to voxel space
        mpiprint ("Compiling xyz to voxel coordinates routine... ", end="")
        jSOAS_xyz_to_voxel(np.ones((5,3), dtype=float), self.iacell) # compile first
        mpiprint ("done.")
        jSOAS_xyz_to_voxel(self.readfile.xyz, self.iacell) # transform

        # spatial domain decomposition for atoms, so that voxelisation complexity scales linearly with number of atoms
        dskin = 3
        wrapindex = self.readfile.xyz[:,0] % global_shape[0]
        _bool = (wrapindex >= slabindex[me]-dskin)*(wrapindex <= slabindex[me]+dskin+self.uvox.shape[0])

        # to make sure PBCs do not cause issues, treat edge cases separately. probably there is a more elegant way
        if slabindex[me] < dskin:
            _bool[wrapindex > global_shape[0]-dskin+slabindex[me]] = True
        if slabindex[me] + self.uvox.shape[0] + dskin > global_shape[0]:
            _bool[wrapindex + global_shape[0] < slabindex[me]+self.uvox.shape[0]+dskin] = True

        # only retain atoms in this MPI rank that lie within the local slab plus a skin region 
        self.readfile.xyz = self.readfile.xyz[_bool]

        mpiprint ("Compiling voxelisation routine... ", end="")
        jSOAS_voxel_tri(kernel, np.random.rand(10,3), self.uvox, global_shape, slabindex[me]) # compile first
        mpiprint ("done.")

        # prepare histogram for binning k-points over all k-directions 
        _spectrum = np.zeros(int(self.kmax/kres)+1)
        _counts = np.zeros(int(self.kmax/kres)+1, dtype=int)

        mpiprint ("Compiling binning routine... ", end="")
        _psi_k = np.ones((10,10,10), dtype=complex)
        kshift = np.zeros(3) 
        jSOAS_spectrum_inplace(_spectrum, _counts, kres, self.bmat, kshift, _psi_k, global_shape, 0, 0, 0) 
        _spectrum[:] = 0.0
        _counts *= 0
        mpiprint ("done.")
        
        if nsobol > 0:
            mpiprint ("Compiling complex modulation routine... ", end="")
            jSOAS_modulate_voxels(self.uvox, kshift, global_shape, slabindex[me])
            mpiprint ("done.")

        clock_startup = time.time() - clock_startup
        mpiprint ("Completed start-up process in %.3f seconds." % clock_startup)

        # initialise vectors for storing diffraction pattern
        if dexport:
            _frac = 0.01 # initial size (this is dynamically increased) 
            kvectors = np.zeros((int(_frac*self.uvox.size), 3), dtype=float)
            intensities = np.zeros(int(_frac*self.uvox.size), dtype=float)
            _diffraction_pattern = [] 

        # loop over all Sobol subsampling points, compute and bin diffraction pattern
        clock_sobol = 0
        for _isobol,_fshift in enumerate(fshifts):
            clock_tot = time.time()

            if nsobol > 0:
                mpiprint ("\n==============================================")
                mpiprint ("Sampling k-point number %d out of %d." % (1+_isobol, len(fshifts))) 
                mpiprint ("==============================================\n")

            # reinitialise tensor distributed along first axis 
            if _isobol > 0:
                if fftmode == "FFTW":
                    self.uvox = newDistArray(fft, False)
                else:
                    uvoxdist = DistributedTensor(global_shape, distaxis=0, value=0.0, numpytype=complex)
                    self.uvox = uvoxdist.array

            clock = time.time()
            mpiprint ("Voxelising... ", end="")
            jSOAS_voxel_tri(kernel, self.readfile.xyz, self.uvox, global_shape, slabindex[me])
            mpiprint ("done.")
            clock = time.time() - clock

            mpiprint ("Populated %d voxels in %.3f seconds." % (nvox, clock), end=" ")
            mpiprint ("Performance: %.3f ns/vox.\n" % (1e9*clock/nvox))
            
            # default value if no Sobol subsampling: 1st point is the origin of reciprocal unit cell
            kshift = np.zeros(3) 
            if nsobol > 0 and _isobol > 0:
                # get k-point coordinates in reciprocal space from Sobol reduced coordinates
                kshift = self.bmat@_fshift
                # module voxel density with complex phase 
                jSOAS_modulate_voxels(self.uvox, _fshift, global_shape, slabindex[me])

            # obtain structure factor from fourier transform
            clock = time.time()

            if fftmode == "FFTW":
                mpiprint ("Applying FFT... ", end="")
                self.uvox = fft.forward(self.uvox)
                self.uvox *= nvox # normalise

                # after the FFT, the tensor is distributed along axis=1 in FFTW 
                newindexlist = comm.allgather(self.uvox.shape[1])
                newslabindex = np.r_[0, np.cumsum(newindexlist)]
                slabi,slabj,slabk = 0,newslabindex[me],0  # distributed slab index offsets

            elif fftmode == "SCIPY":
                mpiprint ("Doing scipy complex FFT along axis 2... ", end="")
                self.uvox[:] = scipy.fft.fft(self.uvox, axis=2, norm="backward")

                mpiprint ("FFT along axis 1... ", end="")
                self.uvox[:] = scipy.fft.fft(self.uvox, axis=1, norm="backward")
                
                mpiprint ("aligning tensor along axis 0... ", end="")
                uvoxdist.redistribute(-1)
                self.uvox = uvoxdist.array

                mpiprint ("FFT along axis 0... ", end="")
                self.uvox[:] = scipy.fft.fft(self.uvox, axis=0, norm="backward")
                
                # after the FFT, the tensor is distributed along axis=2 in SCIPY 
                newindexlist = comm.allgather(self.uvox.shape[2]) # scipy version
                newslabindex = np.r_[0, np.cumsum(newindexlist)]
                slabi,slabj,slabk = 0,0,newslabindex[me] # distributed slab index offsets

            mpiprint ("done.")

            clock = time.time() - clock
            mpiprint ("Transformed %d voxels in %.3f seconds." % (nvox, clock), end=" ")
            mpiprint ("Performance: %.3f ns/vox.\n" % (1e9*clock/nvox))

            # bin the computed diffraction pattern into a line-profile
            clock = time.time()
            mpiprint ("Binning... ", end="")
            jSOAS_spectrum_inplace(_spectrum, _counts, kres, self.bmat, kshift, self.uvox, global_shape, slabi, slabj, slabk)
            mpiprint ("done.")
            clock = time.time() - clock

            mpiprint ("Binned all k-points in %.3f seconds." % clock, end= " ")
            mpiprint ("Performance: %.3f ns/vox.\n" % (1e9*clock/nvox))
            
            clock_tot = time.time() - clock_tot
            clock_sobol += clock_tot
            mpiprint ("Completed subsampling iteration in %.3f seconds." % clock_tot)

            # if no threshold is given, estimate it for a given target sparsity from the first sobol iteration
            if dexport and int_threshold == 0 and _isobol == 0:

                # collate counts and intensities from all threads to all
                krange, spectrum, counts = self.reduce_spectrum(_spectrum, _counts, kres, self.acell)

                # from this spectrum, determine approximate intensity threshold for target sparsity 
                int_threshold = self.determine_threshold(dsparsity, dkmin, dkmax, krange, spectrum, counts)
                mpiprint ("\nDetermined approximate intensity threshold of {} for target sparsity of {}%%.\n".format(int_threshold, 100*dsparsity))
            
            # threshold intensities and store local diffraction pattern
            if dexport: 
                _npts = -1
                while (_npts == -1):
                    # populate k-points and intensities for the diffraction pattern
                    _npts = jSOAS_diffraction(kvectors, intensities, int_threshold, dkmin, dkmax, 
                                              self.bmat, kshift, self.uvox, global_shape, slabi, slabj, slabk)

                    # keep increasing the size of the storage vectors if out-of-bounds access occurrs
                    if _npts == -1:
                        _frac += 0.01
                        print ("Diffraction pattern: out-of-access on rank %d, increasing array size to %.2f%% size of voxel tensor." % (me, _frac))
                        kvectors = np.zeros((int(_frac*self.uvox.size), 3), dtype=float)
                        intensities = np.zeros(int(_frac*self.uvox.size), dtype=float)
                
                _diffraction_pattern += [np.c_[kvectors[:_npts], intensities[:_npts]]]
                #print ("Sparsity on rank %d: %.3f%%" % (me, _npts/self.uvox.size*100))

        if nsobol > 1:
            mpiprint ("\nFinished subsampling.\n")
            mpiprint ("Completed subsampling in %.3f seconds." % clock_sobol)

        # export sparse diffraction spectrum data
        if dexport:
            mpiprint ("Exporting diffraction patterns...", end=" ")
            _diffraction_pattern = np.vstack(_diffraction_pattern)
            with open('%s.%d.xyz' % (dexport, me), 'w') as ofile:
                ofile.write("%d\n" % len(_diffraction_pattern))
                ofile.write("# kx ky kz intensity\n")
                np.savetxt(ofile, _diffraction_pattern, fmt="%16.8e %16.8e %16.8e %16.8e")
            comm.barrier()
            mpiprint ("done.\n")

        # collate counts and intensities from all threads to all
        krange, spectrum, counts = self.reduce_spectrum(_spectrum, _counts, kres, self.acell)
 
        # construct spectrum for exporting
        self.spectrum = np.c_[krange, spectrum, counts]

        # fetch position of last occupied bin and cut off spectrum beyond that value
        lastindex = np.nonzero(counts)[0][-1]
        self.spectrum = self.spectrum[:lastindex]

        # report sparsity of diffraction profile 
        if dexport:
            _size = _diffraction_pattern.size
            totsize = np.sum(comm.allgather(_size))
            _bool = (krange>=dkmin)*(krange<=dkmax)
            fraction = totsize/np.sum(counts[_bool])
            mpiprint ("Sparsity of stored diffraction pattern for %.3f<=|k|<%.3f: %.3f%%" % (dkmin, dkmax, 100*fraction))

        # report intensity threshold for hypothetical diffraction pattern export 
        int_threshold = self.determine_threshold(dsparsity, dkmin, dkmax, krange, spectrum, counts)
        mpiprint ("\nThe intensity threshold for storing a 3D diffraction pattern with sparsity of {}%% is {}.\n".format(100.*dsparsity, int_threshold))

        clock_complete = time.time() - clock_complete
        mpiprint ("Completed entire routine in %.3f seconds." % clock_complete)
        
        comm.barrier()
        return 0


    def reduce_spectrum(self, _spectrum, _counts, kres, acell):

        # collate results from all threads into one 
        spectrum = np.zeros_like(_spectrum)
        counts = np.zeros_like(_counts)
        comm.Allreduce([_spectrum, MPI.DOUBLE], [spectrum, MPI.DOUBLE], op=MPI.SUM)
        comm.Allreduce([_counts, MPI.INT], [counts, MPI.INT], op=MPI.SUM)

        # remove window in k-space due to gaussian smoothing
        krange = .5*kres + kres*np.arange(len(spectrum))    # wavenumber bin mid-points
        sigma = .5*np.min(np.linalg.norm(acell, axis=0))
        spectrum *= np.exp(krange*krange*4*np.pi*np.pi*sigma*sigma)

        # normalisation: divide by number of kpts per shell
        spectrum[counts>0] *= 1/counts[counts>0]            

        return krange, spectrum, counts

    def determine_threshold(self, sparsity, dkmin, dkmax, krange, spectrum, counts):
        
        _bool = (krange>=dkmin)*(krange<=dkmax)
        fspec   = spectrum[_bool]
        fcounts = counts[_bool]

        # bin counts over intensities
        intmax = np.max(fspec)
        ires = intmax/1000
        irange = .5*ires + np.arange(0, intmax+ires, ires)      # intensity bin mid-points
        histo = np.zeros(len(irange))                    
    
        # bin intensity and count
        for _ii in range(len(fspec)):
            histo[int(fspec[_ii]/ires)] += fcounts[_ii]

        # normalise to obtain frequency distribution of counts over intensity
        histo *= 1/(ires*np.sum(histo))

        # determine cumulative function and find threshold intensity for which target sparsity is met 
        histocumulative = ires * np.r_[0, np.cumsum(histo)]
        _ix = np.argwhere(histocumulative > 1.-sparsity)[0,0]
        int_threshold = irange[_ix]

        return int_threshold


    def SOAS_kernel(self, acell, tri=False):
    
        nfine = 10
        iafine = 1./nfine

        if tri:
            nfine += 1
        kernel = np.zeros((nfine,nfine,nfine, 4,4,4))

        # find the unit cell for for fine-mesh subdivision of the voxels
        afine = acell*iafine

        # chose standard deviation of Gaussian smoothing as half the shortest vertex spacing 
        sigma = .5*np.min(np.linalg.norm(acell, axis=0))
        i2s2 = 1./(2.*sigma*sigma) 

        # construct the kernel
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

        return kernel



@jit(nopython=True, fastmath=True)
def jSOAS_modulate_voxels(uvox, kshift, global_shape, slabindex):

    # fetch shape of the slab on this MPI process
    local_shape = uvox.shape

    # faster algorithm exploiting orthonormality of lattice to reciprocal vectors
    nx,ny,nz = global_shape
    for i in range(local_shape[0]):
        iv = i + slabindex
        for j in range(local_shape[1]): 
            for k in range(local_shape[2]):
                uvox[i,j,k] *= cmath.exp(2j*np.pi*(iv/nx*kshift[0] + j/ny*kshift[1] + k/nz*kshift[2]))
    return 0



@jit(nopython=True, fastmath=True)
def jSOAS_diffraction(kvectors, intensities, int_threshold, kmin, kmax, bmat, kshift, psi_k, global_shape, slabi, slabj, slabk):

    # fetch shape of the slab on this MPI process
    local_shape = psi_k.shape
    
    # store diffraction spots in sparse format for the given MPI thread 
    c = 0
    for i in range(local_shape[0]):
        # offset i,j,k depending on how the tensor is distributed
        iv = i + slabi
        for j in range(local_shape[1]):
            jv = j + slabj
            for k in range(local_shape[2]):
                kv = k + slabk

                # compute intensity
                _psi_re = psi_k[i,j,k].real 
                _psi_im = psi_k[i,j,k].imag
                _int_k = _psi_re*_psi_re + _psi_im*_psi_im

                # only consider intensities above a given threshold for sparse representation
                if _int_k < int_threshold:
                    continue

                # in the other axes, frequencies are ordered as {0, 1, 2, ..., Nx/2, -Nx/2 + 1, ..., -2, -1}.
                if iv >= global_shape[0]/2:
                    iv -= global_shape[0]
                if jv >= global_shape[1]/2:
                    jv -= global_shape[1]
                if kv >= global_shape[2]/2:
                    kv -= global_shape[2]

                kx = iv*bmat[0,0] + jv*bmat[1,0] + kv*bmat[2,0] - kshift[0]
                ky = iv*bmat[0,1] + jv*bmat[1,1] + kv*bmat[2,1] - kshift[1]
                kz = iv*bmat[0,2] + jv*bmat[1,2] + kv*bmat[2,2] - kshift[2]

                # squared norm of the k vector belonging to this voxel
                ksq = kx*kx + ky*ky + kz*kz
    
                if c >= len(kvectors):
                    return -1

                # we do not care about the 000 spot
                if ksq < kmin*kmin:
                    continue

                if ksq > kmax*kmax:
                    continue

                kvectors[c,0] = kx 
                kvectors[c,1] = ky 
                kvectors[c,2] = kz 
                intensities[c] = _int_k
                c += 1

    # return length of vectors containing stored information 
    return c



@jit(nopython=True, fastmath=True)
def jSOAS_spectrum_inplace(spectrum, counts, kres, bmat, kshift, psi_k, global_shape, slabi, slabj, slabk):

    # fetch shape of the slab on this MPI process
    local_shape = psi_k.shape

    # bin diffraction intensity over all k-directions
    for i in range(local_shape[0]):
        # offset i,j,k depending on how the tensor is distributed
        iv = i + slabi
        for j in range(local_shape[1]):
            jv = j + slabj
            for k in range(local_shape[2]):
                kv = k + slabk

                # in the other axes, frequencies are ordered as {0, 1, 2, ..., Nx/2, -Nx/2 + 1, ..., -2, -1}.
                if iv >= global_shape[0]/2:
                    iv -= global_shape[0]
                if jv >= global_shape[1]/2:
                    jv -= global_shape[1]
                if kv >= global_shape[2]/2:
                    kv -= global_shape[2]

                kx = iv*bmat[0,0] + jv*bmat[1,0] + kv*bmat[2,0] - kshift[0]
                ky = iv*bmat[0,1] + jv*bmat[1,1] + kv*bmat[2,1] - kshift[1]
                kz = iv*bmat[0,2] + jv*bmat[1,2] + kv*bmat[2,2] - kshift[2]

                # norm of the k vector belonging to this voxel
                knorm = math.sqrt(kx*kx + ky*ky + kz*kz)

                # compute intensity
                _psi_re = psi_k[i,j,k].real 
                _psi_im = psi_k[i,j,k].imag
                _int_k = _psi_re*_psi_re + _psi_im*_psi_im

                # bin intensity and count
                spectrum[int(knorm/kres)] += _int_k 
                counts[int(knorm/kres)] +=  1
    return 0

@jit(nopython=True, fastmath=True)
def jSOAS_xyz_to_voxel(xyz, iacell):

    natoms = len(xyz)

    # loop through each atom    
    for ii in range(natoms):
        # find the position of the atom in global voxel space
        # should be 0:nx_global-1 etc., but the atom may be outside the periodic supercell bounds. Will fix below
        xx = iacell[0,0]*xyz[ii,0] + iacell[1,0]*xyz[ii,1] + iacell[2,0]*xyz[ii,2]
        yy = iacell[0,1]*xyz[ii,1] + iacell[1,1]*xyz[ii,1] + iacell[2,1]*xyz[ii,2]
        zz = iacell[0,2]*xyz[ii,2] + iacell[1,2]*xyz[ii,1] + iacell[2,2]*xyz[ii,2]
        xyz[ii,0] = xx
        xyz[ii,1] = yy
        xyz[ii,2] = zz


@jit(nopython=True, fastmath=True)
def jSOAS_voxel_tri(kernel, xxyyzz, rho, global_shape, slabindex):

    nx_global,ny_global,nz_global = global_shape # shape of global tensor 
    nx,ny,nz = rho.shape                         # shape of distributed tensor 
 
    natoms = len(xxyyzz)
    nfine = 10

    kernel_linint = np.zeros((4,4,4), dtype=float) 

    # loop through each atom    
    rho *= 0        
    for ii in range(natoms):
        xx,yy,zz = xxyyzz[ii] 

        # find which global voxel the atom sits in. Note: haven't yet made sure this cell is within range...
        jx = int(xx) # int acts as floor, eg xx = 12.3456 -> jx = 12
        jy = int(yy)
        jz = int(zz)

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
def jSOAS_voxel(kernel, xxyyzz, rho, global_shape, slabindex):

    nx_global,ny_global,nz_global = global_shape # shape of global tensor 
    nx,ny,nz = rho.shape                         # shape of distributed tensor 
 
    natoms = len(xxyyzz)
    nfine = 10

    # loop through each atom    
    rho *= 0        
    for ii in range(natoms):
        xx,yy,zz = xxyyzz[ii] 

        # find which global voxel the atom sits in. Note: haven't yet made sure this cell is within range...
        jx = int(xx) # eg xx = 12.3456 -> jx = 12
        jy = int(yy)
        jz = int(zz)

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
