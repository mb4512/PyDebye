import sys, time
import numpy as np
from numba import jit
from math import erf, sin, cos

from geodesicDome.geodesic_dome import GeodesicDome
from scipy.spatial import cKDTree

from scipy.spatial.transform import Rotation

# testing out parallel FFTW
from mpi4py_fft import PFFT, newDistArray

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

#@jit(nopython=True, cache=True, fastmath=True)
@jit(nopython=True, fastmath=True)
def jdisvec(xyz, _dvec, _r):
    _dr = (xyz-_r).T
    _dvec[:len(xyz)] = np.sqrt(_dr[0]*_dr[0] + _dr[1]*_dr[1] + _dr[2]*_dr[2]) 
    return len(xyz)

@jit(nopython=True, fastmath=True)
def jdisgauss(xyz, _r, sigma):
    _dr = (xyz-_r).T
    gaussdensity = np.sum(np.exp(-(_dr[0]*_dr[0] + _dr[1]*_dr[1] + _dr[2]*_dr[2])/(2.*sigma*sigma)))
    return gaussdensity


#@jit(nopython=True, cache=True, fastmath=True)
@jit(nopython=True, fastmath=True)
def jdisvec_pbc(xyz, _dvec, _r, box):
    _dr = (xyz-_r).T
    pbc = .5*box
    
    _dr[0] = _dr[0] - 2.*pbc[0]*np.trunc(_dr[0]/pbc[0])
    _dr[1] = _dr[1] - 2.*pbc[1]*np.trunc(_dr[1]/pbc[1])
    _dr[2] = _dr[2] - 2.*pbc[2]*np.trunc(_dr[2]/pbc[2])

    _dvec[:len(xyz)] = np.sqrt(_dr[0]*_dr[0] + _dr[1]*_dr[1] + _dr[2]*_dr[2])
    return len(xyz) 


#@jit(nopython=True, cache=True, fastmath=True)
@jit(nopython=True, fastmath=True)
def jiterate_histogram(n, _distances, _bins, rcut, dr):
    for i in range(n):
        # filter out distances beyond cutoff radius
        if _distances[i] < rcut:
            _bins[int(_distances[i]/dr)] += 1


#@jit(nopython=True, cache=True, fastmath=True)
@jit(nopython=True, fastmath=True)
def jpbc_wrap(xyz, box):
    '''Wraps xyz coordinates back into periodic box so KDTree with PBC on does not complain.'''
    for i in range(len(xyz)): 
        xyz[i] = xyz[i]-box*np.floor(xyz[i]/box)


#@jit(nopython=True, cache=True, fastmath=True, nogil=True)
@jit(nopython=True, fastmath=True, nogil=True)
def jaccumulate_spectrum(binlist, ncont, damping, ri, srange):
    return np.array([np.sum((binlist-ncont)*damping*np.sin(2*np.pi*_s*ri)/(2*np.pi*_s*ri)) for _s in srange], dtype=np.float64)

@jit(nopython=True, fastmath=True, nogil=True)
def jsnap_k(ksnap, knorm, kvec, bmat, ibmat):
    for _i,_k in enumerate(kvec):
        ksnap[_i] = np.dot(bmat, np.rint(np.dot(ibmat, knorm*_k)))

@jit(nopython=True, fastmath=True, nogil=True)
def jintensity_accumulate(xyz, kvec, kvals, int_k):
    for _kv in kvec:
        _kdotr_unit = _kv[0]*xyz[:,0] + _kv[1]*xyz[:,1] + _kv[2]*xyz[:,2]

        for _ki,_knorm in enumerate(kvals):
            u,v = 0.0,0.0
            for _k in _kdotr_unit:
                u += np.cos(_knorm*_k)
                v += np.sin(_knorm*_k)
            int_k[_ki] += u*u + v*v

    int_k *= 1./len(kvec)


@jit(nopython=True, fastmath=True, nogil=True)
def jintensity_accumulate_voxel(xyz, dens, kvec, kvals, int_k):
    for _kv in kvec:
        _kdotr_unit = _kv[0]*xyz[:,0] + _kv[1]*xyz[:,1] + _kv[2]*xyz[:,2]

        for _ki,_knorm in enumerate(kvals):
            u,v = 0.0,0.0
            for _ri,_k in enumerate(_kdotr_unit):
                u += dens[_ri]*np.cos(_knorm*_k)
                v += dens[_ri]*np.sin(_knorm*_k)
            int_k[_ki] += u*u + v*v

    int_k *= 1./len(kvec)

@jit(nopython=True, fastmath=True, nogil=True)
def jscattered_accumulate_voxel(xyz, dens, kvec, psi_re, psi_im):
    for _ki,_kv in enumerate(kvec):
        _kdotr = _kv[0]*xyz[:,0] + _kv[1]*xyz[:,1] + _kv[2]*xyz[:,2]
        psi_re[_ki] += np.sum(dens*np.cos(_kdotr))
        psi_im[_ki] -= np.sum(dens*np.sin(_kdotr))


@jit(nopython=True, fastmath=True, nogil=True)
def jintensity_accumulate_snap(xyz, kvec, kvals, int_k, ksnap, bmat, ibmat):

    for _ki,_knorm in enumerate(kvals):
        jsnap_k(ksnap, _knorm, kvec, bmat, ibmat)
        for _kv in ksnap:
            _kdotr_unit = _kv[0]*xyz[:,0] + _kv[1]*xyz[:,1] + _kv[2]*xyz[:,2]
            u,v = 0.0,0.0
            for _k in _kdotr_unit:
                u += np.cos(_k)
                v += np.sin(_k)
            int_k[_ki] += u*u + v*v
    int_k *= 1./len(kvec)




#@jit(fastmath=True, nogil=True)
@jit(nopython=True, fastmath=True, nogil=True)
def jintensity_quadrature(xyz, kvec, kvals, int_k, gtris, ktri_areas):

    sre = np.zeros(len(kvec), dtype=np.float64)
    sim = np.zeros(len(kvec), dtype=np.float64)

    sre_tri = np.zeros(gtris.shape, dtype=np.float64)
    sim_tri = np.zeros(gtris.shape, dtype=np.float64)


    for _ki,_knorm in enumerate(kvals):
        for _vi,_kv in enumerate(kvec):
            _kdotr_unit = _kv[0]*xyz[:,0] + _kv[1]*xyz[:,1] + _kv[2]*xyz[:,2]
            _kdotr = _knorm*_kdotr_unit
            sre[_vi] = np.sum(np.cos(_kdotr))
            sim[_vi] = np.sum(np.sin(_kdotr))

        sre_tri[:,0] = sre[gtris[:,0]]
        sre_tri[:,1] = sre[gtris[:,1]]
        sre_tri[:,2] = sre[gtris[:,2]]
 
        sim_tri[:,0] = sim[gtris[:,0]]
        sim_tri[:,1] = sim[gtris[:,1]]
        sim_tri[:,2] = sim[gtris[:,2]]

        # integration rule for (int_S s(r) dS)/(int_S dS), where S are individual triangles
        int_k[_ki]  = np.sum(ktri_areas*(sre_tri[:,0]*sre_tri[:,0] + sre_tri[:,1]*sre_tri[:,1] + sre_tri[:,2]*sre_tri[:,2]))
        int_k[_ki] += np.sum(ktri_areas*(sim_tri[:,0]*sim_tri[:,0] + sim_tri[:,1]*sim_tri[:,1] + sim_tri[:,2]*sim_tri[:,2]))
        int_k[_ki] += np.sum(ktri_areas*(sre_tri[:,0]*sre_tri[:,1] + sre_tri[:,1]*sre_tri[:,2] + sre_tri[:,2]*sre_tri[:,0]))
        int_k[_ki] += np.sum(ktri_areas*(sim_tri[:,0]*sim_tri[:,1] + sim_tri[:,1]*sim_tri[:,2] + sim_tri[:,2]*sim_tri[:,0]))
        int_k[_ki] *= 1./(3*np.sum(ktri_areas))


class StructureFactor:
    def __init__(self, readfile, pbc=False, precision="double", dx=0.3):

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
        if precision == "single":
            self.single = True
            mpiprint ("Constructing voxel field and computing FFT in single precision mode.\n")
        else:
            self.single = False
            mpiprint ("Constructing voxel field and computing FFT in double precision mode.\n")

        if self.single:
            self.readfile.xyz = self.readfile.xyz.astype(np.single)

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
        rbuffer = 1.5*self.rc

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


    def build_structurefactor(self, smin, smax, ns, dr=0.001):
        if self.skip:
            mpiprint ("Skipping structure factor building.")
            return 0
 
        if nprocs > 1:
            mpiprint ("Building structure factor in parallel using %d threads." % nprocs) 
        else:
            mpiprint ("Building structure factor in serial.")

        clock = time.time()
        _clock = clock 
        comm.barrier()

        # construct geodesic with a given frequency and get k unit vectors
        gdome = GeodesicDome(freq=4)
        kvec = gdome.get_vertices()
        nkvec = len(kvec)
        mpiprint ("Constructed geodesic of %d k-unit vectors." % nkvec) 

        #'''
        # randomly reorient geodesic to avoid biasing crystallographic directions 
        if (me == 0):
            rndmat = Rotation.random(1).as_matrix()[0]
        else:
            rndmat = None
        rndmat = comm.bcast(rndmat, root=0)
        kvec = np.array([np.dot(rndmat, _kv) for _kv in kvec])
        #'''

        # fetch triangle ids and k unit vectors of triangle points
        gtris = gdome.get_triangles()
        ktris = kvec[gtris]

        # compute triangle area in k-space
        ktri_areas = np.array([.5*np.linalg.norm(np.cross(_ktri[1]-_ktri[0], _ktri[2]-_ktri[1])) for _ktri in ktris])
        karea = np.sum(ktri_areas) # slightly less than 4.*np.pi because curvature of sphere in triangle is neglected

        # construct reciprocal supercell vectors 
        cmat = self.readfile.cmat
        #fac = 2.*np.pi/np.dot(cmat[0], np.cross(cmat[1], cmat[2]))
        fac = 1./np.dot(cmat[0], np.cross(cmat[1], cmat[2]))
        b1 = fac * np.cross(cmat[1], cmat[2])
        b2 = fac * np.cross(cmat[2], cmat[0])
        b3 = fac * np.cross(cmat[0], cmat[1])
        bmat = np.r_[[b1,b2,b3]].T
        ibmat = np.linalg.inv(bmat)

        kstart = 2*np.pi*smin
        kend   = 2*np.pi*smax
        kpts   = ns
        kvals = np.linspace(kstart, kend, kpts, endpoint=False)

        # compiling
        mpiprint ("Compiling...")
        int_k = np.zeros(kpts, dtype=np.float64)
        jintensity_accumulate(self.readfile.xyz[:10], kvec, kvals, int_k)

        #ksnap = np.copy(kvec)
        #jintensity_accumulate_snap(self.readfile.xyz[:10], kvec, kvals, int_k, ksnap, bmat, ibmat)

        #jintensity_quadrature(self.readfile.xyz[:10], kvec, kvals, int_k, gtris, ktri_areas)
        mpiprint ("done.\n")

        '''
        # profiling 
        if (me == 0):
            clock = time.time()
            # evaluate 4 kvals
            int_k = np.zeros(kpts, dtype=np.float64)
            #jintensity_accumulate(self.readfile.xyz, kvec, kvals[-4:], int_k)

            #ksnap = np.copy(kvec)
            #jintensity_accumulate_snap(self.readfile.xyz, kvec, kvals[:1], int_k, ksnap, bmat, ibmat)
            jintensity_accumulate(self.readfile.xyz, kvec, kvals[:1], int_k)
            clock = time.time() - clock
            looptime  = clock/len(kvals[:1])
            totaltime = looptime * len(kvals)/nprocs # total est. runtime in seconds
        
            print ("Estimated runtime per k-pt:  %.3fms" % (1000*looptime))
            print ("Estimated total runtime:     %.3fs"  % totaltime)
            print ()
        sys.stdout.flush()
        comm.barrier()
        '''

        # evenly split spectrum computation over all available threads
        clock = time.time()
        chunk = kpts/nprocs

        # build list of local spectrum array sizes
        sendcounts = []
        for _proc_id in range(nprocs):
            i0 = int(np.round(_proc_id*chunk))
            ie = min(int(np.round((_proc_id+1)*chunk)), kpts)
            sendcounts += [ie-i0]

        # compute spectrum in each thread 
        i0 = int(np.round(me*chunk))
        ie = min(int(np.round((me+1)*chunk)), kpts)

        _kpts = ie-i0
        _int_k = np.zeros(_kpts, dtype=np.float64)
        jintensity_accumulate(self.readfile.xyz, kvec, kvals[i0:ie], _int_k)
        #ksnap = np.copy(kvec)
        #jintensity_accumulate_snap(self.readfile.xyz, kvec, kvals[i0:ie], _int_k, ksnap, bmat, ibmat)
        _int_k *= 1/len(self.readfile.xyz)
        comm.barrier()
 
        # gather spectrum into main thread 
        if (me == 0): 
            int_k = np.zeros(kpts, dtype=np.float64) 
        else:
            int_k = None
        comm.barrier()

        comm.Gatherv(sendbuf=_int_k, recvbuf=(int_k, sendcounts), root=0)

        #int_k = np.zeros(kpts, dtype=float)
        #jintensity_accumulate(self.readfile.xyz, kvec, kvals, int_k)

        self.kvals = kvals
        self.int_k = int_k

        comm.barrier()
        clock = time.time() - clock
        hours = np.floor(clock/3600)
        mins  = np.floor(clock/60 - hours*60)
        secs  = np.floor(clock - mins*60 - hours*3600)
        mpiprint ("Sampled %e k-pts to compute structure factor in %.3fs (%dh %dm %ds).\n" % (kpts*nkvec, clock, hours, mins, secs))

        '''
        ### QUADRATURE VERSION
        clock = time.time()
        _int_k = np.zeros(_kpts, dtype=np.float64)
        jintensity_quadrature(self.readfile.xyz, kvec, kvals[i0:ie], _int_k, gtris, ktri_areas)
        _int_k *= 1/len(self.readfile.xyz)
        comm.barrier()
 
        # gather spectrum into main thread 
        if (me == 0): 
            int_k = np.zeros(kpts, dtype=np.float64) 
        else:
            int_k = None
        comm.barrier()

        comm.Gatherv(sendbuf=_int_k, recvbuf=(int_k, sendcounts), root=0)

        mpiprint (kvals)
        mpiprint (int_k)

        comm.barrier()
        clock = time.time() - clock
        hours = np.floor(clock/3600)
        mins  = np.floor(clock/60 - hours*60)
        secs  = np.floor(clock - mins*60 - hours*3600)
        mpiprint ("Sampled %e k-pts to computed structure factor in %.3fs (%dh %dm %ds).\n" % (kpts*nkvec, clock, hours, mins, secs))
        '''

        '''
        return 0

        #print ("\tRank %3d finished binning %d atoms in %.3f (tdis+tbin: %.3f+%.3f) seconds." % (me, natom, time.time()-_clock, tdis, tbin))
        #sys.stdout.flush()
        comm.barrier()
        _clock = time.time() - _clock
        mpiprint ("Binned all atoms in %.3f seconds." % _clock)

        # consolidate the binlists from each thread
        _clock = time.time()
        _bins = np.zeros(self.nbin, dtype=int)
        comm.Allreduce([self.binlist, MPI.INT], [_bins, MPI.INT], op=MPI.SUM)
        self.binlist = _bins

        _clock = time.time() - _clock
        mpiprint ("Consolidated binlists in %.3f seconds.\n" % _clock)

        self.dr = dr
        self.ri = np.r_[[i*dr+.5*dr for i in range(self.nbin)]] # real-space bin mid-points
       
        clock = time.time() - clock
        hours = np.floor(clock/3600)
        mins  = np.floor(clock/60 - hours*60)
        secs  = np.floor(clock - mins*60 - hours*3600)
        mpiprint ("Computed %e interatomic distances in %.3fs (%dh %dm %ds).\n" % (np.sum(self.binlist), clock, hours, mins, secs))
        '''

        comm.barrier()
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
        print ("resulting tensor shape on thread %d:" % me, self.psi_k.shape)


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


    def build_structurefactor_voxel(self, smin, smax, ns, dr=0.001):
        if self.skip:
            mpiprint ("Skipping structure factor building.")
            return 0
 
        if nprocs > 1:
            mpiprint ("Building structure factor in parallel using %d threads." % nprocs) 
        else:
            mpiprint ("Building structure factor in serial.")

        clock = time.time()
        _clock = clock 
        comm.barrier()

        # build gaussian density representation of crystal on a voxel grid
        self.create_voxels()

        # construct geodesic with a given frequency and get k unit vectors
        gdome = GeodesicDome(freq=3)
        kvec = gdome.get_vertices()
        nkvec = len(kvec)
        mpiprint ("Constructed geodesic of %d k-unit vectors." % nkvec) 

        # fetch triangle ids and k unit vectors of triangle points
        gtris = gdome.get_triangles()
        ktris = kvec[gtris]

        # compute triangle area in k-space
        ktri_areas = np.array([.5*np.linalg.norm(np.cross(_ktri[1]-_ktri[0], _ktri[2]-_ktri[1])) for _ktri in ktris])
        karea = np.sum(ktri_areas) # slightly less than 4.*np.pi because curvature of sphere in triangle is neglected

        # construct reciprocal supercell vectors 
        cmat = self.readfile.cmat
        fac = 1./np.dot(cmat[0], np.cross(cmat[1], cmat[2]))
        b1 = fac * np.cross(cmat[1], cmat[2])
        b2 = fac * np.cross(cmat[2], cmat[0])
        b3 = fac * np.cross(cmat[0], cmat[1])
        bmat = np.r_[[b1,b2,b3]].T
        ibmat = np.linalg.inv(bmat)

        # define k-norm spacings
        kstart = 2*np.pi*smin
        kend   = 2*np.pi*smax
        kpts   = ns
        kvals = np.linspace(kstart, kend, kpts, endpoint=False)

        # list of all k vectors to iterate over
        kvectors = np.vstack([_k*kvec for _k in kvals])
        nk = len(kvectors)

        mpiprint ("A total of %d k-points are to be evaluated." % nk)

        _psi_re = np.zeros(100, dtype=np.float64)
        _psi_im = np.zeros(100, dtype=np.float64)

        # compiling
        mpiprint ("Compiling...")
        #int_k = np.zeros(kpts, dtype=np.float64)
        #jintensity_accumulate_voxel(self.voxelxyz[:10], self.voxeldensity[:10], kvec, kvals, int_k)
        jscattered_accumulate_voxel(self.voxelxyz[:100], self.voxeldensity[:100], kvectors[:10], _psi_re[:10], _psi_im[:10])
        mpiprint ("done.\n")

        # profiling 
        if (me == 0):
            clock = time.time()
            # evaluate 50 k-points
            nb = 50
            #int_k = np.zeros(kpts, dtype=np.float64)
            #jintensity_accumulate_voxel(self.voxelxyz, self.voxeldensity, kvec[:20], kvals[:2], int_k)
            for bstart in range(0, len(self.voxelxyz), BSIZE):
                bend = min(bstart+BSIZE, len(self.voxelxyz))
                jscattered_accumulate_voxel(self.voxelxyz[bstart:bend], self.voxeldensity[bstart:bend], kvectors[:nb], _psi_re[:nb], _psi_im[:nb])
            clock = time.time() - clock
            looptime  = clock/nb
            totaltime = looptime * kpts*nkvec/nprocs # total est. runtime in seconds
        
            print ("Estimated runtime per k-pt:  %.3fms" % (1000*looptime))
            print ("Estimated total runtime:     %.3fs"  % totaltime)
            print ()
        sys.stdout.flush()
        comm.barrier()

        # evenly split spectrum computation over all available threads
        clock = time.time()
        chunk = nk/nprocs

        # build list of local spectrum array sizes
        sendcounts = []
        for _proc_id in range(nprocs):
            i0 = int(np.round(_proc_id*chunk))
            ie = min(int(np.round((_proc_id+1)*chunk)), nk)
            sendcounts += [ie-i0]

        # compute spectrum in each thread 
        i0 = int(np.round(me*chunk))
        ie = min(int(np.round((me+1)*chunk)), nk)
        _kpts = ie-i0

        # allocate local arrays for the scattered wave
        _psi_re = np.zeros(_kpts, dtype=np.float64)
        _psi_im = np.zeros(_kpts, dtype=np.float64)

        #_int_k = np.zeros(_kpts, dtype=np.float64)
        #jintensity_accumulate_voxel(self.voxelxyz, self.voxeldensity, kvec, kvals[i0:ie], _int_k)
        #_int_k *= 1/len(self.voxelxyz)
 
        for bstart in range(0, len(self.voxelxyz), BSIZE):
            bend = min(bstart+BSIZE, len(self.voxelxyz))
            jscattered_accumulate_voxel(self.voxelxyz[bstart:bend], self.voxeldensity[bstart:bend], kvectors[i0:ie], _psi_re, _psi_im)

        comm.barrier()
 
        # gather spectrum into main thread 
        if (me == 0): 
            psi_re = np.zeros(nk, dtype=np.float64)
            psi_im = np.zeros(nk, dtype=np.float64)
            #int_k = np.zeros(kpts, dtype=np.float64) 
        else:
            psi_re = None
            psi_im = None
            #int_k = None
        comm.barrier()

        #comm.Gatherv(sendbuf=_int_k, recvbuf=(int_k, sendcounts), root=0)
        comm.Gatherv(sendbuf=_psi_re, recvbuf=(psi_re, sendcounts), root=0)
        comm.Gatherv(sendbuf=_psi_im, recvbuf=(psi_im, sendcounts), root=0)

        #int_k = np.zeros(kpts, dtype=float)
        #jintensity_accumulate(self.readfile.xyz, kvec, kvals, int_k)

        comm.barrier()
        clock = time.time() - clock
        hours = np.floor(clock/3600)
        mins  = np.floor(clock/60 - hours*60)
        secs  = np.floor(clock - mins*60 - hours*3600)
        mpiprint ("Sampled %d k-pts to compute structure factor in %.3fs (%dh %dm %ds).\n" % (kpts*nkvec, clock, hours, mins, secs))

        # compute intensity
        int_k = np.zeros(kpts)
        if (me == 0):
            for i in range(nk):
                int_k[int(i/nkvec)] += psi_re[i]*psi_re[i] + psi_im[i]*psi_im[i]
            int_k *= 1/nkvec

        comm.barrier()

        int_k = comm.bcast(int_k, root=0) 

        self.kvals = kvals/(2.*np.pi)
        self.int_k = int_k

        comm.barrier()
        return 0



    def build_debyescherrer(self, smin, smax, ns, damp=False, ccorrection=False, nrho=None):
        # abort if bins have not been computed
        if not hasattr(self, "kvals"):
            mpiprint ("Empty k values, no structure factor computed.")
            return 0

        if (me == 0):
            output = np.c_[self.kvals, self.int_k]
        else:
            output = None
        output = comm.bcast(output, root=0)
        comm.barrier()

        return output 
    
    
