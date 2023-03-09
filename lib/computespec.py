import sys, time
import numpy as np
from numba import jit
from math import erf

from scipy.spatial import cKDTree

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

@jit(nopython=True, fastmath=True)
def jiterate_histogram_weighted(n, _distances, _w1i, _weights2, _bins, rcut, dr):
    for i in range(n):
        # filter out distances beyond cutoff radius
        if _distances[i] < rcut:
            _bins[int(_distances[i]/dr)] += min(_w1i, _weights2[i])



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

#@jit(nopython=True, cache=True, fastmath=True)
@jit(nopython=True, fastmath=True)
def jfuzzy_weights(_r, xyz_partitions, rcut, rfuzz):
    # get distances between partition mid_points and atom
    _dr = (xyz_partitions-_r).T
    _dist = np.sqrt(_dr[0]*_dr[0] + _dr[1]*_dr[1] + _dr[2]*_dr[2])
    return np.array([.5*(1.-erf((_d-rcut)/rfuzz)) for _d in _dist], dtype=np.float64)



class ComputeSpectrum:
    def __init__(self, readfile, rcut=np.inf, pbc=False, rpartition=None, precision="double", dr=0.001, skip=False, fuzzy=0.0, gsim=None):

        # store reference to readfile instance
        self.readfile = readfile 
 
        # skip initialisation for histogram-import mode
        self.skip = skip
        self.nrho = None
        if skip:
            self.ri = readfile.histogram[:,0]
            self.binlist = readfile.histogram[:,1]
            self.dr = self.ri[1]-self.ri[0]
            self.rcut = readfile.rcut
            self.nrho = readfile.nrho
            self.nbin = len(self.binlist)
            return None 

      
        # do not permit a cutoff larger than half the box length (minimum image convention)
        #if pbc:
        #    if rcut > .5*np.min(self.readfile.box):
        #        rcut = .5*np.min(self.readfile.box)
 
        # if pbc, wrap atoms back into box
        if pbc:
            mpiprint ("Rewrapping atoms back into periodic box...")
            # compile first with small test set, then rewrap all atoms
            ncomp = min([10,self.readfile.natoms])
            jpbc_wrap(self.readfile.xyz[:ncomp], self.readfile.box) 
            jpbc_wrap(self.readfile.xyz, self.readfile.box) 

        comm.barrier()
        mpiprint ("done.\n")
        
        self.rpartition = rpartition
        self.pbc = pbc
        self.rcut = rcut
        self.binlist = []
        self.fuzzy = fuzzy

        if gsim is not None:
            self.gsim = True
            self.R0 = gsim[0] # mean grain radius
            self.dR = gsim[1] # grain radius standard deviation

            mpiprint ("Computing weight for each atom for grain simulation...")
            cmat = self.readfile.cmat
            self.gcentre = .5*(cmat[0] + cmat[1] + cmat[2]) # put grain centre at sim box centre
            self.wt = jfuzzy_weights(self.gcentre, self.readfile.xyz, self.R0, self.dR)
            mpiprint ("done.\n")
        else:
            self.gsim = False


        # change precision if single precision mode is requested 
        if precision == "single":
            self.single = True
            mpiprint ("Computing pairwise distances in single precision mode.\n")
        else:
            self.single = False
            mpiprint ("Computing pairwise distances in double precision mode.\n")

        if self.single:
            self.readfile.xyz = self.readfile.xyz.astype(np.single)

        # partition xyz coordinates into boxes
        # (this is bothersome to parallelise)
        if rpartition and rcut < np.inf:
            self.create_partitions()
        else:
            rpartition = False
            self.rpartition = False

        comm.barrier()

        if pbc and not self.readfile.ortho:
            raise RuntimeError('PBC are currently only supported for orthogonal boxes.')

        # define distance function according to pbc and partition specification
        if pbc:
            if rpartition:
                self.getdistances = self._getdistances_pbc_partitioned
            else:
                self.getdistances = self._getdistances_pbc
        else:
            if rpartition:
                self.getdistances = self._getdistances_partitioned
            else:
                self.getdistances = self._getdistances


        # test distance function and report on estimated computing time
        mpiprint ("Compiling distance function...")
        self.hist_loop([self.readfile.xyz[0]], dr=dr)
        comm.barrier()
        mpiprint ("done.\n")

        if (me == 0):
            clock = time.time()
            nsample = min(100, self.readfile.natoms) # evaluate 100 atoms 
            self.hist_loop(self.readfile.xyz[:nsample], dr=dr)

            clock = time.time() - clock
            looptime  = clock/nsample
            totaltime = looptime * self.readfile.natoms/nprocs # total est. runtime in seconds
        
            print ("Estimated runtime per atom: %.3fms" % (1000*looptime))
            print ("Estimated total runtime:    %.3fs"  % totaltime)
            print ()

            # clear binlist again
            #self.binlist = np.zeros(self.nbin, dtype=int)

        comm.barrier() 
    
    def create_partitions(self):
        mpiprint ("Building partitions in serial.")
        clock = time.time()       
 
        xyz = self.readfile.xyz
        cmat = self.readfile.cmat
        rpart = self.rpartition
      
        dpart = np.r_[self.rpartition, self.rpartition, self.rpartition]
        rbuffer = self.rpartition

        xyz_partitioned = []
        partition_map = []

        # get bin representations of atomic positions 
        partition_ixarray = np.array(xyz/dpart, dtype=int)

        # construct string array for ez sorting
        partition_sarray = np.array(["%dx%dx%d" % tuple(_p) for _p in partition_ixarray])
       
        # creates an array of indices, sorted by unique element
        idx_sort = np.argsort(partition_sarray)

        # sorts records array so all unique elements are together 
        sorted_records_array = partition_sarray[idx_sort]

        # returns the unique values, the index of the first occurrence of a value, and the count for each element
        vals, idx_start, count = np.unique(sorted_records_array, return_counts=True, return_index=True)

        # splits the indices into separate arrays
        res = np.split(idx_sort, idx_start[1:])

        # partition xyz data according to bins
        xyz_partitioned = [xyz[_r] for _r in res]
        
        # reorder bin partitions according to partition_sarray sorting 
        partition_map = partition_ixarray[idx_sort][idx_start]

        # define partition mid-points for kdtree search
        partition_midpts = dpart*(np.r_[partition_map]+0.5)

        self.rbuffer = rbuffer
        self.xyz_partitioned = xyz_partitioned

        #print ("partition_midpts", partition_midpts)
        self.partsize = len(partition_midpts)
        #print (self.partsize)

        # duplicate and translate partitions using periodic cell vectors, for PBC beyond minimal image convention
        if self.pbc:

            # determine how many periodic images are needed to fill a given cutoff
            cmin = np.min(np.linalg.norm(cmat,axis=1)) # shortest norm of cell vector
            nimg = 1 + int(self.rcut/cmin)
            mpiprint ("Extending periodic replica %d times to fit a cutoff radius of %.3f." % (nimg, self.rcut))
 
            self.partsize = len(partition_midpts)
            partition_periodic = np.copy(partition_midpts) 
            self.ctrans = np.r_[[[0.,0.,0.]]]
            for ix in range(-nimg, nimg+1): 
                for iy in range(-nimg, nimg+1):
                    for iz in range(-nimg, nimg+1):
                        if ix == iy == iz == 0:
                            continue
                        _cvec = ix*cmat[0] + iy*cmat[1] + iz*cmat[2]
                        _pcopy = np.copy(partition_midpts)
                        _pcopy += _cvec 
                        partition_periodic = np.r_[partition_periodic, _pcopy]
                        self.ctrans = np.r_[self.ctrans, [_cvec]]

            # partition coordinates now include periodic images
            partition_midpts = partition_periodic
        else:
            self.ctrans = None
            nimg = None

        # construct kd-tree of periodically repeated partitions now
        self.pkdtree = cKDTree(partition_midpts)
        self.partition_midpts = partition_midpts
        self.nimg = nimg

        clock = time.time() - clock
        mpiprint ("Partitioned %d atoms into %d cuboidal partitions of %f Ang side length in %.3f seconds.\n" % (
                    self.readfile.natoms, len(self.xyz_partitioned), dpart[0], clock))

        comm.barrier()

        return 0

    def hist_loop(self, xyz_thread, dr=0.001):

        # define maximum number of bins (upper limit)
        if self.pbc:
            self.nbin = int(self.rcut/dr)
        else:
            self.nbin = 2*int(np.linalg.norm(self.readfile.box)/dr)

        # estimate number of neighbours by comparing atom number density of the system
        cmat = self.readfile.cmat
        natoms = self.readfile.natoms
        ndensity = natoms/(np.dot(cmat[0], np.cross(cmat[1], cmat[2])))
        if self.pbc:
            cutvol = 4./3.*np.pi*(self.rcut+1.5*self.rbuffer)**3
            _dsize = int(ndensity*cutvol)
        else:
            _dsize = natoms 

        # preallocate arrays
        if self.gsim: 
            self.binlist = np.zeros(self.nbin, dtype=np.double)
        else:
            self.binlist = np.zeros(self.nbin, dtype=int)

        if self.single:
            _dvec = np.zeros(_dsize, dtype=np.single)
        else:
            _dvec = np.zeros(_dsize, dtype=np.double)

        if self.rpartition and self.fuzzy > 0.0:
            # array storing number of atoms per partition  
            psize = np.array([len(_x) for _x in self.xyz_partitioned], dtype=int)


        # loop over all atoms in the system and compute pairwise distances
        if self.rpartition:
            # if partitions are on, need to find neighbours for a given atom _r first; hence flipped loop order

            for _ri,_r in enumerate(xyz_thread):

                # fetch indices of periodically repeated partitions, and get periodic cell vector
                # belonging to the partitions and translate atoms by them
                pindices = self.pkdtree.query_ball_point(_r, self.rcut+self.rbuffer)
                npartnebs = len(pindices)

                if self.fuzzy > 0.0:
                    # compute distances between atom _r and neighbouring partitions, find the fuzzy weight for these distances 
                    weights = jfuzzy_weights(_r, self.partition_midpts[pindices], self.rcut-3*self.fuzzy, self.fuzzy)
                    _pmod = np.array([_p%self.partsize for _p in pindices], dtype=int)
                    # instead of full partition size psize, only count atoms with local indices between 0 and round(weight*psize)
                    psize_fuzzy = np.rint(weights*psize[_pmod]).astype(int)
                    cnebs = [(self.xyz_partitioned[_pmod[_pi]][:psize_fuzzy[_pi]] + self.ctrans[int(_p/self.partsize)]) for _pi,_p in enumerate(pindices)]
                else:
                    if self.pbc:
                        cnebs = [(self.xyz_partitioned[_p%self.partsize] + self.ctrans[int(_p/self.partsize)]) for _p in pindices]
                    else:
                        cnebs = [self.xyz_partitioned[_p%self.partsize] for _p in pindices]
                cnebs = np.concatenate(cnebs)

                for bstart in range(0, len(cnebs), BSIZE):
                    bend = min(bstart+BSIZE, len(cnebs))
                    n = self.getdistances(cnebs[bstart:bend], _dvec[bstart:bend], _r)
                    jiterate_histogram(n, _dvec[bstart:bend], self.binlist, self.rcut, dr)
        else:
            # without partitions, we know that cache blocks extend to number of atoms in the system
            for bstart in range(0, self.readfile.natoms, BSIZE):
                bend = min(bstart+BSIZE, self.readfile.natoms)

                if self.gsim:
                    # fetch weights
                    _w1 = jfuzzy_weights(self.gcentre, np.r_[xyz_thread], self.R0, self.dR)
                    _w2 = self.wt[bstart:bend]
                
                    for _ri,_r in enumerate(xyz_thread):
                        # get weight of current atom
                        _w1i = _w1[_ri]
                        n = self.getdistances(self.readfile.xyz[bstart:bend], _dvec[bstart:bend], _r)
                        jiterate_histogram_weighted(n, _dvec[bstart:bend], _w1i, _w2, self.binlist, self.rcut, dr)
                else:
                    for _r in xyz_thread:
                        n = self.getdistances(self.readfile.xyz[bstart:bend], _dvec[bstart:bend], _r)
                        jiterate_histogram(n, _dvec[bstart:bend], self.binlist, self.rcut, dr)

        return 0

    def build_histogram(self, dr=0.001):
        if self.skip:
            mpiprint ("Skipping histogram building.")
            return 0

        if nprocs > 1:
            mpiprint ("Building histogram in parallel using %d threads." % nprocs) 
        else:
            mpiprint ("Building histogram in serial.")

        clock = time.time()
        _clock = clock 
        comm.barrier()

        # distribute atoms into chunks across all threads
        chunk = int(np.ceil(self.readfile.natoms/nprocs))
        i0 = me*chunk
        ie = min((me+1)*chunk, self.readfile.natoms)
        natom = ie-i0

        # compute and bin pairwise distances
        self.hist_loop(self.readfile.xyz[i0:ie], dr=dr)

        #print ("\tRank %3d finished binning %d atoms in %.3f (tdis+tbin: %.3f+%.3f) seconds." % (me, natom, time.time()-_clock, tdis, tbin))
        #sys.stdout.flush()
        comm.barrier()
        _clock = time.time() - _clock
        mpiprint ("Binned all atoms in %.3f seconds." % _clock)

        # consolidate the binlists from each thread
        _clock = time.time()
        if self.gsim:
            _bins = np.zeros(self.nbin, dtype=float)
            comm.Allreduce([self.binlist, MPI.DOUBLE], [_bins, MPI.DOUBLE], op=MPI.SUM)
        else:
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
        if self.gsim:
            mpiprint ("Computed %e interatomic distances in %.3fs (%dh %dm %ds).\n" % (self.readfile.natoms**2, clock, hours, mins, secs))
        else:
            mpiprint ("Computed %e interatomic distances in %.3fs (%dh %dm %ds).\n" % (np.sum(self.binlist), clock, hours, mins, secs))
        comm.barrier()
        return 0


    def build_debyescherrer(self, smin, smax, ns, damp=False, ccorrection=False, nrho=None):
        # abort if bins have not been computed
        if np.sum(self.binlist) == 0:
            mpiprint ("Empty bin list, no DS spectrum computed.")
            return 0
      
        srange = np.linspace(smin, smax, int(ns))
        ri = self.ri
        natoms = self.readfile.natoms
        
        # use the provided atom number density, otherwise compute a default one 
        if nrho:
            ndensity = nrho
        else:
            cmat = self.readfile.cmat
            ndensity = natoms/(np.dot(cmat[0], np.cross(cmat[1], cmat[2])))
        
        # dampening term to reduce low-s oscillations when using cutoff
        # this is basically a Lanczos window, see https://en.wikipedia.org/wiki/Window_function
        if damp and self.rcut == np.inf:
            mpiprint ("Warning: damp set to true but no cutoff used. Not applying dampening term.")

        if damp and self.rcut < np.inf:
            damping = np.sin(np.pi*ri/self.rcut)/(np.pi*ri/self.rcut)
        else:
            damping = np.ones(self.nbin)
            
        # continuum correction for pbc
        if ccorrection:
            ncont = natoms * 4*np.pi*ri*ri*self.dr * ndensity
        else:
            ncont = np.zeros(self.nbin)

        # compile function
        mpiprint ("Compiling DS spectrum function...")
        spectrum = jaccumulate_spectrum(self.binlist, ncont, damping, ri, srange[:3])
        mpiprint ("done.\n")
 
        if nprocs > 1:
            mpiprint ("Building spectrum in parallel using %d threads." % nprocs) 
        else:
            mpiprint ("Building spectrum in serial.")
        
        # evenly split spectrum computation over all available threads
        _clock = time.time()
        chunk = ns/nprocs

        # build list of local spectrum array sizes
        sendcounts = []
        for _proc_id in range(nprocs):
            i0 = int(np.round(_proc_id*chunk))
            ie = min(int(np.round((_proc_id+1)*chunk)), ns)
            sendcounts += [ie-i0]

        # compute spectrum in each thread 
        i0 = int(np.round(me*chunk))
        ie = min(int(np.round((me+1)*chunk)), ns)

        _ns = ie-i0
        _spectrum = jaccumulate_spectrum(self.binlist, ncont, damping, ri, srange[i0:ie])
        comm.barrier()
 
        # gather spectrum into main thread 
        if (me == 0): 
            spectrum = np.zeros(ns, dtype=np.float64) 
        else:
            spectrum = None
        comm.barrier()

        comm.Gatherv(sendbuf=_spectrum, recvbuf=(spectrum, sendcounts), root=0)

        if (me == 0):
            #spectrum = natoms + 2.*spectrum
            output = np.c_[srange, spectrum]
        else:
            output = None

        _clock = time.time() - _clock
        mpiprint ("Computed DS spectrum in %.3f seconds." % _clock)

        return output 
    
    
    def _getdistances(self, xyz, _dvec, _r):
        n = jdisvec(xyz, _dvec, _r)
        return n 
    
    def _getdistances_pbc(self, xyz, _dvec, _r):
        n = jdisvec(xyz, _dvec, _r)
        return n 

    def _getdistances_pbc_partitioned(self, xyz, _dvec, _r):
        n = jdisvec(xyz, _dvec, _r)
        return n 

    def _getdistances_partitioned(self, xyz, _dvec, _r):
        n = jdisvec(xyz, _dvec, _r)
        return n 
