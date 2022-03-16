import sys, time
import numpy as np
from numba import jit

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


@jit(nopython=True, cache=True, fastmath=True)
def jdisvec(xyz, _dvec, _r):
    _dr = (xyz-_r).T
    _dvec[:len(xyz)] = np.sqrt(_dr[0]*_dr[0] + _dr[1]*_dr[1] + _dr[2]*_dr[2]) 
    return len(xyz)


@jit(nopython=True, cache=True, fastmath=True)
def jdisvec_pbc(xyz, _dvec, _r, box):
    _dr = (xyz-_r).T
    pbc = .5*box
    
    _dr[0] = _dr[0] - 2.*pbc[0]*np.trunc(_dr[0]/pbc[0])
    _dr[1] = _dr[1] - 2.*pbc[1]*np.trunc(_dr[1]/pbc[1])
    _dr[2] = _dr[2] - 2.*pbc[2]*np.trunc(_dr[2]/pbc[2])

    _dvec[:len(xyz)] = np.sqrt(_dr[0]*_dr[0] + _dr[1]*_dr[1] + _dr[2]*_dr[2])
    return len(xyz) 


@jit(nopython=True, cache=True, fastmath=True)
def jiterate_histogram(n, _distances, _bins, rcut, dr):
    for i in range(n):
        # filter out distances beyond cutoff radius
        if _distances[i] < rcut:
            _bins[int(_distances[i]/dr)] += 1


@jit(nopython=True, cache=True, fastmath=True)
def jpbc_wrap(xyz, box):
    '''Wraps xyz coordinates back into periodic box so KDTree with PBC on does not complain.'''
    for i in range(len(xyz)): 
        xyz[i] = xyz[i]-box*np.floor(xyz[i]/box)

@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def jaccumulate_histogram(bins_net, bins_inc):
    # this actually seems slower than the pure python version
    n1 = len(bins_net)
    n2 = len(bins_inc)

    if n1 == 0 and n2 == 0:
        return bins_net
    
    if n1 == 0:
        bins_net = bins_inc
        return bins_net
    
    if n2 > n1:
        bins_inc[:n1] += bins_net
        bins_net = bins_inc 
    else:
        bins_net[:n2] += bins_inc 

    return bins_net



class ComputeSpectrum:
    def __init__(self, readfile, rcut=np.inf, pbc=False, rpartition=None):
        # store reference to readfile instance
        self.readfile = readfile 
        
        # do not permit a cutoff larger than half the box length (minimum image convention)
        if pbc:
            if rcut > .5*np.min(self.readfile.box):
                rcut = .5*np.min(self.readfile.box)
 
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

        # partition xyz coordinates into boxes
        # (this is bothersome to parallelise)
        if rpartition and rcut < np.inf:
            self.create_partitions()
        else:
            rpartition = False

        comm.barrier()

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

        # bin width for testing/benchmarking
        dr = 0.001

        # preallocate arrays 
        _dvec = np.zeros(self.readfile.natoms, dtype=float) 
        self.binlist = np.zeros(2*int(np.linalg.norm(self.readfile.box)/dr), dtype=int)

        # test distance function and report on estimated computing time
        mpiprint ("Compiling distance function...")
        n = self.getdistances(_dvec, self.readfile.xyz[0]) 
        jiterate_histogram(n, _dvec , self.binlist, self.rcut, dr)
        comm.barrier()
        mpiprint ("done.\n")

        if (me == 0):
            clock = time.time()
            nsample = min(100, self.readfile.natoms) # evaluate 100 atoms 

            for _r in self.readfile.xyz[:nsample]:
                n = self.getdistances(_dvec, _r) 
                jiterate_histogram(n, _dvec, self.binlist, self.rcut, dr)
            clock = time.time() - clock
            looptime  = clock/nsample
            totaltime = looptime * self.readfile.natoms/nprocs # total est. runtime in seconds
        
            print ("Estimated runtime per atom: %.3fms" % (1000*looptime))
            print ("Estimated total runtime:    %.3fs"  % totaltime)
            print ()

            # clear binlist again
            self.binlist = np.zeros(2*int(np.linalg.norm(self.readfile.box)/dr), dtype=int)

        comm.barrier() 
    
    def create_partitions(self):
        mpiprint ("Building partitions in serial.")
        clock = time.time()       
 
        xyz = self.readfile.xyz
        box = self.readfile.box
        rpart = self.rpartition
      
        # define number of partitions, rounding up to have it at (1,1,1) at least 
        partitions = np.array(box/rpart+1, dtype=int)

        # partition box dimensions
        dpart = box/partitions
        rbuffer = np.max(dpart)

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
        partition_midpts = (box/partitions)*(np.r_[partition_map]+0.5)

        self.rbuffer = rbuffer
        self.xyz_partitioned = xyz_partitioned

        if self.pbc:
            self.pkdtree = cKDTree(partition_midpts, boxsize=box)
        else:
            self.pkdtree = cKDTree(partition_midpts)

        clock = time.time() - clock
        mpiprint ("Partitioned %d atoms into %d (%dx%dx%d grid) cells in %.3f seconds.\n" % (
                    self.readfile.natoms, len(self.xyz_partitioned), partitions[0], partitions[1], partitions[2], clock))

        return 0

    def accumulate_histogram(self, _bins):
        n1 = len(self.binlist)
        n2 = len(_bins)

        if n1 == 0 and n2 == 0:
            return 0
        
        if n1 == 0:
            self.binlist = _bins
            return 0
        
        if n2 > n1:
            _bins[:n1] += self.binlist
            self.binlist = _bins
        else:
            self.binlist[:n2] += _bins
 
    def build_histogram(self, dr=0.001):
        if nprocs > 1:
            mpiprint ("Building histogram in parallel using %d threads." % nprocs) 
        else:
            mpiprint ("Building histogram in serial.")

        clock = time.time()
        _clock = clock 
        comm.barrier()

        # evenly split binning computations over all available threads 
        self.binlist = [] 
        chunk = int(np.ceil(self.readfile.natoms/nprocs))

        #tdis = 0.
        #tbin = 0.

        # preallocate arrays 
        _dvec = np.zeros(self.readfile.natoms, dtype=float) 
        self.binlist = np.zeros(2*int(np.linalg.norm(self.readfile.box)/dr), dtype=int)

        # loop over all atoms in the system and compute pairwise distances
        i0 = me*chunk
        ie = min((me+1)*chunk, self.readfile.natoms)
        for _r in self.readfile.xyz[i0:ie]:
            n = self.getdistances(_dvec, _r)
            jiterate_histogram(n, _dvec, self.binlist, self.rcut, dr) 
        natom = ie-i0

        #print ("\tRank %3d finished binning %d atoms in %.3f (tdis+tbin: %.3f+%.3f) seconds." % (me, natom, time.time()-_clock, tdis, tbin))
        #sys.stdout.flush()
        comm.barrier()
        _clock = time.time() - _clock
        mpiprint ("Binned all atoms in %.3f seconds." % _clock)

        # consolidate the binlists from each thread
        _clock = time.time()
        comm.barrier()
        for nc in range(nprocs):
            _bins = comm.bcast(self.binlist, root=nc)
            if me == 0 and nc > 0:
                self.accumulate_histogram(_bins) 
        comm.barrier()
        _clock = time.time() - _clock
        mpiprint ("Consolidated binlists in %.3f seconds.\n" % _clock)


        self.binlist = comm.bcast(self.binlist, root=0)
        self.dr = dr
        self.ri = np.r_[[i*dr+.5*dr for i in range(len(self.binlist))]] # real-space bin mid-points
       
        clock = time.time() - clock
        hours = np.floor(clock/3600)
        mins  = np.floor(clock/60 - hours*60)
        secs  = np.floor(clock - mins*60 - hours*3600)
        mpiprint ("Computed %d interatomic distances in %.3fs (%dh %dm %ds).\n" % (np.sum(self.binlist), clock, hours, mins, secs))
        comm.barrier()
        return 0


    def build_debyescherrer(self, smin, smax, ns, damp=False, ccorrection=False, nrho=None):
        
        # abort if bins have not been computed
        if len(self.binlist) == 0:
            return 0
        
        srange = np.linspace(smin, smax, int(ns))
        ri = self.ri
        natoms = self.readfile.natoms
        
        # use the provided atom number density, otherwise compute a default one 
        if nrho:
            ndensity = nrho
        else:
            ndensity = natoms/np.product(self.readfile.box)
        
        # dampening term to reduce low-s oscillations when using cutoff
        if damp:
            damping = np.sin(np.pi*ri/self.rcut)/(np.pi*ri/self.rcut)
        else:
            damping = np.ones(len(self.binlist))
            
        # continuum correction for pbc
        if ccorrection:
            ncont = natoms * 4*np.pi*ri*ri*self.dr * ndensity
        else:
            ncont = np.zeros(len(self.binlist))
        
        spectrum = [np.sum((self.binlist-ncont)*damping*np.sin(2*np.pi*_s*ri)/(2*np.pi*_s*ri)) for _s in srange]
        spectrum = natoms + 2.*np.r_[spectrum]
    
        return np.c_[srange, spectrum]
    
    
    def _getdistances(self, _dvec, _r):
        n = jdisvec(self.readfile.xyz, _dvec, _r)
        return n 
    
    def _getdistances_pbc(self, _dvec, _r):
        n = jdisvec_pbc(self.readfile.xyz, _dvec, _r, self.readfile.box)
        return n 

    def _getdistances_pbc_partitioned(self, _dvec, _r):
        pindices = self.pkdtree.query_ball_point(_r, self.rcut+self.rbuffer)
        cnebs = np.concatenate([self.xyz_partitioned[_p] for _p in pindices])
        n = jdisvec_pbc(cnebs, _dvec, _r, self.readfile.box)
        return n 

    def _getdistances_partitioned(self, _dvec, _r):
        pindices = self.pkdtree.query_ball_point(_r, self.rcut+self.rbuffer)
        cnebs = np.concatenate([self.xyz_partitioned[_p] for _p in pindices])
        n = jdisvec(cnebs, _dvec, _r)
        return n 


