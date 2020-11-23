import sys, time
import numpy as np
from scipy.spatial import cKDTree

from numba import jit

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


@jit(nopython=True, cache=True)
def jdisvec(xyz, _r):
    _dr = (xyz-_r).T
    return np.sqrt(_dr[0]*_dr[0] + _dr[1]*_dr[1] + _dr[2]*_dr[2])


@jit(nopython=True, cache=True)
def jdisvec_pbc(xyz, _r, box):
    _dr = (xyz-_r).T
    pbc = .5*box
    
    _dr[0] = _dr[0] - 2.*pbc[0]*np.trunc(_dr[0]/pbc[0])
    _dr[1] = _dr[1] - 2.*pbc[1]*np.trunc(_dr[1]/pbc[1])
    _dr[2] = _dr[2] - 2.*pbc[2]*np.trunc(_dr[2]/pbc[2])

    return np.sqrt(_dr[0]*_dr[0] + _dr[1]*_dr[1] + _dr[2]*_dr[2])


class ComputeSpectrum:
    def __init__(self, readfile, rcut=np.inf, pbc=False, rpartition=None):
        # store reference to readfile instance
        self.readfile = readfile 
        
        # do not permit a cutoff larger than half the box length (minimum image convention)
        if pbc:
            if rcut > .5*np.min(self.readfile.box):
                rcut = .5*np.min(self.readfile.box)
        
        self.rpartition = rpartition
        self.pbc = pbc
        self.rcut = rcut
        self.binlist = []
        
        # partition xyz coordinates into boxes
        # (this is bothersome to parallelise)
        if rpartition and pbc:
            self.create_partitions()
        if rpartition and not pbc and rcut < np.inf:
            self.create_partitions()
        
        # define distance function according to pbc and partition specification
        if pbc:
            if rpartition:
                self.getdistances = self._getdistances_pbc_partitioned
            else:
                self.getdistances = self._getdistances_pbc
        else:
            self.getdistances = self._getdistances
   
        # test distance function and report on estimated computing time
        mpiprint ("Compiling distance function...")
        self.getdistances(self.readfile.xyz[0]) # compile first on all threads
        if (me == 0):
            clock = time.time()
            nsample = min(100, self.readfile.natoms) # evaluate 100 samples
            [self.getdistances(_x) for _x in self.readfile.xyz[:nsample]]
            clock = time.time() - clock
            looptime  = clock/nsample
            totaltime = looptime * self.readfile.natoms/nprocs # total est. runtime in seconds
        
            print ("Estimated runtime per atom: %.3fms" % (1000*looptime))
            print ("Estimated total runtime:    %.3fs"  % totaltime)
            print ()

 

    def create_partitions(self):
        mpiprint ("Building partitions in serial.")
        clock = time.time()       
 
        xyz = self.readfile.xyz
        box = self.readfile.box
        rpart = self.rpartition
        
        partitions=(box/rpart).astype(np.int)
        partitions[partitions==0] +=1 # keep number of partitions at least at (1,1,1)

        dpart = box/partitions
        rbuffer = np.max(dpart)
        xbins = [(i*dpart[0], (i+1)*dpart[0]) for i in range(partitions[0])]
        ybins = [(i*dpart[1], (i+1)*dpart[1]) for i in range(partitions[1])]
        zbins = [(i*dpart[2], (i+1)*dpart[2]) for i in range(partitions[2])]

        xyz_partitioned = []
        partition_map = []

        for _ix,_xb in enumerate(xbins):
            _xbool = (xyz[:,0]<_xb[1])*(xyz[:,0]>_xb[0])

            for _iy,_yb in enumerate(ybins):
                _ybool = (xyz[:,1]<_yb[1])*(xyz[:,1]>_yb[0])

                for _iz,_zb in enumerate(zbins):
                    _zbool = (xyz[:,2]<_zb[1])*(xyz[:,2]>_zb[0])

                    xyz_partitioned.append(xyz[_xbool*_ybool*_zbool])
                    partition_map.append([_ix, _iy, _iz])

        partition_midpts = (box/partitions)*(np.r_[partition_map]+0.5)

        self.rbuffer = rbuffer
        self.xyz_partitioned = xyz_partitioned
        self.pkdtree = cKDTree(partition_midpts, boxsize=box)

        clock = time.time() - clock
        mpiprint ("Partitioned %d atoms into %d (%dx%dx%d) cells in %.3f seconds.\n" % (
                    self.readfile.natoms, len(self.xyz_partitioned), partitions[0], partitions[1], partitions[2], clock))

        return 0

    def iterate_histogram(self, _distances, dr):
        _ix = np.array(_distances/dr, dtype=np.int)
        _bins = np.bincount(_ix)
        return _bins
       
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

        # evenly split binning computations over all available threads 
        self.binlist = []
        chunk = int(np.ceil(self.readfile.natoms/nprocs))
        for _i,_r in enumerate(self.readfile.xyz):
            if me*chunk <= _i < (me+1)*chunk:
                _distances = self.getdistances(_r)
                _bins = self.iterate_histogram(_distances, dr)
                self.accumulate_histogram(_bins)
        comm.barrier()

        # consolidate the binlists from each thread
        for nc in range(nprocs):
            _bins = comm.bcast(self.binlist, root=nc)
            if me == 0 and nc > 0:
                self.accumulate_histogram(_bins) 
        comm.barrier()

        self.binlist = comm.bcast(self.binlist, root=0)
        self.dr = dr
        self.ri = np.r_[[i*dr+.5*dr for i in range(len(self.binlist))]]
       
        clock = time.time() - clock
        hours = np.floor(clock/3600)
        mins  = np.floor(clock/60 - hours*60)
        secs  = np.floor(clock - mins*60 - hours*3600)
        comm.barrier()

        mpiprint ("Computed %d interatomic distances in %.3fs (%dh %dm %ds).\n" % (np.sum(self.binlist), clock, hours, mins, secs))

 
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
    
    
    def _getdistances(self, _r):
        _distances = jdisvec(self.readfile.xyz, _r)
        return _distances[_distances<self.rcut]
    
    def _getdistances_pbc(self, _r):
        _distances = jdisvec_pbc(self.readfile.xyz, _r, self.readfile.box)
        return _distances[_distances<self.rcut]

    def _getdistances_pbc_partitioned(self, _r):
        pindices = self.pkdtree.query_ball_point(_r, self.rcut+self.rbuffer)
        cnebs = np.concatenate([self.xyz_partitioned[_p] for _p in pindices])
        
        _distances = jdisvec_pbc(cnebs, _r, self.readfile.box)
        return _distances[_distances<self.rcut]

