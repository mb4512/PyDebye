import sys
import numpy as np

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


class ReadFile:
    def __init__(self, fpath, filetype="restart"):
        self.fpath = fpath
        self.filetype = filetype
        
    def load(self):
        # file is read in by one thread
        if (me == 0): 
            if self.filetype == "restart":
                self.xyz, self.cell = self.read_restart(self.fpath)
            elif self.filetype == "dump":
                self.xyz, self.cell = self.read_dump(self.fpath)
            else:
                mpiprint ("Error: unknown file type, only restart or dump are accepted.")
                
            self.xyz    = self.xyz - self.cell[:,0]
            self.box    = self.cell[:,1] - self.cell[:,0]
            self.natoms = len(self.xyz)
        else:
            self.xyz    = None
            self.box    = None
            self.natoms = None
        comm.barrier()
        
        # import data is copied to all other threads
        self.xyz    = comm.bcast(self.xyz, root=0)
        self.box    = comm.bcast(self.box, root=0)
        self.natoms = comm.bcast(self.natoms, root=0)

        mpiprint ("Imported %d atoms from %s file: %s" % (self.natoms, self.filetype, self.fpath))
        mpiprint ()

    def read_dump(self, fpath):
        with open(fpath, 'r') as _dfile:
            _dfile.readline()
            _dfile.readline()
            _dfile.readline()
            natoms = int(_dfile.readline())

            _dfile.readline()

            # read in box dimensions (assuming cubic dimensions)
            xlo,xhi = _dfile.readline().split()
            ylo,yhi = _dfile.readline().split()
            zlo,zhi = _dfile.readline().split()
            _cell = np.array([[xlo,xhi], [ylo,yhi], [zlo,zhi]], dtype=np.float)
            
            _dfile.readline()

            # read in atomic coordinates
            _xyz = [_dfile.readline().rstrip("\n").split(" ")[2:] for i in range(natoms)]
            _xyz = np.array(_xyz, dtype=np.float)

        return _xyz, _cell

    
    def read_restart(self, fpath):
        with open(fpath, 'r') as _dfile:
            _dfile.readline()
            _dfile.readline()
            natoms = int(_dfile.readline().split()[0])

            _dfile.readline()
            _dfile.readline()
            xlo,xhi = _dfile.readline().split()[:2]
            ylo,yhi = _dfile.readline().split()[:2]
            zlo,zhi = _dfile.readline().split()[:2]
            _cell = np.array([[xlo,xhi], [ylo,yhi], [zlo,zhi]], dtype=np.float)

            _dfile.readline()
            _dfile.readline()
            _dfile.readline()
            _dfile.readline()
            _dfile.readline()
            _dfile.readline()
            _dfile.readline()
            _atomdat = [_dfile.readline().rstrip('\n').split(' ') for i in range(natoms)]
            _atomdat = np.array(_atomdat, dtype=np.float)
            _xyz = _atomdat[:,2:5]
            
        return _xyz, _cell
