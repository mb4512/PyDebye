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
    def __init__(self, fpath, filetype="data"):
        self.fpath = fpath
        self.filetype = filetype
        
    def load(self, shuffle=False, exx=0.0, eiso=0.0, nbcc=0):

        # histogram files are handled specially
        if self.filetype == "hist":
            # file is read in by one thread
            if (me == 0):
                self.read_hist(self.fpath)
            else:
                self.rcut      = None
                self.natoms    = None
                self.nrho      = None
                self.histogram = None
            comm.barrier()

            # import data is copied to all other threads
            self.rcut      = comm.bcast(self.rcut, root=0)
            self.natoms    = comm.bcast(self.natoms, root=0)
            self.nrho      = comm.bcast(self.nrho, root=0)
            self.histogram = comm.bcast(self.histogram, root=0)

            mpiprint ("Imported %d bins from %s file: %s" % (len(self.histogram), self.filetype, self.fpath))
            mpiprint ()

            # exit early since no data needs to be distributed to other threads 
            return 0


        # file is read in by one thread
        if (me == 0) and nbcc == 0: 
            if self.filetype == "data":
                self.xyz, self.cell, self.cmat = self.read_data(self.fpath)
            elif self.filetype == "dump":
                self.xyz, self.cell, self.cmat = self.read_dump(self.fpath)
            else:
                mpiprint ("Error: unknown file type, only data or dump are accepted.")
                
            self.xyz    = self.xyz - self.cell[:,0]
            self.box    = self.cell[:,1] - self.cell[:,0]
            self.natoms = len(self.xyz)           
 
            if shuffle:
                print ("Shuffling atoms.")
                np.random.shuffle(self.xyz)
        else:
            self.xyz    = None
            self.box    = None
            self.natoms = None
            self.ortho  = None
            self.cmat = None
        comm.barrier()
        
        # imported data is copied to all other threads
        self.xyz    = comm.bcast(self.xyz, root=0)
        self.box    = comm.bcast(self.box, root=0)
        self.cmat   = comm.bcast(self.cmat, root=0)
        self.natoms = comm.bcast(self.natoms, root=0)
        self.ortho  = comm.bcast(self.ortho, root=0)

        # for debugging: possibility to add uniaxial or isotropic strains
        if nbcc > 0:
            mpiprint ("Constructing a bcc crystal of %d^3 unit cells, consisting of %d atoms in total." % (nbcc, 2*nbcc**3))
            a0 = 3.1652
            x = a0*np.arange(nbcc)
            self.xyz = np.vstack(np.meshgrid(x,x,x)).reshape(3,-1).T
            self.xyz = np.r_[self.xyz, self.xyz + a0*.5*np.r_[1,1,1]]
            self.cmat = nbcc*a0*np.identity(3)
            self.natoms = len(self.xyz)

        # for debugging: possibility to add uniaxial or isotropic strains
        if np.abs(exx) > 0.0:
            self.cmat[0,0] += 10.0 # zero-pad to keep voxel dimensions fixed
            self.cmat[1,1] += 10.0
            self.cmat[2,2] += 10.0
            self.xyz += 5.0        # move atoms to centre

            mpiprint ("applying a strain of %d along x-direction." % exx)
            self.xyz[:,0] *= 1.+exx

        if np.abs(eiso) > 0.0:
            self.cmat[0,0] += 10.0 # zero-pad to keep voxel dimensions fixed
            self.cmat[1,1] += 10.0
            self.cmat[2,2] += 10.0
            self.xyz += 5.0        # move atoms to centre

            mpiprint ("applying an isotropic strain of %d." % eiso)
            self.xyz *= 1.+eiso
            self.cmat *= 1+eiso

        mpiprint ("Imported %d atoms from %s file: %s" % (self.natoms, self.filetype, self.fpath))
        mpiprint ()

    def read_dump(self, fpath):
        with open(fpath, 'r') as _dfile:
            _dfile.readline()
            _dfile.readline()
            _dfile.readline()
            natoms = int(_dfile.readline())

            self.ortho = True 
            # read in box dimensions 
            if 'xy xz yz' in _dfile.readline():
                self.ortho = False

                # Triclinic case
                xlb,xhb,xy = np.array(_dfile.readline().split(), dtype=float)
                ylb,yhb,xz = np.array(_dfile.readline().split(), dtype=float)
                zlo,zhi,yz = np.array(_dfile.readline().split(), dtype=float)

                # Relevant documentation: https://lammps.sandia.gov/doc/Howto_triclinic.html
                xlo = xlb - min(0.0,xy,xz,xy+xz)
                xhi = xhb - max(0.0,xy,xz,xy+xz)
                ylo = ylb - min(0.0,yz)
                yhi = yhb - max(0.0,yz)

                # Basis matrix for converting scaled -> cart coords
                a = np.array([xhi - xlo, 0., 0.])
                b = np.array([xy,yhi - ylo, 0.])
                c = np.array([xz,yz,zhi-zlo])
                _cmat = np.array([a,b,c], dtype=float)
                _cell = np.array([[xlb,xhb], [ylb,yhb], [zlo,zhi]], dtype=float)
                self.ortho = False
            else:
                # Orthogonal case 
                xlo,xhi = np.array(_dfile.readline().split(), dtype=float)
                ylo,yhi = np.array(_dfile.readline().split(), dtype=float)
                zlo,zhi = np.array(_dfile.readline().split(), dtype=float)

                # cell vector matrix
                a = np.array([xhi - xlo, 0., 0.])
                b = np.array([0.,yhi - ylo, 0.])
                c = np.array([0.,0.,zhi-zlo])
                _cmat = np.array([a,b,c], dtype=float)
                _cell = np.array([[xlo,xhi], [ylo,yhi], [zlo,zhi]], dtype=float)

            # determine in which columns the atomic coordinates are stored 
            colstring = _dfile.readline()
            colstring = colstring.split()[2:] # first two entries typically are 'ITEM:' and 'ATOMS'
            
            reduced = False
            try:
                ix = colstring.index('x') 
                iy = colstring.index('y') 
                iz = colstring.index('z') 
            except:
                try:
                    ix = colstring.index('xs')
                    iy = colstring.index('ys')
                    iz = colstring.index('zs')
                    reduced = True
                except:
                    raise RuntimeError("Could not find coordinates 'x', 'y', 'z' or reduced coordinates 'xs', 'ys', 'zs' in the dump file!") 
            
            # read in atomic coordinates
            ixyz = np.array([ix,iy,iz])
            if reduced:
                _xyz = [np.array([float(f) for f in _dfile.readline().rstrip("\n").split(" ")])[ixyz]@_cmat for i in range(natoms)]
                _xyz = np.array(_xyz, dtype=float)
            else:
                _xyz = [np.array(_dfile.readline().rstrip("\n").split(" "))[ixyz] for i in range(natoms)]
                _xyz = np.array(_xyz, dtype=float)

            '''
            if 'xs ys zs' in _dfile.readline():
                _xyz = [np.array([float(f) for f in _dfile.readline().rstrip("\n").split(" ")[2:5]])@L for i in range(natoms)]
                _xyz = np.array(_xyz, dtype=float)
            else:
                _xyz = [_dfile.readline().rstrip("\n").split(" ")[2:5] for i in range(natoms)]
                _xyz = np.array(_xyz, dtype=float)
            '''

        return _xyz, _cell, _cmat

    
    def read_data(self, fpath):
        # currently only supports orthogonal boxes
        with open(fpath, 'r') as _dfile:
            _dfile.readline()
            _dfile.readline()
            natoms = int(_dfile.readline().split()[0])

            _dfile.readline()
            _dfile.readline()

            xlo,xhi = np.array(_dfile.readline().split()[:2], dtype=float)
            ylo,yhi = np.array(_dfile.readline().split()[:2], dtype=float)
            zlo,zhi = np.array(_dfile.readline().split()[:2], dtype=float)

            # cell vector matrix
            a = np.array([xhi - xlo, 0., 0.])
            b = np.array([0.,yhi - ylo, 0.])
            c = np.array([0.,0.,zhi-zlo])
            L = np.array([a,b,c], dtype=float)

            _cmat = L 
            _cell = np.array([[xlo,xhi], [ylo,yhi], [zlo,zhi]], dtype=float)

            _dfile.readline()
            _dfile.readline()
            _dfile.readline()
            _dfile.readline()
            _dfile.readline()
            _dfile.readline()
            _dfile.readline()
            _atomdat = [_dfile.readline().rstrip('\n').split(' ') for i in range(natoms)]
            _atomdat = np.array(_atomdat, dtype=float)
            _xyz = _atomdat[:,2:5]

            self.ortho = True
            
        return _xyz, _cell, _cmat

    def read_hist(self, fpath):
        with open(fpath, 'r') as fopen:
            self.rcut   = float(fopen.readline().split()[-1])
            self.natoms =   int(fopen.readline().split()[-1])
            self.nrho   = float(fopen.readline().split()[-1])

        self.histogram = np.loadtxt(fpath, dtype=(float, int))




