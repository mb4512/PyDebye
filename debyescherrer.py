import sys, os
from pathlib import Path
import argparse, textwrap

import numpy as np

from lib.computespec import ComputeSpectrum 
from lib.structurefactor import StructureFactor
from lib.readfile import ReadFile

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


# better text formatting for argparse
class RawFormatter(argparse.HelpFormatter):
    def _fill_text(self, text, width, indent):
        return "\n".join([textwrap.fill(line, width) for line in textwrap.indent(textwrap.dedent(text), indent).splitlines()])

def main():

    program_descripton = f'''
        PyDebye v0.3

        Tool to compute the Debye-Scherrer line profile for LAMMPS dump and restart files, written in 100% Python. 
        Supports periodic boundary conditions for orthogonal simulation cells beyond the minimum image convention.
        Supports real-space method and N*log(N) scaling fourier-space method (requires FFTW library).

        Max Boleininger, Nov 2020
        max.boleininger@ukaea.uk

        Licensed under the Creative Commons Zero v1.0 Universal
        https://creativecommons.org/publicdomain/zero/1.0/

        Distributed on an "AS IS" basis without warranties
        or conditions of any kind, either express or implied.

        USAGE:
        '''

    # parse all input arguments
    parser = argparse.ArgumentParser(description=program_descripton, formatter_class=RawFormatter)

    parser.add_argument("inputfile", help="path to input file, accepting LAMMPS dump or data, or histogram files")
    parser.add_argument("-ft", "--filetype", nargs="?", default="dump", const="all", 
                        choices=("dump", "data", "hist"), help="input file format, accepting dump, data, hist (default: %(default)s)")

    # real space or fourier space method 
    parser.add_argument("-m", "--method", nargs="?", default="real", const="all",
                        choices=("real", "fft"), help="Real space or fourier space method (default: %(default)s)")

    # real space method histogram construction arguments 
    parser.add_argument("-ord", "--ordered", action="store_false", 
                        help="Keep atoms ordered as in input file  (default: %(default)s)")
    parser.add_argument("-pbc", "--pbc", action="store_true", 
                        help="enable 3D periodic boundary conditions (default: %(default)s)") 
    parser.add_argument("-rc", "--rcut", type=float, default="inf",
                        help="cutoff radius in Angstrom, or 'inf' for full N^2 computation. Must be finite value if PBC are used (default: %(default)s)")
    parser.add_argument("-rp", "--rpartition", type=float, default=10.0, 
                        help="size of spatial partitioning cells in Angstrom, used if finite rcut is given (default: %(default)s)") 
    parser.add_argument("-dr", "--dr", type=float, default=0.001, 
                        help="histogram spacing in Angstrom (default: %(default)s)")
    parser.add_argument("-hx", "--histexport", default="histogram.dat", 
                        help="export path of histogram file (default: %(default)s)")
    parser.add_argument("-fuzz", "--fuzzygrain", type=float, default=0.0,
                        help="fuzzy grain approximation for PBC (WIP!) (default: %(default)s)")
    parser.add_argument("-gsim", "--grainsim", nargs=2, type=float, default=None,
                        help=" mean radius and standard deviation in Ang of simulated grain size distribution (default: %(default)s)")

    # real space method spectrum construction options
    parser.add_argument("-smin", "--smin", type=float, default=0.3,
                        help="minimum s value of spectrum range (default: %(default)s)")
    parser.add_argument("-smax", "--smax", type=float, default=1.2,
                        help="maximum s value of spectrum range (default: %(default)s)")
    parser.add_argument("-spts", "--spoints", type=int, default=200,
                        help="maximum s value of spectrum range (default: %(default)s)")
    parser.add_argument("-damp", "--infdamping", action="store_true", 
                        help="apply infinite medium dampening (default: %(default)s)")
    parser.add_argument("-ccor", "--continuumcorrection", action="store_true", 
                        help="apply continuum correction dampening (default: %(default)s)")

    # export path of final spectrum
    parser.add_argument("-sx", "--specexport", default="spectrum.dat", 
                        help="export path of spectrum file (default: %(default)s)")

    args = parser.parse_args()

    # use numpy inf for consistency 
    if args.rcut == float("inf"):
        args.rcut = np.inf

    # check if input file exists
    fpath = Path(args.inputfile)
    try:
        _path = fpath.resolve(strict=True)
    except FileNotFoundError:
        pass 

    # build or import the histogram
    filedat = ReadFile(fpath, filetype=args.filetype)
    filedat.load(shuffle = ~args.ordered)

    # fft based method
    if args.method == "fft":
        sfac = StructureFactor(filedat)
        sfac.build_structurefactor_fftw()
        spectrum = sfac.spectrum

    # real space based method
    elif args.method == "real":
        if args.filetype=="hist":
            doskip = True 
        else:
            doskip = False

            cspec = ComputeSpectrum(filedat, rpartition=args.rpartition, pbc=args.pbc, rcut=args.rcut, skip=doskip, fuzzy=args.fuzzygrain, gsim=args.grainsim)
            cspec.build_histogram(dr=args.dr)

            if args.filetype != "hist":
                if (me == 0):
                    print ("\nExporting histogram to file %s" % args.histexport)
                    # write additional parameters needed for computing the spectrum to header
                    natoms = cspec.readfile.natoms
                    hheader = "rcut %f\nnatoms %d\nnrho %f" % (cspec.rcut, natoms, natoms/np.product(cspec.readfile.box))

                    if args.grainsim:
                        np.savetxt(args.histexport, np.c_[cspec.ri, cspec.binlist], fmt=("%f", "%f"), header=hheader)
                    else:
                        np.savetxt(args.histexport, np.c_[cspec.ri, cspec.binlist], fmt=("%f", "%d"), header=hheader)
                    mpiprint ()

            spectrum = cspec.build_debyescherrer(args.smin, args.smax, args.spoints, damp=args.infdamping, ccorrection=args.continuumcorrection, nrho=cspec.nrho)


    if (me == 0):
        print ("\nExporting DS spectrum to file %s" % args.specexport)
        np.savetxt(args.specexport, spectrum)
        print ()
    comm.barrier()

    return 0

if __name__ == "__main__":
    main()
    if mode == 'MPI':
        MPI.Finalize()



