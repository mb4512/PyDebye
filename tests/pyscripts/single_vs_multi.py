#!/usr/bin/env python3
import numpy as np
import glob, os

def main ():

    prefix = "svm"
    suffix = "dat"
    refdir = "reference_output"
    gendir = "generated_output"

    # fetch reference files
    reffiles = glob.glob("{}/{}*.{}".format(refdir, prefix, suffix))
  
    assert os.path.isdir(refdir), "Error: could not find directory %s for reference files." % refdir
    assert os.path.isdir(gendir), "Error: could not find directory %s for generated files." % gendir
    if len(reffiles) == 0:
        raise Exception("Error, no reference files found in directory %d" % refdir) 
 
    # from reference files, create names of generated files 
    genfiles = ["{}/{}".format(gendir, os.path.split(_rfile)[1]) for _rfile in reffiles]

    # check if generated files exist, otherwise throw error and exit 
    for _ig,_gfile in enumerate(genfiles):
        assert os.path.isfile(_gfile) == True, "Error: could not find generated file %s belonging to reference file %s." % (_gfile, reffiles[_ig])

    for _i in range(len(reffiles)):
        rf = reffiles[_i]
        gf = genfiles[_i]
        print ("Comparing file %s to file %s..." % (rf, gf), end=" ")
        f0 = np.loadtxt(rf)
        f1 = np.loadtxt(gf)
        check = np.allclose(f0, f1, rtol=1e-10, atol=1e-12)
        assert check == True, "Error: difference detected between files {} and {}.".format(rf, gf)
        print ("passed.")

    print ()
    print ("=============================================================================")
    print ("Passed check to within 1e-10 relative tolerance and 1e-12 absolute tolerance.")
    print ("=============================================================================")
    print ()

    return 0

if __name__=="__main__":
    main ()
