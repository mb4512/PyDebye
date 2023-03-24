#!/usr/bin/env python3
import numpy as np
import scipy
import sys, os, time

def get_first_element_in_set(s):
    for e in s:
        break
    return e


def main ():

    assert len(sys.argv) > 1, "Error: please supply the path to the diffraction pattern."

    fpath = sys.argv[1]
    assert os.path.isfile(fpath), "Error: could not find file {}. Does it exist?".format(fpath)

    # import data
    rawdata = np.loadtxt(fpath, skiprows=2) 
    kpts = rawdata[:,:3]
    ints = rawdata[:,-1]

    # build KD-tree of kpt data
    kdtree = scipy.spatial.KDTree(kpts)    

    # find nearest neighbours of every 1000th point
    dist,_ = kdtree.query(kpts[::1000], k=2)
    dist = np.array(dist)[:,1]

    # determine cluster resolution as multiples of the mean of distances within 1-std of distances
    meandist = np.mean(dist)
    stddist = np.std(dist)
    dist = dist[dist < stddist]
    meandistance = np.mean(dist)
    dvoxel = 30*meandistance 

    # determine voxel bounding boxes
    kmin = np.min(kpts, axis=0)
    kmax = np.max(kpts, axis=0)
    kdim = kmax-kmin
    nx,ny,nz = 5+(kdim/dvoxel).astype(int)

    # create voxel grid
    vgrid = np.zeros((nx,ny,nz), dtype=int)

    # convert k points to ijk format
    kpts_ijk = ((kpts-kmin+dvoxel)/dvoxel).astype(int)

    # populate voxel grid 
    unique_ijk,inverse_ijk = np.unique(kpts_ijk, axis=0, return_inverse=True)
    for _ijk in unique_ijk:
        vgrid[tuple(_ijk)] = 1

    print ("Sparsity of voxel grid: %.5f%%" % (100.*np.sum(vgrid)/vgrid.size))

    # iterate over elements containing atoms and build clusters
    vindices = np.argwhere(vgrid) 
    iremainder = np.copy(vindices)

    print ("Identified clusters of size", end="")
    sys.stdout.flush()

    # represent indices as as set, so we can use very efficient set comparison methods 
    iremainder = set([tuple(_row) for _row in iremainder])

    clusters = []
    while (len(iremainder) > 0):

        # get indices of neighbouring points
        icheck = set([get_first_element_in_set(iremainder)])
        cluster_indices = set([get_first_element_in_set(iremainder)])

        while len(icheck) > 0:
            # look for nebs around the first point in the list
            _icheck = get_first_element_in_set(icheck)
            _i,_j,_k = _icheck
            new_indices = []
            for _io in range(-1,2): # yeah triple loop, so what
                _iv = _i + _io
                for _jo in range(-1,2):
                    _jv = _j + _jo
                    for _ko in range(-1,2):
                        _kv = _k + _ko
                        if vgrid[_iv, _jv ,_kv] > 0:
                            new_indices += [(_iv, _jv, _kv)]
            new_indices = set(new_indices) 

            # all found neighbours that are not part of the cluster are added to the remaining neighbour search stack
            tocheck_indices = new_indices.difference(cluster_indices) 

            # newly found neighbours are merged with known neighbours
            cluster_indices = cluster_indices.union(tocheck_indices)

            if len(tocheck_indices) > 0:
                icheck.pop()
                icheck = icheck.union(tocheck_indices)
            else:
                icheck.pop()

        print ("...%d" % len(cluster_indices), end="")
        sys.stdout.flush()

        clusters += [cluster_indices]

        # represent indices as as set, so we can use very efficient set comparison methods 
        cluster_indices = set([tuple(_row) for _row in cluster_indices])

        # remove points from the global pool that are part of the cluster
        iremainder = iremainder.difference(cluster_indices)

    print ("...done.")

    # create map of voxel ijk to cluster id
    ijkmap = {}
    for ci,cluster in enumerate(clusters):
        for _voxel in cluster:
            ijkmap[tuple(_voxel)] = ci

    fpath = 'spots.xyz'
    print ("\nExporting into file %s..." % fpath, end='')
    sys.stdout.flush()

    with open(fpath, 'w') as ofile:
        ofile.write("%d\n" % len(kpts))
        ofile.write("# kx ky kz intensity id\n")

        # for each k point, convert to ijk format, fetch cluster id, and then write a line
        for ki in range(len(kpts)):
            _ijk = kpts_ijk[ki] 
            _cid = ijkmap[tuple(_ijk)]
            ofile.write("%16.8e %16.8e %16.8e %16.8e %8d\n" % (kpts[ki,0], kpts[ki,1], kpts[ki,2], ints[ki], _cid)) 
    print ("done.")
    sys.stdout.flush()

    # determine and print out first moments
    print ("Constructing cluster moments and total intensities...", end='')
    sys.stdout.flush()

    # build dictionary of the moments and total intensities of each cluster 
    moments = {}
    for ki in range(len(kpts)):
        _ijk = kpts_ijk[ki] 
        _cid = ijkmap[tuple(_ijk)]
        if _cid not in moments:
            moments[_cid] = [0,0]
        moments[_cid][0] += kpts[ki]*ints[ki]
        moments[_cid][1] += ints[ki]

    # normalise moment and convert dictionary into a list so we can sort by brightness 
    moment_list = []
    brightness_list = []
    for _cid in moments:
        moment_list += [moments[_cid][0]/moments[_cid][1]]
        brightness_list += [moments[_cid][1]]
    moment_list = np.vstack(moment_list)
    brightness_list = np.array(brightness_list)

    # sort by brightness in descending order
    intsort = np.flip(np.argsort(brightness_list))
    moment_list = moment_list[intsort]
    brightness_list = brightness_list[intsort]

    print ("done.")
    sys.stdout.flush()

    print ("\nkx (1/A) ky (1/A) kz (1/A) Brightness (a.u.)")
    print ("===========================================")
    for i in range(len(moment_list)):
        firstmom = moment_list[i]
        totint = brightness_list[i]
        print ("%16.8f %16.8f %16.8f %16.8e" % (firstmom[0], firstmom[1], firstmom[2], totint))

    return 0


if __name__=="__main__":
    main ()

