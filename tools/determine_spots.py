#!/usr/bin/env python3
import numpy as np
import scipy
import sys, os, time, math
from numba import jit

EXPORT = True 

def get_first_element_in_set(s):
    '''Return the first element in a set.''' 
    for e in s:
        break
    return e

@jit(nopython=True, fastmath=True)
def jminfunc(kpts, ints, i0, k0x, k0y, k0z, sxx, sxy, sxz, syy, syz, szz):

    sn =  np.std(ints)
    if sn == 0.0:    
        sn = ints[0]

    isn2 = 0.5/(sn*sn)
    logl = 0.0
    for ki in range(len(kpts)):
        kx = kpts[ki][0] - k0x 
        ky = kpts[ki][1] - k0y
        kz = kpts[ki][2] - k0z

        _dist = kx*sxx*kx + kx*sxy*ky + kx*sxz*kz + \
                ky*sxy*kx + ky*syy*ky + ky*syz*kz + \
                kz*sxz*kx + kz*syz*ky + kz*szz*kz  

        _idiff = ints[ki] - i0*math.exp(-.5*_dist)
        logl += isn2*_idiff*_idiff

    logl -= len(ints)*(.5*math.log(2*np.pi) + math.log(sn))
    return logl


@jit(nopython=True, fastmath=True)
def jmingrad(kpts, ints, i0, k0x, k0y, k0z, sxx, sxy, sxz, syy, syz, szz):

    sn =  np.std(ints)
    if sn == 0.0:    
        sn = ints[0]

    isn = 1./(sn*sn)
    grads = np.zeros(10, dtype=np.float64)
    for ki in range(len(kpts)):
        kx = kpts[ki][0] - k0x 
        ky = kpts[ki][1] - k0y
        kz = kpts[ki][2] - k0z

        _dist = kx*sxx*kx + kx*sxy*ky + kx*sxz*kz + \
                ky*sxy*kx + ky*syy*ky + ky*syz*kz + \
                kz*sxz*kx + kz*syz*ky + kz*szz*kz  

        _expo = math.exp(-.5*_dist)
        _idiff = ints[ki] - i0*_expo

        grads[0] +=    -isn*_idiff * (   _expo) # diff w.r.t i0
        grads[1] +=    -isn*_idiff * (i0*_expo) * (sxx*kx + sxy*ky + sxz*kz) # diff w.r.t k0x
        grads[2] +=    -isn*_idiff * (i0*_expo) * (sxy*kx + syy*ky + syz*kz) # diff w.r.t k0y
        grads[3] +=    -isn*_idiff * (i0*_expo) * (sxz*kx + syz*ky + szz*kz) # diff w.r.t k0z
        grads[4] += .5*isn*_idiff * (i0*_expo) * (  kx*kx) # diff w.r.t sxx
        grads[5] += .5*isn*_idiff * (i0*_expo) * (2*kx*ky) # diff w.r.t sxy
        grads[6] += .5*isn*_idiff * (i0*_expo) * (2*kx*kz) # diff w.r.t sxz
        grads[7] += .5*isn*_idiff * (i0*_expo) * (  ky*ky) # diff w.r.t syy
        grads[8] += .5*isn*_idiff * (i0*_expo) * (2*ky*kz) # diff w.r.t syz
        grads[9] += .5*isn*_idiff * (i0*_expo) * (  kz*kz) # diff w.r.t szz

    # rescale gradients to become comparable in magnitude 
    #grads[0] *= i0
    #grads[1:4] *= 1e-4
    #grads[4:] *= sxx+syy+szz

    return grads 



def get_cluster_indices(kpts, kpts_ijk, ijkmap):
    '''
    # determine how many points lie inside each voxel 
    unique_ijk,inverse_ijk = np.unique(kpts_ijk, axis=0, return_inverse=True)

    # fetch cluster id for each set of voxel indices
    cids = [ijkmap[tuple(_ijk)] for _ijk in unique_ijk] 
    
    # build dictionary
    clusterpoints = {_cid:inverse_ijk[_i] for _i,_cid in enumerate(cids)}

    print (clusterpoints[15])
    '''

    # loop over all k points and assign them to their cluster
    clusterpoints = {}
    for ki,_ijk in enumerate(kpts_ijk):

        # fetch cluster id belonging to this voxel 
        _cid = ijkmap[tuple(_ijk)]

        # add the point ID to the list of point IDs belonging to this cluster 
        if _cid not in clusterpoints:
            clusterpoints[_cid] = [ki]
        else:
            clusterpoints[_cid] += [ki]

    return clusterpoints


def malahabonis_filter(clusterpoints, sigmainv, moments1st, kpts):

    # loop over all clusters and only keep points that lie within the FWHM 
    filtered_clusterpoints = {}
    for _cid in clusterpoints:

        # compute malahabonis distances squared: d^2 = k_i*S_ij*k_j
        dk = kpts[clusterpoints[_cid]] - moments1st[_cid]
        dmala = np.einsum('...i,ij,...j->...', dk, sigmainv[_cid], dk)

        # filter out points lying outside half the FWHM
        _bool = (dmala < 2.*np.log(2.))
        if np.sum(_bool) > 0:
            filtered_clusterpoints[_cid] = np.array(clusterpoints[_cid])[_bool]

    return filtered_clusterpoints


def get_moments(kpts, ints, clusterpoints): 

    # build dictionaries for the moments and total intensities of each cluster
    nclusters = len(clusterpoints.keys())
    moments1st = {_cid:np.zeros(3) for _cid in range(nclusters)}
    moments2nd = {_cid:np.zeros((3,3)) for _cid in range(nclusters)}
    brightness = {_cid:0.0 for _cid in range(nclusters)}    # brightness needed to normalise
    peakintensity = {_cid:0.0 for _cid in range(nclusters)} # brightest point in the cluster 

    # loop over all clusters and compute their first and second moments
    for _cid in clusterpoints.keys():
        kids = clusterpoints[_cid]

        moments1st[_cid] = np.sum(kpts[kids].T*ints[kids], axis=1)/np.sum(ints[kids])
        moments2nd[_cid] = 0.0 
        _kmean = moments1st[_cid]
        for _kid in kids:
            moments2nd[_cid] += np.outer(kpts[_kid]-_kmean, kpts[_kid]-_kmean)*ints[_kid]
        moments2nd[_cid] *= 1/np.sum(ints[kids])

        brightness[_cid]    = np.sum(ints[kids]) # total brightness 
        peakintensity[_cid] = np.max(ints[kids]) # peak maximum (for fitting)

    return moments1st, moments2nd, brightness, peakintensity


def dict_to_sorted_list(sortdict, tosortdict):
    order = np.flip(np.argsort(list(sortdict.values())))
    return np.array(list(tosortdict.values()))[order]


def check_derivative(func, dfunc, rscale, p0, dp=1e-5):

    ngradient = np.zeros(len(p0), dtype=float)
    for i in range(len(p0)):
        pc = np.copy(p0) 
        pc[i] +=   dp*rscale[i]
        fplus = func(pc)
        pc[i] -= 2*dp*rscale[i]
        fminu = func(pc)
        ngradient[i] = (fplus-fminu)/(2*dp*rscale[i])

    print ()
    print ("Testing derivative:")
    print ("===================")

    print ("Numerical gradient:")
    print ("%16.9f "*len(ngradient) % tuple(ngradient))
    
    agradient = dfunc(p0)
    print ("Analytical gradient:")
    print ("%16.9f "*len(agradient) % tuple(agradient))


def fisher_information_criterion(dfunc, rscale, p0, dp=1e-6, report=False):

    # compute uncertainies in k-position using Fisher information criterion
    khessian = np.zeros((len(p0),len(p0)), dtype=float)
    for i in range(10):
        pc = np.copy(p0)
        pc[i] +=   dp*rscale[i]
        dfplus = dfunc(pc)
        pc[i] -= 2*dp*rscale[i]
        dfminu = dfunc(pc)
        khessian[i] = (dfplus-dfminu)/(2*dp*rscale[i])

    # symmetrise
    khessian = .5*(khessian + khessian.T)
    varcovar = np.linalg.inv(khessian)

    def isPSD(A, tol=1e-5):
      E = np.linalg.eigvalsh(A)
      return np.all(E > -tol)

    psd = isPSD(varcovar)

    if report and not psd: 
        print (">>>>>>>>>>>>>>>>>>>> Warning: variance-covariance matrix is not positive semi-definite!")
        print (">>>>>>>>>>>>>>>>>>>> Eigenvalues:", np.linalg.eigvalsh(varcovar))

    if report:
        print ()
        print ("Fisher information matrix:")
        for _row in khessian:
            print ("%12.5e "*len(_row) % tuple(_row))
        print ()
        print ("Estimated Variance-Covariance matrix:")
        for _row in varcovar: 
            print ("%12.5e "*len(_row) % tuple(_row))
        print ()

    return varcovar, psd

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
    dist = dist[np.abs(dist-meandist) < stddist]
    meandistance = np.mean(dist)
    dvoxel = 40*meandistance 

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

    # represent indices as a set, so we can use very efficient set comparison methods 
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

    print ("...done.\n")

    # create map of voxel ijk to cluster id
    ijkmap = {}
    for ci,cluster in enumerate(clusters):
        for _voxel in cluster:
            ijkmap[tuple(_voxel)] = ci


    # determine and print out first moments
    print ("Constructing cluster moments and total intensities...", end='')
    sys.stdout.flush()

    # assign atoms to clusters
    clusterpoints = get_cluster_indices(kpts, kpts_ijk, ijkmap)

    # build dictionaries of the moments and total intensities of each cluster
    moments1st, moments2nd, brightness, peakintensity = get_moments(kpts, ints, clusterpoints)

    # sort by brightness in descending order
    moments1st_list = dict_to_sorted_list(brightness, moments1st)
    brightness_list = dict_to_sorted_list(brightness, brightness)
 
    print ("done.\n")
    sys.stdout.flush()

    print ("       kx (1/A)         ky (1/A)         kz (1/A)   brightness (a.u.)")
    print ("=====================================================================")
    for i in range(len(moments1st_list)):
        mom1st = moments1st_list[i]
        bright = brightness_list[i]
        print ("%16.8f %16.8f %16.8f %16.8e" % (mom1st[0], mom1st[1], mom1st[2], bright))


    # the second central moments are equal to the variance-covariance terms of a multinomial gaussian
    # pdf for intensity: I(k_i) = I_0 * exp(-0.5*(k_i-mu_i)*(Sigma^-1)_ij*(k_j-mu_j))
    # where I_0 is peak cluster intensity, mu_i are 1st moments, and Sigma is the matrix of 2nd central moments
    
    # build initial guesses for covariance and precision matrices
    nclusters = len(moments1st.keys())
    sigma    = {_cid:moments2nd[_cid] for _cid in range(nclusters)}
    sigmainv = {_cid:np.linalg.inv(moments2nd[_cid]) for _cid in range(nclusters)}

    print ("\nRemoving points outside the FWHM of the peaks...")

    # for each cluster, filter out points that lie outside the FWHM
    filtered_clusterpoints = malahabonis_filter(clusterpoints, sigmainv, moments1st, kpts)

    # build dictionaries of the moments and total intensities of each filtered cluster
    new_moments1st, new_moments2nd, new_brightness, peakintensity = get_moments(kpts, ints, filtered_clusterpoints)

    # sort by brightness in descending order
    moments1st_list = dict_to_sorted_list(new_brightness, new_moments1st)
    brightness_list = dict_to_sorted_list(new_brightness, new_brightness)
 
    print ("done.\n")
    sys.stdout.flush()

    print ("       kx (1/A)         ky (1/A)         kz (1/A)   brightness (a.u.)")
    print ("=====================================================================")
    for i in range(len(moments1st_list)):
        mom1st = moments1st_list[i]
        bright = brightness_list[i]
        if bright == 0.0:
            continue
        print ("%16.8f %16.8f %16.8f %16.8e" % (mom1st[0], mom1st[1], mom1st[2], bright))


    # minimisation time!
    fitted_spots = []
    print ("\nMaximum likelihood fitting of spots", end="")
    sys.stdout.flush()
    
    success = {}  # dict for storing success of fit 
    varcovar = {} # dict for storing variance-covariance matrix
    psd = {}      # dict for storing whether variance-covariance matrix is semi-positive definite
    for _cid in filtered_clusterpoints:
        _kpts = kpts[filtered_clusterpoints[_cid]]
        _ints = ints[filtered_clusterpoints[_cid]]

        if new_brightness[_cid] == 0.0:
            continue 

        minfunc = lambda pi: jminfunc(_kpts, _ints, *pi)
        mingrad = lambda pi: jmingrad(_kpts, _ints, *pi)

        k0 = new_moments1st[_cid]
        sij = 5*sigmainv[_cid] # make initial covariance matrix larger for better initial fit guess
        p0 = [peakintensity[_cid], k0[0], k0[1], k0[2], sij[0,0], sij[0,1], sij[0,2], sij[1,1], sij[1,2], sij[2,2]]
        res = scipy.optimize.minimize(minfunc, p0, options={"disp": False, "maxiter":1000, "gtol":1e-9, "maxCGit": 500, "maxfun": 1000}, method="TNC", jac=mingrad)

        success[_cid] = res.success
        if not res.success:
            print ("peak id {} fit unsuccessful, with message: {}".format(_cid, res.message), end="")
            print (". Doing Nelder-Mead instead: ", end="") 
            sys.stdout.flush()

            res = scipy.optimize.minimize(minfunc, p0, options={"disp":False, "maxiter":20000, "xatol":1e-4, "adaptive":True}, method='Nelder-Mead')
            success[_cid] = res.success
            if res.success:
                print ("successful.", end="")
            else:
                print ("also unsuccessful.", end="")
            sys.stdout.flush()


        '''
        print ()
        print ("Spot %d:" % _cid)
        print ("==========")
        print ("Initial results:")
        print ("%16.9f "*len(p0) % tuple(p0))

        print ("Fitting results:")
        print ("%16.9f "*len(res.x) % tuple(res.x))
        print ()

        # displace w.r.t scale
        check_derivative(minfunc, mingrad, rscale, res.x, dp=1e-5)
        '''    

        '''
        for i in range(len(ngradient)):
            #pc = np.copy(p0)
            pc = np.array(res.x)
            pc[i] +=   dp*rscale[i]
            fplus = jminfunc(_kpts, _ints, *tuple(pc))
            pc[i] -= 2*dp*rscale[i]
            fminu = jminfunc(_kpts, _ints, *tuple(pc))
            ngradient[i] = (fplus-fminu)/(2*dp*rscale[i])

        print ()
        print ("Spot %d:" % _cid)
        print ("==========")

        print ("Numerical gradient:")
        print ("%16.9f "*len(ngradient) % tuple(ngradient))
        
        #agradient = jmingrad(_kpts, _ints, *p0)
        agradient = jmingrad(_kpts, _ints, *tuple(res.x))
        print ("Analytical gradient:")
        print ("%16.9f "*len(agradient) % tuple(agradient))
        '''

        # set some scales for central finite difference 
        ss = np.abs(res.x[4]+res.x[7]+res.x[9]) # tr(sigma_ij) 
        ks = meandistance
        rscale = np.array([peakintensity[_cid], ks, ks, ks, ss, ss, ss, ss, ss, ss])

        # compute uncertainies in k-position using Fisher information criterion
        varcovar[_cid], psd[_cid] = fisher_information_criterion(mingrad, rscale, res.x, report=False)
        if not psd[_cid]:
            print ("Warning: variance-covariance matrix for peak %d is not positive semi-definite!" % _cid, end="")
            sys.stdout.flush()

        # check if spot centre has moved away considerably during optimisation            
        _kres = np.array([res.x[1], res.x[2], res.x[3]])
        if np.linalg.norm(_kres-k0) > dvoxel:
            print ("Warning: large change in spot position in cluster %d! Something went wrong in the minimisation." % _cid, end="")            
            print ("Initial spot position:", k0[0], k0[1], k0[2], end="")
            print ("Fitting spot position:", res.x[1], res.x[2], res.x[3], end="")
            sys.stdout.flush()

        fitted_spots += [[res.x[1], res.x[2], res.x[3], _cid, new_brightness[_cid]]] 
        print ('.', end="")
        sys.stdout.flush()

    print ('done.')

    fitted_spots = np.array(fitted_spots)
    order = np.flip(np.argsort(fitted_spots[:,-1]))
    fitted_spots = fitted_spots[order]

    print ("\nPeak positions after maximium likelihood fitting:")
    print ("       kx (1/A)         ky (1/A)         kz (1/A)   brightness (a.u.)  cluster   sigma_kx (1/A)   sigma_ky (1/A)   sigma_kz (1/A)")
    print ("=================================================================================================================================")
    for _fsp in fitted_spots:
        _vcv = varcovar[_fsp[3]]
        skx = np.sqrt(_vcv[1,1])
        sky = np.sqrt(_vcv[2,2])
        skz = np.sqrt(_vcv[3,3])
        print ("%16.8f %16.8f %16.8f %16.8e %8d %16.8f %16.8f %16.8f" % (_fsp[0], _fsp[1], _fsp[2], _fsp[4], _fsp[3], skx, sky, skz))


    if EXPORT:
        fpath = 'spots.xyz'
        print ("\nExporting into file %s..." % fpath, end='')
        sys.stdout.flush()
        with open(fpath, 'w') as ofile:
            ofile.write("%d\n" % len(kpts))
            ofile.write("# kx ky kz intensity id malahabonis\n")

            for _cid in clusterpoints:
                if _cid in filtered_clusterpoints:
                    filtered_ki = filtered_clusterpoints[_cid]

                for ki in clusterpoints[_cid]:
                    if ki in filtered_ki:
                        md = 1
                    else:
                        md = 0
                    ofile.write("%16.8e %16.8e %16.8e %16.8e %8d %2d\n" % (kpts[ki,0], kpts[ki,1], kpts[ki,2], ints[ki], _cid, md))

        print ("done.")
        sys.stdout.flush()

    return 0


if __name__=="__main__":
    main ()



