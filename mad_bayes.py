# Rohan Banerjee
# MAD Bayes code for 6.882 project: Bayesian non-parametric modeling of 3D point clouds

## Based on Kulis and Jordan, 2012

import numpy as np

def mad_bayes_dpgmm(X, lmb = 0.25, num_iter = 10):
    """
    Runs MAD-Bayes (small-variance) inference algorithm for a GMM.
    
    X: input data (N x D)
    
    2 hyperparameters (distance and convergence):
    lmb: cluster penalty parameter - lower indicates a greater willingness to create clusters
    num_iter: number of iterations to run algorithm
    
    Returns:
    K: estimated K
    z: inferred z-labels for each cluster
    mu: estimated cluster means
    """
    
    # initialization
    (N,D) = X.shape
    K = 1
    mu = np.array([np.average(X, axis=0)])
    z = np.zeros((N,1))    # cluster assignments - initially everything is assigned to the same cluster
    
    #print "initial mean: ", mu
    #print "closest distance to initial mean: ", np.amin(np.linalg.norm(X-mu,axis=1))
    
    for i in xrange(num_iter):  
        # check to see, for each data point, if the closest cluster is larger than lambda away
        for j in xrange(N):
            x = X[j]
            # calculate distances between point and all of the existing clusters
            distances = np.linalg.norm(x - mu,axis=1)   # K x 1 array (x-mu is K x D)
            min_dist = np.amin(distances)               # minimum distance to any of existing clusters
            if min_dist > lmb:
                # create a new cluster!
                K+=1
                z[j] = K
                mu = np.vstack((mu,x))
            else:
                # assign x to min-dist cluster
                z[j] = np.argmin(distances)

        #print "mu before re-assignment: ", mu     
        
        # check to see if any of the cluster means have NOTHING assigned to them
        # if so, remove them and re-adjust the z's accordingly.
        k = 0
        while k < K:
            (indices, _) = np.where(z == k)
            if len(indices) == 0:
                # delete mean
                mu = np.delete(mu, (k), axis=0) 
                
                # decrement K
                K -= 1
                
                # re-label all z's. everything that had a label greater than k needs to 
                # be downshifted by 1
                (indices_two, _) = np.where(z > k)
                z[indices_two] -= 1
                
                # decrement counter to reflect this re-labeling
                k -= 1
                
            k += 1 


        # re-compute cluster means (inefficently)
        for k in xrange(K):
            (indices, _) = np.where(z == k)
            #print indices
            assert len(indices) != 0
            mu[k] = np.average(X[indices],axis=0)
            
        #print "K: ", K
        
        assert K == len(mu)
        #print "mu after re-assignment: ", mu
        
    #print "final z: ", z
    
    return K,z,mu

## Based on Straub et al., 2015

def mad_bayes_dpvmf(X, lmb = -1, num_iter = 10):
    """
    Runs MAD-Bayes (small-variance) inference algorithm for a vMF mixture.
    
    X: input data (N x D)
    
    2 hyperparameters (distance and convergence):
    lmb: cluster penalty parameter 
        - lower indicates a lower willingness to create clusters
        - should be in [-2, 0]
    num_iter: number of iterations to run algorithm
    
    Returns:
    K: estimated K
    z: inferred z-labels for each cluster
    mu: estimated cluster means
    """
    
    #initialization
    (N,D) = X.shape
    K = 1
    sum_dir = np.sum(X, axis=0)
    mu = np.array([sum_dir/np.linalg.norm(sum_dir)])
    z = np.zeros((N,1))    # cluster assignments - initially everything is assigned to the same cluster
    
    #print "initial mean: ", mu
    #print "closest distance to initial mean: ", np.amin(np.linalg.norm(X-mu,axis=1))
    
    #print "--------------------"
    
    for i in xrange(num_iter): 
        #if i % 10 == 0:
        #    print "iteration number: ", i
        
        # check to see, for each data point, if the closest cluster is larger than lambda away, 
        # in the inner product sense
        for j in xrange(N):
            x = X[j]
            # calculate inner product between point and all of the existing clusters
            inner_prods = np.dot(mu, np.transpose(x))           # K x 1 array (x is 1 X D, mu is K x D)
            max_inner_prod = np.amax(inner_prods)               # max inner product to any of existing clusters
            if max_inner_prod < lmb + 1:
                # create a new cluster!
                K+=1
                z[j] = K
                mu = np.vstack((mu,x))
            else:
                # assign x to min-dist cluster
                z[j] = np.argmax(inner_prods)

        #print "mu before re-assignment: ", mu     
        
        
        # check to see if any of the cluster means have NOTHING assigned to them
        # if so, remove them and re-adjust the z's accordingly.
        k = 0
        while k < K:
            (indices, _) = np.where(z == k)
            if len(indices) == 0:
                # delete mean
                mu = np.delete(mu, (k), axis=0) 
                
                # decrement K
                K -= 1
                
                # re-label all z's. everything that had a label greater than k needs to 
                # be downshifted by 1
                (indices_two, _) = np.where(z > k)
                z[indices_two] -= 1
                
                # decrement counter to reflect this re-labeling
                k -= 1
                
            k += 1 
            
        #if i % 10 == 0:
        #    print "estimated K: ", K

            
        # re-compute cluster means (inefficently)
        for k in xrange(K):
            (indices, _) = np.where(z == k)
            #print indices
            assert len(indices) != 0
            sum_xs = np.sum(X[indices],axis=0)
            mu[k] = sum_xs/np.linalg.norm(sum_xs)
            
        #print "K: ", K
        
        assert K == len(mu)
        
        #print "mu after re-assignment: ", mu
        
        
        
    #print "final z: ", z
    
    #print [np.where(z==k)[0] for k in xrange(K)]
    
    return K,z,mu