# Rohan Banerjee
# dataset-generating code for 6.882 project: Bayesian non-parametric modeling of 3D point clouds

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sklearn
import sklearn.metrics.cluster

import quaternion_utils as qutils
import quaternion

####
# Dataset 1: cluster dataset
####

def generate_cluster_dataset(K=10,variance=1):
    """
    Generates a toy dataset with a fixed number of Gaussian clusters.
    
    Returns:
    X - data
    z - true cluster assignments
    
    """

    ## 3D GMM with a fixed number of clusters, K and dimension, D
    D = 3
    # for now, fix the covariance matrix for each cluster
    covariance = variance*np.eye(3)

    # sample the means. uniformly drawn from a box [-W,W] x [-W,W] x [-W,W]
    # shape: K x D
    W = 10
    np.random.seed(1) # for reproducibility
    means = np.random.uniform(0,2*W,(K,D))-W

    # (inefficiently) sample the datapoints. for now, the cluster weights are equal
    N = 1000
    z = np.random.randint(0,K-1,(N,1))
    X = np.zeros((N,D))

    # consider each cluster, find the points that were assigned to that cluster, and then sample from that cluster
    for k in xrange(K):
        (points_in_k, _) = np.where(z == k)
        samples = np.random.multivariate_normal(means[k], covariance, size=(len(points_in_k)))
        X[points_in_k,:] = samples  

    return X, z

def plot_clusters(X, z):
    """
    Plots clusters, given by data matrix and labels z
    
    X: N X 3 matrix
    z: vector
    
    """
    # from https://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html

    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    xs = X[:,0]
    ys = X[:,1]
    zs = X[:,2]

    ax.scatter(xs,ys,zs, c = z)

    plt.show()
    
    
####
# Dataset 2: hypersphere dataset
####
 
def apply_axis_angle(v, k, theta):
    """
    Applies an axis-angle rotation about k by theta to vector v

    v: vector to be rotated
    k: unit vector that we wish to rotate around
    theta: angle to rotate

    Returns: rotated vector v
    
    """    

    assert np.linalg.norm(k) != 0

    # generate rotation quaternion
    rotation_quat = qutils.axisangle_to_q(k, theta)

    # rotate vector v according to rotation quaternion    
    v_rot = qutils.qv_mult(rotation_quat, v)
    
    return v_rot

def rotate_points(data, orig_vector, dest_vector):
    """
    Rotates data vectors in S_2 based on rotation from orig_vector to final_vector.
    
    data: data points to be rotated (N x 3)
    orig_vector: initial orientation direction
    dest_vector: final orientation direction
    
    """
    axis = np.cross(orig_vector, dest_vector)
    angle = np.arccos(np.dot(orig_vector, dest_vector)/(np.linalg.norm(orig_vector) * np.linalg.norm(orig_vector)))
    
    # TODO: maybe need to change this to np.apply_along_axis, so ?
    
    # if angle is (sufficently close) to 0, then make the axis random
    angle_tol = 1e-5
    if np.abs(angle) <= angle_tol:
        axis = np.array([0,0,1])
        
    # if angle is (sufficiently close) to 180, then we can choose any axis in the x-y plane
    # for simplicity, just choose x-axis
    if np.abs(angle - np.pi) <= angle_tol:
        axis = np.array([1,0,0])
     
    assert np.linalg.norm(axis) != 0, "Axis of rotation should not be the zero vector."
    
    def axis_angle_transform(point):
        return apply_axis_angle(point, axis, angle)
    
    return np.apply_along_axis(axis_angle_transform, 1, data)

def sample_from_vMF(mu, kappa, num_samples):
    """
    Uses method from Jakob (2012) and Jung (2009) and others 
    to sample from a vMF distribution on the 2-sphere.
    
    Parameters:
    mu - direction vector (3 x 1)
    kappa - concentration parameter (constant)
    N - number of samples
    
    Returns:
    omega - the sampled points
    
    """
    np.random.seed(32)  # for reproducibility
        
    #### Step 1: sample from V - uniformly distributed on unit circle
    thetas = np.random.uniform(low = -np.pi, high = np.pi, size=num_samples)
    v = np.transpose(np.vstack((np.cos(thetas), np.sin(thetas))))
        
    #### Step 2: sample from W
    # uses inverse CDF method
    
    # generate uniform samples on [0,1]
    xi = np.random.uniform(size=num_samples)
    
    # pass through inverse CDF
    
    w = 1.0/kappa * np.log(np.exp(-kappa) + 2*xi*np.sinh(kappa))
    w = np.reshape(w, (num_samples,1))
    
    #### Step 3: combine to form (w_kappa) samples
        
    omega = np.hstack((np.sqrt(1-w**2)*v,w))
                
    #### Step 4: finally, apply rotation matrix to conver from (0,0,1) to mu
    
    mu_ref = np.array([0,0,1])    # standard, reference direction
        
    omega = rotate_points(omega, mu_ref, mu)
    
    #print "(final, rotated) omega: ", omega
    
    return omega

def generate_hypersphere_dataset(K=10, tau=100):
    """
    
    Generates a toy 3D hypersphere dataset with a fixed number of vMF clusters. 
    
    [Goal is to generate data from a finite-dimensional vMF mixture]
    
    Based on Straub (2015)
    
    Hyperparameters:
    - alpha: Dirichlet hyperparameter
    - K: number of clusters
    - mu_0:  prior direction for vMF means
    - tau_0: prior concentration for vMF means
    
    Returns:
    X - data
    z - true surface cluster assignments
    """

    np.random.seed(5) # for reproducibility
    
    ## specify hyper-parameters
    alpha = 1 # Dirichlet hyperparameter
    #K = 10      # number of clusters
    D = 3       # dimensionality of the data
    
    ## sample cluster weights and directional means
    
    # weights are Dir(alpha)
    weights = np.random.dirichlet(alpha*np.ones(K))
    
    # means are drawn from a vMF(mu_0, tau_0)
    mu_0 = np.array([1,0,0])
    tau_0 = 1
    mu = sample_from_vMF(mu_0, tau_0, num_samples=K)
    
    #print "weights:", weights
    #print "mu:", mu

    ## sample datapoints
    
    # (inefficiently) sample the datapoints. for now, the cluster weights are equal
    N = 1000
    #tau = 100    # hyperparameter: dispersion factor for generating data
    #z = np.random.multinomial(N, weights, size=1)    # TODO: might be more efficient, but for now will skip
    z = np.random.choice(np.arange(K),N,p=weights)
    z = np.reshape(z,(N,1))
    X = np.zeros((N,D))
    
    # consider each cluster, find the points that were assigned to that cluster, and then sample from that cluster
    for k in xrange(K):
        (points_in_k, _) = np.where(z == k)
        if(len(points_in_k) > 0):
            samples = sample_from_vMF(mu[k], tau, num_samples=len(points_in_k))
            X[points_in_k,:] = samples  

    return X, z

def plot_vMF_data(X, z=None):
    """
    Plots raw data, given by data matrix X.
    
    X: N X 3 matrix    
    z: (optional) cluster assignments
    """
    # from https://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html

    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    xs = X[:,0]
    ys = X[:,1]
    zs = X[:,2]
    
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_zlim([-1,1])

    if z is None:
        ax.scatter(xs,ys,zs)
    else:
        ax.scatter(xs,ys,zs, c = z)

    plt.show()