# Quaternion code
# contains useful methods that use the quaternion class cited below.
# sources: 
# - https://stackoverflow.com/questions/4870393/rotating-coordinate-system-via-a-quaternion
# - https://github.com/moble/quaternion/blob/master/README.md

import numpy as np
import quaternion

def normalize(q):
    """
    Returns the unit quaternion (or vector) corresponding to q.
    
    """
    return q/np.linalg.norm(q)

def qv_mult(q1, v1):
    """
    Rotates vector v1 according to quaternion q1.
    
    """
    (vx, vy, vz) = v1
    q2 = np.quaternion(0.0, vx, vy, vz)
    product = (q1 * q2) * np.conjugate(q1)
    # only take the last three components to get the vector
    return np.array([product.x, product.y, product.z])

def axisangle_to_q(v, theta):
    """
    Takes axis-angle representation 
    and converts it into a quaternion
    
    """
    v = normalize(v)
    x, y, z = v
    theta /= 2
    w = np.cos(theta)
    x = x * np.sin(theta)
    y = y * np.sin(theta)
    z = z * np.sin(theta)
    return np.quaternion(w,x,y,z)
