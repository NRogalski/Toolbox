

import numpy as np


def fun_Euclidean_Norm(x):

    # L2 row-wise normalization of array x

    if x.ndim==1:
        return x/np.sqrt(np.sum(x**2))
    else:
        return x/np.sqrt(np.sum(x**2,axis=1,keepdims=True))


def fun_Cotangent(v_0,v_1,v_2):

    # Cotangent of the angle at vertex v_0

    v_01,v_02 = v_1-v_0,v_2-v_0
    return np.dot(v_01,v_02)/np.sqrt(np.sum(np.cross(v_01,v_02)**2))


def fun_Axang2rotm(Axis,Angle):

    # Rotation matrix from axis (normalized) and angle (radians)

    Vx = np.array([[0,-Axis[2],Axis[1]],[Axis[2],0,-Axis[0]],[-Axis[1],Axis[0],0]])
    return np.eye(3)+np.sin(Angle)*Vx+(1-np.cos(Angle))*np.linalg.matrix_power(Vx,2)
    
