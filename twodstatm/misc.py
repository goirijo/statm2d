import numpy as np
import math

def is_close(a, b, rel_tol=1e-09, abs_tol=0.0):
    """Float errors are annoying"""
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def is_unitary(M):
    """Well... kind of unitary. Checks if
    M*M.T=I

    :M: 2x2 matrix
    :returns: TODO

    """
    I=np.identity(2)
    uniproduct=M*M.T

    return np.allclose(uniproduct,I)

def vec_angle(vec1, vec2):
    """Calculate the angle in degrees between
    two vectors

    :vec1: 2x1 vector
    :vec2: 2x1 vector
    :returns: float

    """
    radangle=math.atan2(vec2[1],vec2[0])-math.atan2(vec1[1],vec1[0])
    angle=math.degrees(radangle)
    
    return angle
