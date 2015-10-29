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

def independent_indices(matlist):
    """Find the elements in the list that are
    linearly independent. This function is
    mainly for finding the linearly independent
    tensor basis elements.
    
    First unroll each matrix in the list as a four element vector
    L=[mxx,mxy,myx,myy]

    Put all (presumably 4) vectors in a matrix
    M=[L1,L2,L3,L4]

    If you take the QR decomposition of M, the columns of R
    with non-zero value along the diagonal correspond to
    linearly independent columns of L

    :matlist: list of 2x2 matrix
    :returns: tuple of int

    """
    vectorized=[]
    for mat in matlist:
        vectorized.append(mat.ravel())

    vectorized=np.array(vectorized)
    vectorized=np.squeeze(vectorized)

    Q,R=np.linalg.qr(vectorized.T)
    dependent=np.isclose(R.diagonal(),0)
    independent=np.invert(dependent)

    return np.where(independent)[0]

def tensor_basis(dim=2):
    """Generate dxd matrices with all zeros except
    for one element, such that the sum of all the
    matrices is a matrix full of ones.

    :dim: int
    :returns: list of dim x dim matrix

    """
    basis=[]
    for i in range(dim):
        for j in range(dim):
            basismat=np.zeros((dim,dim))
            basismat[i,j]=1
            basis.append(basismat)

    return basis
