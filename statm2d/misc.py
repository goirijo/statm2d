import numpy as np
import math
import cmath

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

def matrix_tex(mat,indent,expand=4):
    """Generate string for matrix to render in LaTeX.

    begin{pmatrix}
        a&b
        c&d
    end{pmatrix}

    :mat: 2x2 matrix
    :indent: int, indentation level (exoand spaces per value)
    :expand: int, how many spaces per indentation
    :returns: string

    """
    texstring=((indent)*expand)*" "
    texstring+="\\begin{pmatrix}\n"+((indent+1)*expand)*" "
    texstring+=str(mat[0,0])+"&"+str(mat[0,1])+"\\\\\n"+((indent+1)*expand)*" "
    texstring+=str(mat[1,0])+"&"+str(mat[1,1])+"\n"+(indent*expand)*" "
    texstring+="\\end{pmatrix}\n"+(indent*expand)*" "

    return texstring

def norm(v):
    """Calculate the norm of a vector

    :v: 2x1 vector
    :returns: float

    """
    return np.sqrt((v.T*v)[0,0])

def dot(a,b):
    """Dot product for matrix objects.
    How is this a thing I need to write?

    :a: 2x1 vector
    :b: 2x1 vector
    :returns: float

    """
    return np.sum(np.asarray(a)*np.asarray(b))

def voronoi_plot(ax,vor):
    for simplex in vor.ridge_vertices:
        simplex = np.asarray(simplex)
        if np.all(simplex >= 0):
            ax.plot(vor.vertices[simplex,0], vor.vertices[simplex,1], 'k-')

    #center = points.mean(axis=0)
    #for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
    #    simplex = np.asarray(simplex)
    #    if np.any(simplex < 0):
    #        i = simplex[simplex >= 0][0] # finite end Voronoi vertex
    #        t = points[pointidx[1]] - points[pointidx[0]] # tangent
    #        t /= np.linalg.norm(t)
    #        n = np.array([-t[1], t[0]]) # normal
    #        midpoint = points[pointidx].mean(axis=0)
    #        far_point = vor.vertices[i] + np.sign(np.dot(midpoint - center, n)) * n * 100
    #        plt.plot([vor.vertices[i,0], far_point[0]], [vor.vertices[i,1], far_point[1]], 'k--')
    #ax.plot(points[:,0], points[:,1], 'o')
    #ax.plot(vor.vertices[:,0], vor.vertices[:,1], '*')

    return ax
