import numpy as np
import math
from misc import *

def gridpoints(lattice,maxsearch):
    """Return a list of lattice points by extending
    the lattice maxsearch times in all directions.
    This is a sillier version than the smart way of
    enclosing the unit cell in a sphere, but w/e 
    this is easier
    
    :lattice: 2x2 ndarray with vectors as columns
    :maxsearch: How many times to count beyond the cell
    :returns: array of 2x1 gridpoints
    """
    avector=lattice[:,0]
    bvector=lattice[:,1]
    gridpoints=[]
    for xcount in np.arange(-maxsearch,maxsearch+1):
        for ycount in np.arange(-maxsearch,maxsearch+1):
            point=xcount*avector+ycount*bvector
            gridpoints.append(point)
    
    return np.array(gridpoints)

def point_group_operations(lattice,maxsearch):
    """Create a list of gridpoints by combining
    lattice vectors by factors up to maxsearch.
    Combine every point in the list
    of gridpoints and create a possible symmetry
    operation that maps the original lattice
    vectors to the pair of gridpoints.

    If the lattice is defined by two column vectors
    L=[a b]
    and the two gridpoints are
    L'=[a' b']
    then the symmetry matrix is
    S=L'*inv(L)

    The matrix S is automatically discarded if
    its determinant is anything other than unity
    or the matrix is not unitary.

    :lattice: 2x2 matrix
    :gridpointlist: array of 2x1 vectors
    :returns: array of 2x2 matrix

    """
    gridpointlist=gridpoints(lattice,maxsearch)
    candidatelist=[]
    for point1 in gridpointlist:
        for point2 in gridpointlist:
            maplat=np.hstack((point1,point2))   #this is L' now
            S=maplat*np.linalg.inv(lattice)

            #if the determinant is anything other than unity, volume is
            #not preserved and the candidate is no good
            Sdet=np.linalg.det(S)
            if not is_close(abs(Sdet),1.0):
                continue
            elif not is_unitary(S):
                continue
            else:
                candidatelist.append(S)

    return np.array(candidatelist)

def operation_type(mat):
    """Given a symmetry matrix, return
    a name for the operation by calculating
    the angle of rotation or mirror plane
    and the type of operation.

    R = rotaton
    M = reflection
    I = identity

    For R and M, the angle will be given in
    degrees.

    :mat: symmetry matrix
    :returns: (string, int) for type and angle

    """
    det=np.linalg.det(mat)

    #Find out what angles are involved in the operation
    refvec=np.matrix([[1],[0]])
    transvec=mat*refvec
    angle=(360+int(round(vec_angle(refvec,transvec))))%360

    if np.allclose(mat,np.identity(2)):
        typetup=("I",0)
    #If the determinant is 1 we're dealing with rotation
    elif is_close(det,1):
        typetup=("R",angle)
    #if the determinant is -1 it's a mirror
    elif is_close(det,-1):
        typetup=("M",angle/2)
    #if the determinant is something else, you done goofed
    else:
        raise ValueError("The determinant of your matrix was not unity and therefore\
        an invalid point group operation.")

    return typetup

def is_closed(symgroup):
    """Determine if the provided list of symmetry operations
    forms a closed group or not by multiplying them all
    by each other and seeing if a new operation appears or not.

    :symgroup: list of Op
    :returns: bool

    """
    
    for op1 in symgroup:
        for op2 in symgroup:
            testop=op1*op2
            if testop not in symgroup:
                return False

    return True

def matrix_reynolds(matrix,symgroup):
    """Apply the Reynolds operator to the given matrix
    by adding the result of applying every operation 
    of the given group

    Add every S.T*M*S

    :matrix: 2x2 matrix
    :symgroup: list of Op
    :returns: 2x2 matrix

    """
    
    operated=np.zeros(matrix.shape)
    for op in symgroup:
        opmat=op.matrix
        operated+=(opmat.T.dot(matrix)).dot(opmat)

    return operated

class Op(object):

    """A symmetry operation (I, R or M) with a translation vector and
    a label to go with it. Everything is Cartesian."""

    def __init__(self, matrix, shift):
        """Set the matrix and shift vector shift and figure out the label
        based on the matrix

        :matrix: 2x2 operation
        :shift: 2x1 vector
        """

        self.matrix=matrix
        self.shift=shift

        #Label for the operation, e.g. I, R30, M240, etc.
        typetup=operation_type(matrix)
        self.name=typetup[0]+str(typetup[1])

        #Rank for sorting
        if self.name[0]=="I":
            self._rank=0
        if self.name[0]=="R":
            self._rank=1000+int(self.name[1::])
        elif self.name[0]=="M":
            self._rank=2000+int(self.name[1::])
    
    def __eq__(self, other):
        if not np.allclose(self.matrix,other.matrix):
            return False
        elif not np.allclose(self.shift,other.shift):
            return False
        else:
            return True

    def __lt__(self, other):
        return self._rank<other._rank

    def __repr__(self):
        representation="name: "+self.name
        representation+="\n"+self.matrix.__repr__()
        representation+="\n"+self.shift.__repr__()
        return representation

    def __str__(self):
        np.set_printoptions(suppress=True)
        representation=self.name
        representation+="\n"+self.matrix.__str__()
        representation+="\nshift: [%s,%s]" % (self.shift[0,0],self.shift[1,0])
        return representation

    def __mul__(self, other):
        """Combine two operations into one.

        S*v=S.mat*v+S.tau

        S1*S2=S1.mat*S2.mat+S1.mat*S2.tau+S1.tau

        :other: RHS symmetry operation
        :returns: Op

        """
        newmat=self.matrix.dot(other.matrix)
        newshift=self.matrix.dot(other.shift)+self.shift
        return Op(newmat,newshift)

    def tex_formula(self):
        """Return a string that prints the name of the
        operation, the matrix and the shift vector in
        LaTeX format.

        :returns: string

        """
        texstring=self.name+"\n\\begin{equation}\n    S=\n"
        texstring+=matrix_tex(self.matrix,1)
        texstring+=",\\vec{\\tau}=\n    "
        texstring+="\\begin{pmatrix}\n        "
        texstring+=str(self.shift[0,0])+"\\\\\n        "
        texstring+=str(self.shift[1,0])+"\n    "
        texstring+="\\end{pmatrix}\n    "
        texstring+="\\label{"+self.name+"}\n\\end{equation}"

        return texstring
