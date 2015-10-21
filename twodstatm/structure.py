import symmetry as sym
import numpy as np

class Crystal(object):

    """A 2x2 lattice with column vectors and a
    list of basis sites"""

    def __init__(self,a,b,atomlist):
        """Set values for the lattice and the basis list

        :a: 2x1 lattice vector
        :b: 2x1 lattice vector
        :atomlist: list of 2x1 vector with string as type tuple

        """
        self.lattice=np.hstack([a,b])
        self.basis = atomlist
        
    def point_group(self,maxsearch=3):
        """Construct list of symmetry operations that map
        the lattice onto itself. All shift vectors for
        these operations will be zero.

        :maxsearch: int 
        :returns: list of SymOp

        """
        symmat=sym.point_group_operations(self.lattice,maxsearch)
        tau=np.matrix([[0],[0]])

        pgroup=[]
        for mat in symmat:
            toperation=sym.SymOp(mat,tau)
            if toperation not in pgroup:    #not sure if necessary but w/e
                pgroup.append(toperation)

        return pgroup
