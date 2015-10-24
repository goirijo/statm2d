import symmetry as sym
import numpy as np
import matplotlib.patches as patches
import matplotlib.lines as lines

class Site(object):

    """Contains a fractional and a cartesian
    coordinate, which requires initialization
    with a lattice. There is also a string that
    identifies what type of atom is on the site.
    Expects fractional coordinates as default"""

    def __init__(self,coord,specie,lattice=[],fracmode=True):
        """Given a fractional or cartesian coordinate,
        construct a cartesian or fractional coordinate
        and save both along with the species.

        :lattice: 2x2 matrix with a and b vectors as columns
        :coordinates: 2x1 vector for atomic position
        :specie: string
        :fracmode: bool

        """
        self._specie = specie

        if fracmode:
            try:
                self._coord=lattice.dot(coord)
            except:
                raise TypeError("Tried to convert from fractional to Cartesian but have no lattice to do so")

        else:
            self._coord=coord

    def apply_symmetry(self, operation):
        """Apply a symmetry operation to the basis
        and return new version with new coordinates

        :operation: Op
        :returns: Site

        """
        newcoord=operation.matrix.dot(self._coord)+operation.shift
        return Site(newcoord,self._specie,fracmode=False)
        

    def __str__(self):
        coordstring="[%-*s,%s]" % (12,self._coord[0,0],self._coord[1,0])
        representation=self._specie+"    %s    " % (coordstring)
        return representation

        

class Crystal(object):

    """A 2x2 lattice with column vectors and a
    list of basis sites"""

    def __init__(self,a,b,atomlist,fracmode=True):
        """Set values for the lattice and the basis list

        :a: 2x1 lattice vector
        :b: 2x1 lattice vector
        :atomlist: list of coordinates (cartesian or fractional) and atom type tuple
        :fracmode: bool to know what atomlist is specified in

        """
        self._lattice=np.hstack((a,b))
        self._basis=[]

        for atom in atomlist:
            self._basis.append(atom)
        

    def a(self):
        return self._lattice[:,0]

    def b(self):
        return self._lattice[:,1]

    def transformed_basis(self, operation):
        """Apply a symmetry operation to every atom
        in the basis and return transformed coordinates
        as list of sites

        :operation: Op 
        :returns: list of Site

        """
        newbasis=[]
        for atom in self._basis:
            newbasis.append(atom.apply_symmetry(operation))

        return newbasis

    def point_group(self,maxsearch=3):
        """Construct list of symmetry operations that map
        the lattice onto itself. All shift vectors for
        these operations will be zero.

        :maxsearch: int 
        :returns: list of Op

        """
        symmat=sym.point_group_operations(self._lattice,maxsearch)
        tau=np.matrix([[0],[0]])

        pgroup=[]
        for mat in symmat:
            toperation=sym.Op(mat,tau)
            if toperation not in pgroup:    #not sure if necessary but w/e
                pgroup.append(toperation)

        return sorted(pgroup)

    def factor_group(self, maxsearch=3):
        """Find operations that map the basis atoms onto themselves.
        Begin with the point group and then make candidate shifts
        that map the atoms back to the right position, keeping
        only the operations that work. All shifts are within the
        unit cell.

        :maxsearch: int
        :returns: list of Op

        """
        #First determine the point group
        pg=self.point_group(maxsearch)

        return

    def plot(self, ax, arepeat, brepeat,colorshift=0):
        """Visualize structure by scattering
        basis on a plot. Specify how far to
        repeat the unit cell.

        :figure: pyplot axes
        :arepeat: (int,int)
        :barepeat: (int,int)
        :returns: axes

        """

        #If you have more types than there are colors come back
        #and fix this routine
        colorbook={0:"blue",1:"green",2:"red",3:"cyan",4:"magenta",5:"yellow"}
        #assign a color to each basis type
        atomtypes=[]
        for atom in self._basis:
            typename=atom._specie
            if typename not in atomtypes:
                atomtypes.append(typename)


        a=self.a()
        b=self.b()

        axval=a[0,0]
        ayval=a[1,0]
        bxval=b[0,0]
        byval=b[1,0]

        for acount in np.arange(arepeat[0],arepeat[1]):
            for bcount in np.arange(brepeat[0],brepeat[1]):
                for atom in self._basis:
                    colorindex=atomtypes.index(atom._specie)
                    coords=atom._coord+acount*a+bcount*b
                    ax.scatter(coords[0],coords[1],s=30,color=colorbook[colorindex+colorshift])
                    
                #plot dashed lines that extend the a vector, but only once
                if acount==(arepeat[1]-1):
                    astart=bcount*b+arepeat[0]*a
                    aend=bcount*b+arepeat[1]*a

                    l=lines.Line2D((astart[0,0],aend[0,0]),(astart[1,0],aend[1,0]),linestyle="--",alpha=0.5)
                    ax.add_line(l)

            #plot dashed lines that extend the b vector
            bstart=acount*a+brepeat[0]*b
            bend=acount*a+brepeat[1]*b
            
            l=lines.Line2D((bstart[0,0],bend[0,0]),(bstart[1,0],bend[1,0]),linestyle="--",alpha=0.5)
            ax.add_line(l)

        ax.set_aspect("equal")
        
        ax.add_patch(patches.FancyArrow(0,0,axval,ayval,length_includes_head=True,width=0.005,color="black"))
        ax.add_patch(patches.FancyArrow(0,0,bxval,byval,length_includes_head=True,width=0.005,color="black"))
        
        return ax
        

    def __str__(self):
        representation="Lattice:\n"+self._lattice.__str__()
        representation+="\nCoordinates\n"

        for site in self._basis:
            representation+=site.__str__()+"\n"

        return representation


