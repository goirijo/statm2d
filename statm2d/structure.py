import symmetry as sym
from misc import norm
import numpy as np
import math
import matplotlib.patches as patches
import matplotlib.lines as lines
from scipy.spatial import Voronoi

def bring_within(coordinate,lattice):
    """Translate the given coordinate by integer
    amounts of lattice vectors so that the
    coordinate lands within the first image
    of the unit cell.

    :coordinate: 2x1 matrix (Cartesian coordinates)
    :lattice: 2x2 matrix with a and b vectors as columns
    :returns: 2x1 matrix

    """
    #turn the coordinate into fractional
    fraccoord=np.linalg.inv(lattice).dot(coordinate)

    #How many integer translations the coordinate is over by
    aexcess=math.floor(fraccoord[0,0])
    bexcess=math.floor(fraccoord[1,0])

    fracwithin=fraccoord-np.matrix([[aexcess],[bexcess]])

    #square off rounding errors
    closeindx=np.isclose(fracwithin,1)
    fracwithin[closeindx]=0.0
    closeindx=np.isclose(fracwithin,-1)
    fracwithin[closeindx]=0.0
    closeindx=np.isclose(fracwithin,0)
    fracwithin[closeindx]=0.0


    #Now fracwithin is just 0.xxxxx 0.yyyyyy
    #Transform back into cartesian
    return lattice.dot(fracwithin)

def is_valid(testsite,struc):
    """Check if the given site exists in the structure
    :returns: bool

    """
    return testsite.bring_within(struc._lattice) in struc._basis

def reciprocal_lattice(a,b):
    """Construct a reciprocal lattice from the given
    lattice vectors

    :a: 2x1 lattice vector
    :b: 2x1 lattice vector
    :returns: vector,vector

    """
    real=np.hstack((a,b))
    recip=2*math.pi*np.linalg.inv(real).T

    astar=recip[:,0]
    bstar=recip[:,1]

    return astar,bstar

def grid_points(a,b,arepeat,brepeat):
    """Make a list of lattice points within
    the given range

    :a: 2x1 lattice vector
    :b: 2x1 lattice vector
    :arepeat: (int,int)
    :barepeat: (int,int)
    :returns: list of ndarray

    """
    gridpoints=[]
    for acount in np.arange(arepeat[0],arepeat[1]):
        for bcount in np.arange(brepeat[0],brepeat[1]):
            point=np.array((acount*a+bcount*b).T)
            point=np.squeeze(point)
            gridpoints.append(point)

    return gridpoints
    
def coord_split(coordinate,lattice):
    """Split the coordinate into integer a vectors,
    b vectors, x shift and y shift.

    :coordinate: 2x1 matrix (Cartesian)
    :lattice: 2x2 matrix with a and b vectors as columns
    :returns: int,int,float,float

    """
    pass

class Site(object):

    """Cartesian coordinate plus a string to
    determine the type of atom"""

    def __init__(self,coord,specie,lattice=[],fracmode=False):
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

    def copy(self):
        return Site(self._coord,self._specie,fracmode=False)

    def apply_symmetry(self, operation):
        """Apply a symmetry operation to the basis
        and return new version with new coordinates

        :operation: Op
        :returns: Site

        """
        newcoord=operation.matrix.dot(self._coord)+operation.shift
        return Site(newcoord,self._specie,fracmode=False)

    def bring_within(self, lattice):
        """Translate the site into the first image
        of the provided lattice

        :lattice: 2x2 matrix with a and b vectors as columns
        :returns: Site

        """
        return Site(bring_within(self._coord,lattice),self._specie)

    def __str__(self):
        coordstring="[%-*s,%s]" % (12,self._coord[0,0],self._coord[1,0])
        representation=self._specie+"    %s    " % (coordstring)
        return representation

    def __eq__(self,other):
        if self._specie!=other._specie:
            return False
        elif not np.allclose(self._coord,other._coord):
            return False
        else:
            return True
        
def site_delta(site0,site1):
    """Subtract the coordinates of two sites and
    return result as a coordinate and a bool, which
    determines whether the sites are of the same type
    or not

    :site0: Site
    :site1: Site
    :returns: (2x1 vector, bool)

    """
    delta=site1._coord-site0._coord
    samespecie=False
    if site0._specie==site1._specie:
        samespecie=True

    return (delta,samespecie)

def bring_cluster_within(site0,site1,lattice):
    """Bring the first site within the lattice
    and then translate the other one by the same
    amount

    :site0: Site
    :site1: Site
    :lattice: 2x2 matrix with a and b vectors as columns
    :returns: Site,Site

    """
    within0=site0.bring_within(lattice)
    withinshift,_=site_delta(site0,within0)

    within1=Site(site1._coord+withinshift,site1._specie)
    return within0,within1


class Crystal(object):

    """A 2x2 lattice with column vectors and a
    list of basis sites"""

    def __init__(self,a,b,atomlist,fracmode):
        """Set values for the lattice and the basis list

        :a: 2x1 lattice vector
        :b: 2x1 lattice vector
        :atomlist: list of coordinates (cartesian or fractional) and atom type tuple
        :fracmode: bool to know what atomlist is specified in

        """
        self._lattice=np.hstack((a,b))
        self._basis=[]

        for atom in atomlist:
            self._basis.append(atom.bring_within(self._lattice))
        

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
        fgroup=[]

        #First determine the point group
        pg=self.point_group(maxsearch)

        #the operations of the factor group will be a subset of
        #the point group with or without translation. Try
        #every operation in the point group
        for op in pg:
            #get a new set of basis coordinates
            transbasis=self.transformed_basis(op)

            #try mapping every transformed atom back onto an original basis atom
            for atom in self._basis:
                for transatom in transbasis:
                    #if you're trying to map the wrong type of atom don't even bother with the iteration
                    if atom._specie!=transatom._specie:
                        continue

                    #calculate the shift and bring it within a unit cell
                    tshift=bring_within(atom._coord-transatom._coord,self._lattice)

                    #apply the shift to every atom and see if the whole basis maps
                    goodmap=True
                    for mapped in transbasis:
                        backmap=Site(mapped._coord+tshift,mapped._specie)
                        backmap=backmap.bring_within(self._lattice)
                        if backmap not in self._basis:
                            goodmap=False

                    #If all the atoms mapped then the shift is good and we save the operation to the group
                    if goodmap:
                        factorop=sym.Op(op.matrix,tshift)
                        #Save the operation if it's not there already
                        if factorop not in fgroup:
                            fgroup.append(factorop)

        return fgroup

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
        
    def find(self, site):
        """Get the index of the provided site
        in terms of the basis

        :site: Site
        :returns: int

        """
        within=site.bring_within(self._lattice)
        return self._basis.index(within)

    def __str__(self):
        representation="Lattice:\n"+self._lattice.__str__()
        representation+="\nCoordinates\n"

        for site in self._basis:
            representation+=site.__str__()+"\n"

        return representation


