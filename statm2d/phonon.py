from misc import *
import numpy as np
import symmetry as sym
import structure

def is_equivalent_cluster(site0,site1,comp0,comp1,lattice):
    """Check if two pairs of sites are equivalent
    and if the sites are swapped.

    :site0: Site
    :site1: Site
    :comp0: Site
    :comp1: Site
    :returns: bool,bool

    """
    swithin0,swithin1=structure.bring_cluster_within(site0,site1,lattice)
    cwithin0,cwithin1=structure.bring_cluster_within(comp0,comp1,lattice)

    sitedelta,sitemap=structure.site_delta(swithin0,swithin1)
    compdelta,compmap=structure.site_delta(cwithin0,cwithin1)

    goodmap=False
    mapswitch=False
    
    if np.allclose(sitedelta,compdelta):
        goodmap=True

    #if the deltas have opposite signs, the sites map by translation, but they switched places
    elif np.allclose(-sitedelta,compdelta) and sitemap and compmap:
        goodmap=True
        mapswitch=True

    return goodmap,mapswitch 

def map_cluster(site0,site1,op,lattice):
    """Checks if the symmetry operation
    maps a cluster onto itself (with a lattice translation).
    If it does, check if the sites switched places.

    :site0: Site
    :site1: Site
    :op: Op
    :returns: Site,Site,bool,bool

    """
    transsite0=site0.apply_symmetry(op)
    transsite1=site1.apply_symmetry(op)

    #Mapping the clusters should keep the pivot in the same spot
    clustshift,_=structure.site_delta(transsite0,site0)
    transsite0._coord+=clustshift
    transsite1._coord+=clustshift

    goodmap,mapswitch=is_equivalent_cluster(site0,site1,transsite0,transsite1,lattice)
    
    #if np.allclose(clusterdelta,transdelta):
    #    goodmap=True

    ##if the deltas have opposite signs, the sites map by translation, but they switched places
    #elif np.allclose(-clusterdelta,transdelta) and basismap:
    #    goodmap=True
    #    mapswitch=True

    return transsite0,transsite1,goodmap,mapswitch

def cluster_subgroups(site0,site1,symgroup,lattice):
    """Determine the symmetry operations of the given
    symgroup that leave the pair cluster unchanged (with
    a lattice translation). Check if the translation to map
    back involves sites switching places.
    The first list of operations mapped the sites
    back onto themselves, while the second list
    mapped the sites onto each other.

    :site0: Site
    :site1: Site
    :symgroup: list of Op
    :returns: two lists of Op

    """
    subgroupmap=[]
    subgroupswitch=[]
    
    for op in symgroup:
        transsite0,transsite1,goodmap,mapswitch=map_cluster(site0,site1,op,lattice)

        #if the deltas remain the same, then the sites map onto themselves by translation
        if goodmap and not mapswitch:
            subgroupmap.append(op)

        #if the deltas have opposite signs, the sites map by translation, but they switched places
        elif goodmap and mapswitch:
            subgroupswitch.append(op)

        else:
            continue

    return (subgroupmap,subgroupswitch)


def tensor_basis_for_pair(site0,site1,symgroup,lattice):
    """For a particular pair of sites, determine the symmetrized
    tensor basis. For each tensor basis apply the Reynolds operator
    with operations that map the cluster onto itself. In addition,
    apply the Reynolds operator to the transpose of the tensor basis
    for operations that map sites of the cluster onto each other.
    Return ALL basis, regardless of linear independence.

    :site0: Site
    :site1: Site
    :returns: list of 2x2 matrix

    """
    basiscandidates=tensor_basis()
    mapgroup,switchgroup=cluster_subgroups(site0,site1,symgroup,lattice)

    for idx, candidate in enumerate(basiscandidates):
        symmetrized=sym.matrix_reynolds(candidate,mapgroup)
        extrasymmetrized=sym.matrix_reynolds(candidate.T,switchgroup)
        basiscandidates[idx]=(symmetrized+extrasymmetrized)/(len(mapgroup)+len(switchgroup))
    return basiscandidates

def unique_force_tensor_basis_for_pair(site0,site1,symgroup,lattice,constants):
    """For a particular pair of sites, determine the symmetrized
    tensor basis. For each tensor basis apply the Reynolds operator
    with operations that map the cluster onto itself. In addition,
    apply the Reynolds operator to the transpose of the tensor basis
    for operations that map sites of the cluster onto each other.
    Return only the linearly independent entries, tupled with the corresponding
    force constants.

    :site0: Site
    :site1: Site
    :returns: list of (2x2 matrix,float)

    """
    basiscandidates=tensor_basis_for_pair(site0,site1,symgroup,lattice)
    uniqueind=independent_indices(basiscandidates)

    uniquebasis=[]
    for ind in uniqueind:
        uniquebasis.append((basiscandidates[ind],constants[ind]))

    return uniquebasis

def equivalent_clusters(site0,site1,symgroup,lattice):
    """Apply the given group of symmetry operations onto
    the given pair cluster, and find new symmetrically
    equivalent clusters. Make list of resulting clusters
    and list of operations that resulted in a new cluster.

    :site0: Site
    :site1: Site
    :symgroup: list of Op
    :returns: list of Site,Site,Op

    """
    equivsite0=[]
    equivsite1=[]
    mapsym=[]

    for op in symgroup:
        transsite0,transsite1,goodmap,mapswitch=map_cluster(site0,site1,op,lattice)

        newclust=True
        for eq0,eq1 in zip(equivsite0,equivsite1):
            goodmap,mapswitch=is_equivalent_cluster(transsite0,transsite1,eq0,eq1,lattice)
            #if goodmap:    Maybe translational equivalent ones should be discarded?
            if goodmap and not mapswitch:
                newclust=False
                break

        if newclust:
            equivsite0.append(transsite0)
            equivsite1.append(transsite1)
            mapsym.append(op)

    return equivsite0,equivsite1,mapsym

def flatten_tensor_stack(forcebasis):
    """Add up the tensor basis into a matrix,
    multiplying each element by the corresponding
    force constant. Expects only the unique
    tensor basis, tupled with the force constants.

    :basis: list of 2x2 matrix
    :constants: list of float
    :returns: 2x2 matrix

    """
    flattened=np.zeros((2,2))
    flattened=np.asmatrix(flattened)

    for basis,constant in forcebasis:
        flattened+=constant*basis

    return flattened

def dynamical_exp_elem(k,rn,tb0,tb1):
    """Compute the exponential part of the dynamical
    matrix.

    exp(-ik(rn-tb0+tb1))

    :k: 2x1 matrix (cart)
    :rn: 2x1 matrix
    :tb0: 2x1 matrix
    :tb1: 2x1 matrix
    :returns: complex

    """
    return cmath.exp(dot(-k*1j,(rn-tb0+tb1)))

def dynamical_exp_inputs(site0,site1,lattice):
    """Determine rn, tb0 and tb1 needed
    for the exponential term of the dynamical
    matrix calculations for the given cluster.
    Pivot always goes as first site.

    :site0: Site
    :site1: Site
    :lattice: 2x2 matrix with a and b vectors as columns
    :returns: (2x1 vector,2x1 vector, 2x1 vector)

    """
    rn,_=structure.site_delta(site0,site1)
    tb0=structure.bring_within(site0._coord,lattice)
    tb1=structure.bring_within(site1._coord,lattice)

    return rn,tb0,tb1

def connect_clusters(pivot,protosites,sgroup,lattice):
    """Get all the clusters that can be connected to the
    prototype sites, by making prototype pairs and applying
    symmetry to each one.

    :pivot: Site, shared by every pair
    :protosites: Site, needed to make prototype pairs
    :sgroup: list of Op
    :lattice: 2x2 matrix with a and b vectors as columns
    :returns: list of (Site,Site)

    """
    allpairs=[]
    for site in protosites:
        equiv0,equiv1,syms=equivalent_clusters(pivot,site,sgroup,lattice)
        allpairs+=zip(equiv0,equiv1)

    return allpairs

def self_interactions(basisstacks):
    """Find a new entry for the list of stacked tensor basis
    by summing the values. Expects list of stacks that share the
    same pivot, all corresponding to equivalent pair clusters.

    :basisstacks: [ ( [(2x2 mat, float)] ,(Site,Site) ) ]
    :returns: ([(2x2 mat, float)],(Site,Site))

    """
    zerosarr=np.zeros((2,2))

    selfpair=basisstacks[0][1][0]
    selfpair=(selfpair,selfpair)

    selfterms=[]
    for elem in basisstacks[0][0]:
        selfterms.append(np.matrix.copy(zerosarr))

    selfcoeffs=[c for basis,c in basisstacks[0][0]]


    for forcestack,pair in basisstacks:
        assert pair[0]==selfpair[0]

        for ind,basisconst in enumerate(forcestack):
            basis,const=basisconst
            assert const==selfcoeffs[ind]

            selfterms[ind]+=basis

    return zip(selfterms,selfcoeffs),selfpair

def dynamical_basis_entries(protopairs,sgroup,lattice,constants):
    """Compute the tensor basis parts of the dynamical
    matrix, given a list of all the necessary prototype pairs.
    The result is a list of 2x2 matrix which needs to be summed
    in a particular manner. 

    :protopairs: [(Site,Site)]
    :sgroup: list of Op
    :lattice: 2x2 matrix with a and b vectors as columns
    :constants: list of float (force values)
    :returns: [ ( [(2x2 mat, float)] ,(Site,Site) ) ]

    """
    allbasisstacks=[]
    for site0,site1 in protopairs:
        equiv0,equiv1,syms=equivalent_clusters(site0,site1,sgroup,lattice)
        tensorbasis=unique_force_tensor_basis_for_pair(site0,site1,sgroup,lattice,constants)

        protostacks=[]

        for eq0,eq1,op in zip(equiv0,equiv1,syms):
            pair=(eq0,eq1)
            newbasis=[(op.apply(basis),const) for basis,const in tensorbasis]

            #There will be an entry for every cluster
            protostacks.append((newbasis,pair))

        allbasisstacks.append(self_interactions(protostacks))
        allbasisstacks+=protostacks

    return allbasisstacks

def dynamical_pair_location(site0,site1,struc):
    """Run through the tensor basis entries of the dynamical
    matrix, and determine where each flattened stack should
    be added based on the basis sites of the corresponding
    pairs.
    Pivot always goes first.

    :site0: Site
    :site1: Site
    :return: [(slice,slice)]

    """
    b0ind=struc.find(site0)
    b1ind=struc.find(site1)

    return slice(2*b0ind,2*b0ind+2),slice(2*b1ind,2*b1ind+2)

def dynamical_matrix(struc,protopairs,fconstants,k):
    """Calculate the dynamical matrix for a given k point.

    :struc: Crystal
    :protopairs: [(Site,Site)], (probably NN pairs)
    :fconstants: list of 4 floats, for each xx,xy,yx,yy forces
    :k: 2x1 matrix, k-point of interest
    :returns: 2x2 matrix
    """
    D=np.zeros((2*len(struc._basis),2*len(struc._basis)),dtype=complex)

    sg=struc.factor_group()
    dynbasisentries=dynamical_basis_entries(protopairs,sg,struc._lattice,fconstants)

    for stack,pair in dynbasisentries:
        #This is the 2x2 sum of the tensor basis with the force constants
        L=flatten_tensor_stack(stack)

        #This is the exponential part
        rn,tb0,tb1=dynamical_exp_inputs(pair[0],pair[1],struc._lattice)
        expstuff=dynamical_exp_elem(k,rn,tb0,tb1)

        sumvalue=L*expstuff
        D_entry=dynamical_pair_location(pair[0],pair[1],struc)

        D[D_entry]+=sumvalue

    return np.asmatrix(D)

def ksegment(astar,bstar,startcoord,endcoord,density):
    """Get a list of kpoints between the given
    coordinates in a particular density

    :astar: 2x1 vector
    :bstar: 2x1 vector
    :startcoord: 2x1 vector
    :endcoord: 2x1 vector
    :density: list of int
    :returns: list of 2x1 vector 

    """
    delta=endcoord-startcoord
    increment=delta/density

    segment=[i*increment+startcoord for i in range(density)]
    return segment


def kpath(astar,bstar,sympoints,densities):
    """Find list of k-points along a given path, by connecting
    the points.

    :astar: 2x1 vector
    :bstar: 2x1 vector
    :sympoints: list 2x1 vector
    :densities: list of int
    :returns: list of 2x1 vector 

    """
    kpoints=[]

    for start,end,den in zip(sympoints,sympoints[1::]+sympoints[0:1],densities):
        kpoints+=ksegment(astar,bstar,start,end,den)

    return kpoints
