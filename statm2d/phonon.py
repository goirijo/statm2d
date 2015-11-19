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


def force_tensor_basis_for_pair(site0,site1,symgroup,lattice):
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

def stack_tensor_basis(basis,constants):
    """Add up the tensor basis into a matrix,
    multiplying each element by the corresponding
    force constant.
    Only adds unique basis.

    :basis: list of 2x2 matrix
    :constants: list of float
    :returns: 2x2 matrix

    """
    uniqueind=independent_indices(basis)

    stacked=np.zeros((2,2))
    stacked=np.asmatrix(stacked)

    for ind in uniqueind:
        stacked+=constants[ind]*basis[ind]

    return stacked

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
    return math.exp(-k*1j*(rn-tb0+tb1))

def connect_clusters(pivot,protosites,pgroup,lattice):
    """Get all the clusters that can be connected to the
    prototype sites, by making prototype pairs and applying
    symmetry to each one.

    :pivot: Site, shared by every pair
    :protosites: Site, needed to make prototype pairs
    :pgroup: list of Op
    :lattice: 2x2 matrix with a and b vectors as columns
    :returns: list of (Site,Site)

    """
    allpairs=[]
    for site in protosites:
        equiv0,equiv1,syms=equivalent_clusters(pivot,site,pgroup,lattice)
        allpairs+=zip(equiv0,equiv1)

    return allpairs


def dynamical_basis_entries(protopairs,pgroup,lattice,constants):
    """Compute the tensor basis parts of the dynamical
    matrix, given a list of all the necessary prototype pairs.
    The result is a list of 2x2 matrix which needs to be summed
    in a particular manner.
    There is no self interaction term!!

    :protopairs: (Site,Site) 
    :pgroup: list of Op
    :lattice: 2x2 matrix with a and b vectors as columns
    :constants: list of float (force values)
    :returns: 2x2 matrix

    """
    allbasisstacks=[]
    for site0,site1 in protopairs:
        equiv0,equiv1,syms=equivalent_clusters(site0,site1,point_group,lattice)
        tensorbasis=force_tensor_basis_for_pair(site0,site1,pgroup,lattice)

        for eq0,eq1,op in zip(equiv0,equiv1,syms):
            pair=(eq0,eq1)
            newbasis=[op.apply(basis) for basis in tensorbasis]
            newstack=stack_tensor_basis(newbasis,constants)

            #There will be an entry for every cluster
            allbasisstacks.append(newstack,pair)
    pass

