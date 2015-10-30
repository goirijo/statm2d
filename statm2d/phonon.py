from misc import *
import numpy as np
import symmetry as sym
import structure

def cluster_subgroups(site0,site1,symgroup):
    """Determine the symmetry operations of the given
    symgroup that leave the cluster unchanged (with
    a translation). Check if the translation to map
    back involves sites switching places.
    The first list of operations mapped the sites
    back onto themselves, while the second list
    mapped the sites onto each other.

    :site0: Site
    :site1: Site
    :returns: two lists of Op

    """
    subgroupmap=[]
    subgroupswitch=[]
    clusterdelta,basismap=structure.site_delta(site0,site1)
    for op in symgroup:
        transsite0=site0.apply_symmetry(op)
        transsite1=site1.apply_symmetry(op)
        transdelta,transmap=structure.site_delta(transsite0,transsite1)

        #if the deltas remain the same, then the sites map onto themselves by translation
        if np.allclose(clusterdelta,transdelta):
            op.shift=site0._coord-transsite0._coord
            subgroupmap.append(op)

        #if the deltas have opposite signs, the sites map by translation, but they switched places
        elif np.allclose(-clusterdelta,transdelta) and basismap:
            op.shift=site1._coord-transsite0._coord
            subgroupswitch.append(op)

        else:
            continue

    return (subgroupmap,subgroupswitch)


def force_tensor_basis_for_pair(site0,site1,symgroup):
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
    mapgroup,switchgroup=cluster_subgroups(site0,site1,symgroup)

    for idx, candidate in enumerate(basiscandidates):
        symmetrized=sym.matrix_reynolds(candidate,mapgroup)
        extrasymmetrized=sym.matrix_reynolds(candidate.T,switchgroup)
        basiscandidates[idx]=(symmetrized+extrasymmetrized)/(len(mapgroup)+len(switchgroup))
    return basiscandidates
