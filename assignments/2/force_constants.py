import statm2d as sm2d
import numpy as np
import math
import matplotlib.pyplot as plt

#define triangular lattice
a=np.matrix([[1],[0]])
b=np.matrix([[-0.5],[math.sqrt(3)/2]])

triangluar=sm2d.Crystal(a,b,[],fracmode=False)
pgroup=triangluar.point_group()

pair0=sm2d.structure.Site(np.matrix([[0],[0]]),"A",triangluar._lattice,True)
pair1=sm2d.structure.Site(np.matrix([[1],[0]]),"A",triangluar._lattice,True)

print "The selected pair in the triangluar lattice is"
print pair0
print pair1
print ""

submap,subswitch=sm2d.phonon.cluster_subgroups(pair0,pair1,pgroup)
print "The following operations mapped the cluster sites onto themselves:"
for op in submap:
    print op
    print ""
print "The following operations mapped the cluster sites onto each other:"
for op in subswitch:
    print op
    print ""

tensorbasis=sm2d.phonon.force_tensor_basis_for_pair(pair0,pair1,pgroup)

basissum=np.zeros((2,2))
print "The invariant tensor basis is:"
for basis in tensorbasis:
    print basis
    print ""
    basissum+=basis

print "The linearly independent basis are "+str(sm2d.misc.independent_indices(tensorbasis))

print "With R60 applied they become:"
basissumrot=np.zeros((2,2))
R60mat=pgroup[1].matrix
rotbasis=[]
for basis in tensorbasis:
    rotbasis.append(R60mat.T.dot(basis).dot(R60mat))
    print rotbasis[-1]
    print ""
    basissumrot+=rotbasis[-1]

print "The linearly independent basis are "+str(sm2d.misc.independent_indices(rotbasis))

print basissum
print ""
print basissumrot
print ""
