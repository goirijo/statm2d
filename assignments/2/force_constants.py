import statm2d as sm2d
import numpy as np
import math
import matplotlib.pyplot as plt

#define triangular lattice
a=np.matrix([[1],[0]])
b=np.matrix([[-0.5],[math.sqrt(3)/2]])
coord1=np.matrix([0,0]).T
coord2=(2*a+b)/3
site1=sm2d.structure.Site(coord1,"A")
site2=sm2d.structure.Site(coord2,"A")

triangluar=sm2d.Crystal(a,b,[site1],fracmode=False)
honeycomb=sm2d.Crystal(a,b,[site1,site2],fracmode=False)
pgroup=triangluar.point_group()

pair0=sm2d.structure.Site(np.matrix([[0],[0]]),"A",triangluar._lattice,True)
pair1=sm2d.structure.Site(np.matrix([[1],[0]]),"A",triangluar._lattice,True)
#pair1=site2

print "The selected pair in the triangluar lattice is"
print pair0
print pair1
print ""

fig=plt.figure()
ax=fig.add_subplot('111')
ax.set_title(r"\bf{Triangular lattice pair cluster}")
ax.set_xlabel(r"\bf{x}")
ax.set_ylabel(r"\bf{y}")
triangluar.plot(ax,(-1,4),(-1,4))
ax.scatter(pair0._coord[0],pair0._coord[1],color="r",s=70)
ax.scatter(pair1._coord[0],pair1._coord[1],color="r",s=70)
plt.show()

submap,subswitch=sm2d.phonon.cluster_subgroups(pair0,pair1,pgroup,triangluar._lattice)
print "The following operations mapped the cluster sites onto themselves:"
for op in submap:
    print op
    #print op.tex_formula()
    print ""
print "The following operations mapped the cluster sites onto each other:"
for op in subswitch:
    print op
    print ""

tensorbasis=sm2d.phonon.tensor_basis_for_pair(pair0,pair1,pgroup,triangluar._lattice)

print "The invariant tensor basis is:"
for basis in tensorbasis:
    print basis
    print ""
    #print "\\begin{equation}"
    #print "    \\Lambda="
    #print sm2d.misc.matrix_tex(basis,1)
    #print "    \\label{symbasis}"
    #print "\\end{equation}"
    #print ""

print "The linearly independent basis are "+str(sm2d.misc.independent_indices(tensorbasis))

pair0=site1
pair1=site2

print "The selected pair in the honeycomb structure is"
print pair0
print pair1
print ""

fig=plt.figure()
ax=fig.add_subplot('111')
ax.set_title(r"\bf{Honeycomb pair cluster}")
ax.set_xlabel(r"\bf{x}")
ax.set_ylabel(r"\bf{y}")
honeycomb.plot(ax,(-1,4),(-1,4))
ax.scatter(pair0._coord[0],pair0._coord[1],color="r",s=70)
ax.scatter(pair1._coord[0],pair1._coord[1],color="r",s=70)
plt.show()

submap,subswitch=sm2d.phonon.cluster_subgroups(pair0,pair1,pgroup,honeycomb._lattice)
print "The following operations mapped the cluster sites onto themselves:"
for op in submap:
    print op
    print ""
print "The following operations mapped the cluster sites onto each other:"
for op in subswitch:
    print op
    print ""

tensorbasis=sm2d.phonon.tensor_basis_for_pair(pair0,pair1,pgroup,honeycomb._lattice)

print "The invariant tensor basis is:"
for basis in tensorbasis:
    print basis
    print ""

print "The linearly independent basis are "+str(sm2d.misc.independent_indices(tensorbasis))
