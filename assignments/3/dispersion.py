import statm2d as sm2d
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

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
pair3=sm2d.structure.Site(np.matrix([[2],[0]]),"A",triangluar._lattice,True)
pivot=pair0
#pair1=site2

print "The selected pair in the triangluar lattice is"
print pair0
print pair1
print ""

equiv0,equiv1,syms=sm2d.phonon.equivalent_clusters(pair0,pair1,pgroup,triangluar._lattice)

print "debug"
sm2d.phonon.dynamical_basis(pivot,[pair1],pgroup,triangluar._lattice)
exit()

print "The symmetrically equivalent clusters are"
for eq0,eq1,op in zip(equiv0,equiv1,syms):
    print eq0
    print eq1
    print op
    print ""

print "The invariant tensor basis for the selected pair is"
tensorbasis=sm2d.phonon.force_tensor_basis_for_pair(pair0,pair1,pgroup,triangluar._lattice)
uniqueind=sm2d.misc.independent_indices(tensorbasis)
uniquetensorbasis=[tensorbasis[i] for i in uniqueind]

coeffvals=[1,2,2,2]
coeffnames=["xx","xy","yx","yy"]
uniquecoeffnames=[coeffnames[i] for i in uniqueind]
uniquecoeffvals=[coeffvals[i] for i in uniqueind]

for basis,coeff in zip(uniquetensorbasis,uniquecoeffnames):
    print coeff
    print basis
    print ""

print "We pick force constants arbitrarily"
for name,val in zip(uniquecoeffnames,uniquecoeffvals):
    print name+"="+str(val)

print "The invariant force constant matrix for each pair becomes"
for eq0,eq1,op in zip(equiv0,equiv1,syms):
    print op.name
    print eq0
    print eq1

    for basis in uniquetensorbasis:
        print op.apply(basis)

    print ""

print "Combining the tensor basis together with the force constants we get"
for eq0,eq1,op in zip(equiv0,equiv1,syms):
    print op.name
    print eq0
    print eq1

    newbasis=[op.apply(basis) for basis in tensorbasis]

    print sm2d.phonon.stack_tensor_basis(newbasis,coeffvals)

    print ""

exit()

print "The real lattice is"
print a
print b
print "The reciprocal lattice is"
astar,bstar=sm2d.structure.reciprocal_lattice(a,b)
print astar
print bstar

reciprocal=sm2d.Crystal(astar,bstar,[site1],fracmode=False)
points=np.array(sm2d.structure.grid_points(astar,bstar,(-1,3),(-1,3)))
vor=Voronoi(points)

fig=plt.figure()
ax=fig.add_subplot('111')
ax.set_title(r"\bf{Reciprocal triangular}")
ax.set_xlabel(r"\bf{x}")
ax.set_ylabel(r"\bf{y}")
reciprocal.plot(ax,(-1,4),(-1,4))

sm2d.misc.voronoi_plot(ax,vor)

plt.show()
