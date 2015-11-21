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
site2=sm2d.structure.Site(coord2,"B")

triangluar=sm2d.Crystal(a,b,[site1],fracmode=False)
honeycomb=sm2d.Crystal(a,b,[site1,site2],fracmode=False)
pgroup=triangluar.point_group()


pair0=sm2d.structure.Site(np.matrix([[0],[0]]),"A",triangluar._lattice,True)
pair1=sm2d.structure.Site(np.matrix([[1],[0]]),"A",triangluar._lattice,True)
pair3=sm2d.structure.Site(np.matrix([[2],[0]]),"A",triangluar._lattice,True)
pivot=pair0
#pair1=site2

#equiv0,equiv1,syms=sm2d.phonon.equivalent_clusters(pair0,pair1,pgroup,triangluar._lattice)

print "debug"
protopairs=[(site1,site2),(site2,site1)]
teststruc=honeycomb
sg=teststruc.factor_group()
testconst=[-1,-2,-3,-4]
dynbasisentries=sm2d.phonon.dynamical_basis_entries(protopairs,sg,teststruc._lattice,testconst)
for stack,pair in dynbasisentries:

    for basis,const in stack:
        print basis
        print const
        print "----------------"
    print "----------------"
    print sm2d.phonon.flatten_tensor_stack(stack)
    print "----------------"
    for site in pair:
        print site
    print "-------------------------------------------------------------------"

pairs=[p for stack,p in dynbasisentries]
dymlocs=[sm2d.phonon.dynamical_pair_location(s0,s1,teststruc) for s0,s1 in pairs]
expinputs=[sm2d.phonon.dynamical_exp_inputs(s0,s1,teststruc._lattice) for s0,s1 in pairs]

for ds,es in zip(dymlocs,expinputs):
    print ds
    for e in es:
        print e
    print "---"


print "The real lattice is"
print np.array((a,b)).T
print "The reciprocal lattice is"
astar,bstar=sm2d.structure.reciprocal_lattice(a,b)
print np.array((astar,bstar)).T

reciprocal=sm2d.Crystal(astar,bstar,[site1],fracmode=False)
points=np.array(sm2d.structure.grid_points(astar,bstar,(-1,3),(-1,3)))
vor=Voronoi(points)

fig=plt.figure(0)
ax=fig.add_subplot('111')
ax.set_title(r"\bf{Reciprocal triangular}")
ax.set_xlabel(r"\bf{x}")
ax.set_ylabel(r"\bf{y}")
reciprocal.plot(ax,(-1,4),(-1,4))

sm2d.misc.voronoi_plot(ax,vor)

G=0*astar+0*bstar
K=1.0/3*astar+1.0/3*bstar
M=0.5*astar
ax.scatter(G[0,0],G[1,0],color='r')
ax.scatter(K[0,0],K[1,0],color='g')
ax.scatter(M[0,0],M[1,0],color='y')
kpoints=sm2d.phonon.kpath(astar,bstar,[G,K,M],[30,10,20])

figk=plt.figure(1)
axk=figk.add_subplot('111')
axk.set_title(r"\bf{Honeycomb dispersion}")
axk.set_xlabel(r"\bf{k}")
axk.set_ylabel(r"$\omega$")
for ind,k in enumerate(kpoints):
    D=sm2d.phonon.dynamical_matrix(teststruc,protopairs,testconst,k)
    eigval,eigvec=np.linalg.eig(D) 

    kcount=np.ones(eigval.shape)*ind

    axk.scatter(kcount,eigval)

plt.show()
