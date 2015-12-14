import statm2d as sm2d
import numpy as np
import math
import cmath
from statm2d.misc import *
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

print "The selected pair in the triangluar lattice is"
print pair0
print pair1
print ""

#equiv0,equiv1,syms=sm2d.phonon.equivalent_clusters(pair0,pair1,pgroup,triangluar._lattice)

print "debug"
protopairs=[(pair0,pair1)]
teststruc=triangluar
sg=teststruc.factor_group()
testconst=[-2,-200,-300,-5]
dynbasisentries=sm2d.phonon.dynamical_basis_entries(protopairs,sg,teststruc,testconst)

pairs=[p for stack,p in dynbasisentries]
dymlocs=[sm2d.phonon.dynamical_pair_location(s0,s1,teststruc) for s0,s1 in pairs]
expinputs=[sm2d.phonon.dynamical_exp_inputs(s0,s1,teststruc._lattice) for s0,s1 in pairs]

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

###################3
G=0*astar+0*bstar
K=1.0/3*astar+1.0/3*bstar
M=0.5*astar
kpoints=sm2d.phonon.kpath(astar,bstar,[K,G,M],[60,60,20])
kpoints=sm2d.phonon.kpath(astar,bstar,[K,G,M],[2,2,2])

figk=plt.figure(1)
axk=figk.add_subplot('111')
axk.set_title(r"\bf{Triangluar dispersion}")
axk.set_xlabel(r"\bf{k}")
axk.set_ylabel(r"$\omega$")

cummulation=0
symticks=[]
for segment in kpoints:
    symticks.append(cummulation)
    minstep=np.linalg.norm(segment[0]-segment[1])
    print minstep
    for k in segment:
        D=sm2d.phonon.dynamical_matrix(teststruc,protopairs,testconst,k)
        eigval,eigvec=np.linalg.eig(D) 
        print eigval

        assert np.allclose(np.zeros(eigval.shape),eigval.imag)
        eigval=eigval.real

        cummulation+=minstep
        axk.scatter(np.full(eigval.shape,cummulation),np.sqrt(eigval))
symticks.append(cummulation)

axk.set_xticks(symticks)
axk.set_xticklabels([r"\textbf{K}",r"$\Gamma$",r"\textbf{M}",r"\textbf{K}"])

plt.show()

###################3

fig=plt.figure()
ax=fig.add_subplot('111')
ax.set_title(r"\bf{Reciprocal triangular}")
ax.set_xlabel(r"\bf{x}")
ax.set_ylabel(r"\bf{y}")
reciprocal.plot(ax,(-1,4),(-1,4))

sm2d.misc.voronoi_plot(ax,vor)

###################4

print "The displacement k-vector (M) is"
K=M
print K

print "The dynamical matrix at this point is"
D=sm2d.phonon.dynamical_matrix(teststruc,protopairs,testconst,K)
print D

print "The eigenvalue and eigenvectors at this point are"
eigvals,eigvecs=np.linalg.eig(D)
for val,vec in zip(eigval,eigvec):
    print val
    print vec
    print "--"

b1=np.squeeze(np.asarray(coord1))

arange=(-1,3)
brange=(-1,3)

#Since there are multiple modes, we activate them each one at a time
for idx,eigvec in enumerate(eigvecs):
    #The eigenvectors corresponds to the single basis atom
    figd=plt.figure(2+idx)
    axd=figd.add_subplot('111')
    axd=triangluar.plot(axd,arange,brange,supressvec=True)
    axd.set_title(r"\bf{Displacement field at M ("+str(idx+1)+"/2)}")
    axd.set_xlabel(r"\bf{x}")
    axd.set_ylabel(r"\bf{y}")
    evec1=np.squeeze(np.asarray(eigvec))[0:2]

    #This is the most horrendous way to go about it, but I don't give a shit anymore
    x=[]
    y=[]
    u=[]
    v=[]
    for lp in honeycomb.lattice_points_in_range(arange,brange):
        lp=np.squeeze(np.asarray(lp))
        v1=evec1*cmath.exp(dot(K*1j,(lp+b1)))
        #There's no time to sum over the star of the k-point, so we'll just take the real part of the displacement
        v1=np.real(v1)

        x.append((lp+b1)[0])
        y.append((lp+b1)[1])
        u.append(v1[0])
        v.append(v1[1])
    axd.quiver(x,y,u,v)

plt.tight_layout()
plt.show()
