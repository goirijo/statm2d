import statm2d as sm2d
from statm2d.misc import *
import numpy as np
import cmath
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

protopairs=[(site1,site2),(site2,site1)]
teststruc=honeycomb
sg=teststruc.factor_group()
testconst=[-5,-2,-0.5,-3]
dynbasisentries=sm2d.phonon.dynamical_basis_entries(protopairs,sg,teststruc,testconst)

pairs=[p for stack,p in dynbasisentries]
dymlocs=[sm2d.phonon.dynamical_pair_location(s0,s1,teststruc) for s0,s1 in pairs]
expinputs=[sm2d.phonon.dynamical_exp_inputs(s0,s1,teststruc._lattice) for s0,s1 in pairs]


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
ax.set_title(r"\bf{Reciprocal honeycomb}")
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
kpoints=sm2d.phonon.kpath(astar,bstar,[K,G,M],[30,30,30])
#kpoints=sm2d.phonon.kpath(astar,bstar,[K,G,M],[2,2,2])

plt.tight_layout()
fig.savefig("reciprocal.eps")


figk=plt.figure(1)
axk=figk.add_subplot('111')
axk.set_title(r"\bf{Honeycomb dispersion}")
axk.set_xlabel(r"\bf{k}")
axk.set_ylabel(r"$\omega$")

symticks=[]
cummulation=0
for segment in kpoints:
    symticks.append(cummulation)
    minstep=np.linalg.norm(segment[0]-segment[1])
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

plt.tight_layout()
figk.savefig("./frequencies.eps")


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
b2=np.squeeze(np.asarray(coord2))



arange=(-1,3)
brange=(-1,3)

#Since there are multiple modes, we activate them each one at a time
for idx,eigvec in enumerate(eigvecs):
    #The eigenvectors corresponds to two basis atoms, let's start with the first one
    figd=plt.figure(2+idx)
    axd=figd.add_subplot('111')
    axd=honeycomb.plot(axd,arange,brange,supressvec=True)
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

#Since there are multiple modes, we activate them each one at a time
for idx,eigvec in enumerate(eigvecs):
    #The eigenvectors corresponds to two basis atoms, let's start with the first one
    figd=plt.figure(2+idx)
    axd=figd.add_subplot('111')
    axd.set_title(r"\bf{Displacement field at M ("+str(idx+1)+"/4)}")
    axd.set_xlabel(r"\bf{x}")
    axd.set_ylabel(r"\bf{y}")
    evec2=np.squeeze(np.asarray(eigvec))[2:4]

    #This is the most horrendous way to go about it, but I don't give a shit anymore
    x=[]
    y=[]
    u=[]
    v=[]
    for lp in honeycomb.lattice_points_in_range(arange,brange):
        lp=np.squeeze(np.asarray(lp))
        v2=evec2*cmath.exp(dot(K*1j,(lp+b2)))
        #There's no time to sum over the star of the k-point, so we'll just take the real part of the displacement
        v2=np.real(v1)

        x.append((lp+b2)[0])
        y.append((lp+b2)[1])
        u.append(v1[0])
        v.append(v1[1])
    axd.quiver(x,y,u,v)
    plt.tight_layout()
    figd.savefig("displacement"+str(idx)+".eps")

