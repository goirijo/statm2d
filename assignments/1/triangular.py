import statm2d as sm2d
import numpy as np
import math
import matplotlib.pyplot as plt

#define triangular lattice
a=np.matrix([[1],[0]])
b=np.matrix([[-0.5],[math.sqrt(3)/2]])

triangluar=sm2d.Crystal(a,b,[],fracmode=False)
pgroup=triangluar.point_group()

print "The lattice is defined as"
print triangluar._lattice
print ""

print "The point group operations for this lattice are:"
for op in pgroup:
    print op
    print ""

print "The multiplication table of the point_group is:"
tablestring=""
for op1 in pgroup:
    for op2 in pgroup:
        prodsymop=op1*op2
        tablestring+=prodsymop.name+" "*(8-len(prodsymop.name))
    tablestring+="\n"

print tablestring

if sm2d.symmetry.is_closed(pgroup):
    print "The calculated operations form a closed group"


#Define basis for honeycomb structure
dumbshift=np.matrix([1.2,0.2]).T
coord1=np.matrix([0,0]).T
coord2=(2*a+b)/3
#coord1=(2*b+a)/3
site1=sm2d.structure.Site(coord1,"A")
site2=sm2d.structure.Site(coord2,"A")

honeycomb=sm2d.Crystal(a,b,[site1, site2],False)
print honeycomb


for i,op in enumerate(pgroup):
    fig=plt.figure(i)
    ax=fig.add_subplot('111')
    ax.set_title(r"\bf{"+op.name+"}")
    ax.set_xlabel(r"\bf{x}")
    ax.set_ylabel(r"\bf{y}")
    honeycomb.plot(ax,(-1,4),(-1,4))
    transbasis=honeycomb.transformed_basis(op)
    honeycomb2=sm2d.Crystal(a,b,transbasis,fracmode=False)
    honeycomb2.plot(ax,(-4,1),(-4,1),2)

fgroup=honeycomb.factor_group()

print "The factor group for the honeycomb lattice is:"
for op in fgroup:
    print op
    print ""


plt.show()
