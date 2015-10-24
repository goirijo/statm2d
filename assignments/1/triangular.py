import twodstatm as sm2d
import numpy as np
import math
import matplotlib.pyplot as plt

#define triangular lattice
a=np.matrix([[1],[0]])
b=np.matrix([[-0.5],[math.sqrt(3)/2]])

triangluar=sm2d.Crystal(a,b,[])
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
coord1=np.matrix([0,0]).T
coord2=np.matrix([0.5,1.0/3]).T
coord2=(2*a+b)/3
site1=sm2d.structure.Site(coord1,"A",fracmode=False)
site2=sm2d.structure.Site(coord2,"A",fracmode=False)

honeycomb=sm2d.Crystal(a,b,[site1, site2],fracmode=False)
print honeycomb

fig=plt.figure()
ax=fig.add_subplot("111")

honeycomb.plot(ax,(0,5),(0,5))

transbasis=honeycomb.transformed_basis(pgroup[3])
honeycomb2=sm2d.Crystal(a,b,transbasis,fracmode=False)
honeycomb2.plot(ax,(-5,3),(-5,3),2)

plt.show()
