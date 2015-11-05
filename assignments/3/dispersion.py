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

equiv0,equiv1,syms=sm2d.phonon.equivalent_clusters(pair0,pair1,pgroup)

print "The unique equivalent clusters are"
for eq0,eq1,op in zip(equiv0,equiv1,syms):
    print eq0
    print eq1
    print op
    print ""

