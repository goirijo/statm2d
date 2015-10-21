import twodstatm as sm2d
import numpy as np
import math

#define triangular lattice
a=np.matrix([[1],[0]])
b=np.matrix([[-0.5],[math.sqrt(3)/2]])

triangluar=sm2d.Crystal(a,b,[])
pgroup=sorted(triangluar.point_group())

tablestring=""
for op1 in pgroup:
    for op2 in pgroup:
        mat1=op1.matrix
        mat2=op2.matrix

        tau=np.matrix([[0],[0]])
        prodmat=mat1.dot(mat2)
        prodsymop=sm2d.symmetry.SymOp(prodmat,tau)
        tablestring+=prodsymop.name+"    "
    tablestring+="\n"

print tablestring
