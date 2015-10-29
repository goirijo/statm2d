import statm2d as sm2d
import numpy as np
import math
import matplotlib.pyplot as plt

#define triangular lattice
a=np.matrix([[1],[0]])
b=np.matrix([[-0.5],[math.sqrt(3)/2]])

lat=np.hstack((a,b))

tensorbasis=sm2d.misc.tensor_basis()

pg=sm2d.symmetry.point_group_operations(lat,3)

reybasis=[]
for basis in tensorbasis:
    print sm2d.symmetry.matrix_reynolds(basis,pg)
    print ""
    reybasis.append(sm2d.symmetry.matrix_reynolds(basis,pg))

print sm2d.misc.independent_indices(reybasis)
