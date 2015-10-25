import twodstatm as sm2d
import numpy as np
import math
import matplotlib.pyplot as plt

#define triangular lattice
a=np.matrix([[1],[0]])
b=np.matrix([[-0.5],[math.sqrt(3)/2]])

lat=np.hstack((a,b))
