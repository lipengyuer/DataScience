'''
Created on 2018年11月20日

@author: pyli
'''
import numpy as np
from sympy import *
x = symbols("x") 
y = 2*x + x**2
dy = diff(y, x)
print(dy)
print(dy.subs('x', 1))
a = np.array([[1,2], [2,3]])
print(a)
b = np.insert(a, 0, -1, axis=1)
print(b)