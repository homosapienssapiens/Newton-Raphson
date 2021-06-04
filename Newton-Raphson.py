# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 21:25:34 2021

@author: Miguel
"""

import sympy as sp
import numpy as np
from math import log10, floor

x, y, z = sp.symbols('x, y, z')

#Function: Returns the given value to the given significative figures.
# x: Value
# sig: significative figures quantity
round_sig = lambda x, sig=2 : round(x, sig-int(floor(log10(abs(x))))-1)

# Function: Newton-Raphson method to find the aproximation for one variable. 
# f: Function to be found
# i: Initial value (x0)
# e: Error index (How much exactitude do you want?)
def newraph(f, i, e):
    n = 0
    while abs(f.subs(x, i).evalf()) > e:
        d = f.diff(x, 1)
        s1 = f.subs(x, i).evalf()
        s2 = d.subs(x, i).evalf()
        i = i - (s1/s2)
        n += 1
    print('Solution: ' + str(round_sig(i, 10)))
    print('Iterations: ' + str(n))
    print('Exactitude: ' + str(e))
    print('Precission: 10 significative figures')
    print('\n')
    return

# Function: Newton-Raphson method to find the aproximation for two variables. 
# f: Function to be found
# i: Initial value in x and y
# e: Error index (How much exactitude do you want?)
def newraph2(f, i, e):
    n = 0
    while True:
        d = f.jacobian(x0)
        s1 = f.subs([(x, i[0]), (y, i[1])]).evalf()
        s2 = d.subs([(x, i[0]), (y, i[1])]).evalf()
        i = i - s2.inv()*s1
        n += 1
        if np.linalg.norm(s1, np.inf) < e:
            break
    cont = 0
    for _ in i:
        print('Solution ' + str(x0[cont]) + ': ' + str(round_sig(_, 5)))
        cont += 1
    print('Iterations: ' + str(n))
    print('Exactitude: ' + str(e))
    print('Precission: 5 significative figures')
    print('\n')
    return
    
# Function: Newton-Raphson method to find the aproximation for tree variables. 
# f: Function to be found
# i: Initial value in x, y and z
# e: Error index (How much exactitude do you want?)
def newraph3(f, i, e):
    n = 0
    while True:
        d = f.jacobian(x0)
        s1 = f.subs([(x, i[0]), (y, i[1]), (z, i[2])]).evalf()
        s2 = d.subs([(x, i[0]), (y, i[1]), (z, i[2])]).evalf()
        i = i - s2.inv()*s1
        n += 1
        if np.linalg.norm(s1, np.inf) < e:
            break
    cont = 0
    for _ in i:
        print('Solution ' + str(x0[cont]) + ': ' + str(round_sig(_, 5)))
        cont += 1
    print('Iterations: ' + str(n))
    print('Exactitude: ' + str(e))
    print('Precission: 5 significative figures')
    print('\n')
    return



# One variable example
print('First example - One variable: 3 * pow(x, 2) - pow(sp.exp(1), x\n\n')
print('One variable - First solution')
f = 3 * pow(x, 2) - pow(sp.exp(1), x)
e = 0.0001
i = 1.5 
print('Initial value: x = ' + str(i))

newraph(f, i, e)

print('One variable - Second solution')
f = 3 * pow(x, 2) - pow(sp.exp(1), x)
e = 0.0001
i = -1.5
print('Initial value: x = ' + str(i))

newraph(f, i, e)

print('One variable - Third solution')
f = 3 * pow(x, 2) - pow(sp.exp(1), x)
e = 0.0001
i = 4
print('Initial value: x = ' + str(i))

newraph(f, i, e)


# Two variables example

# We'll define our x0 with 3 variables
x0 = sp.Matrix([x, y])

print('Second example - Two variables: x * sp.sin(y) - 1, x**2 + y**2 - 4\n\n')
print('Two variables - First solution')
f = sp.Matrix([x * sp.sin(y) - 1, x**2 + y**2 - 4])
e = 0.0001
i = sp.Matrix([3, 3])
print('Initial values: x = ' + str(i[0]) + ', y = ' + str(i[1]))

newraph2(f, i, e)


print('Two variables - Second solution')
f = sp.Matrix([x * sp.sin(y) - 1, x**2 + y**2 - 4])
e = 0.0001
i = sp.Matrix([3, -3])
print('Valores iniciales: x = ' + str(i[0]) + ', y = ' + str(i[1]))

newraph2(f, i, e)


print('Two variables - Third solution')
f = sp.Matrix([x * sp.sin(y) - 1, x**2 + y**2 - 4])
e = 0.0001
i = sp.Matrix([2, 1])
print('Valores iniciales: x = ' + str(i[0]) + ', y = ' + str(i[1]))

newraph2(f, i, e)


print('Two variables - Fourth solution')
f = sp.Matrix([x * sp.sin(y) - 1, x**2 + y**2 - 4])
e = 0.0001
i = sp.Matrix([-2, -1])
print('Valores iniciales: x = ' + str(i[0]) + ', y = ' + str(i[1]))

newraph2(f, i, e)


# Tree variables example

# We'll define our x0 with 3 variables
x0 = sp.Matrix([x, y, z])

print('Third example - Tree variables: x + y - z + 2, x**2 + y, z - y**2- 1\n\n')
print('Tree variables - First solution')
f = sp.Matrix([x + y - z + 2, x**2 + y, z - y**2- 1])
e = 0.0001
i = sp.Matrix([1, 1, 1])
print('Valores iniciales: x = ' + str(i[0]) + ', y = ' + str(i[1]) + ', z = ' + str(i[2]))

newraph3(f, i, e)

print('Tree variables - Second solution')
f = sp.Matrix([x + y - z + 2, x**2 + y, z - y**2- 1])
e = 0.0001
i = sp.Matrix([1, -1, 1])
print('Valores iniciales: x = ' + str(i[0]) + ', y = ' + str(i[1]) + ', z = ' + str(i[2]))

newraph3(f, i, e)



