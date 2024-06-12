#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 13:30:05 2023

@author: owenpannucci
"""
import numpy as np
from matplotlib import pyplot as plt

##### globals #####
k = (np.pi)**2
# k = np.pi
###################

#%% Functions

#returns the function to be integrated
def func(k,x):
    return np.exp(np.cos(k*x))

#trapezoid rule
def trap(a, b, n):
    #step size h
    #start interval : a
    #end interval: b
    #num points: n
    h = (b - a)/n
    
    summation = 0
    for i in range(1,n):
        xi = a + i*h
        #print("i", i, "xi", xi)
        summation += func(k,xi)
        
    return h * ((func(k,a) + func(k,b))/2 + summation)


#%% Find n such that delta_n+1 < abs(I_n+1 -I_n)

pts = 1
delta = abs(trap(-1,1,pts + 1) - trap(-1,1,pts))
delta_list = []
delta_list.append(delta)

#finds N s.t. absolute error approximation is less than 10E-10
while delta > 10e-10:
    pts += 1
    delta = abs(trap(-1,1,pts + 1) -trap(-1,1,pts))
    delta_list.append(delta)
    
print("N: ",pts)
    
    
#%% Call trapezoid function 
print(trap(-1,1,pts))

#%% Plots

def ref_lin(x):
    return 1/x
def ref_quad(x):
    return 1/(x**2)
def ref_exp(x):
    return 4**-(2*x)

# e^{\cos (\pi * x)}

x = np.linspace(0, pts, pts)
# plt.figure(1)
# plt.loglog(x, delta_list,'bo--',markersize = 3, label = r'$e^{\cos (\pi * x)}$')
# plt.loglog(x, delta_list,'ro--',markersize = 3, label = r'$e^{\cos (\pi^{2} * x)}$')

# # reference curves
# plt.loglog(x,ref_quad(x),'g',label = 'Quad ref')
# plt.loglog(x,ref_lin(x),'orange',label = 'Linear ref')
# plt.loglog(x,ref_exp(x),'purple', label = 'Exp ref')

# # plt.legend()

# plt.xlabel(r'$n$', fontsize='large')
# plt.ylabel(r'$error$', fontsize='large')


   
    
    
    
    
    
    