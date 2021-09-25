# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 02:05:10 2021

@author: minmi
"""

import numpy as np

area = np.array([51,38,57,51,53,77,63,69,72,73])
age = np.array([16,4,16,11,4,22,5,5,2,1])
price = np.array([3.0,3.2,3.3,3.9,4.4,4.5,4.5,5.4,5.4,6.0])

def get_mean(x):
    N = x.size
    m = np.sum(x) / N
    
    return m

def get_deviation(x,y):
    a = 0.0
    m1 = get_mean(x)
    m2 = get_mean(y)
    
    for i in range(x.size):
        a += (x[i] - m1) * (y[i] - m2)
    
    #print(a)
        
    return a

def get_matrix(x,y):
    S11 = get_deviation(x,x)
    S12 = get_deviation(x,y)
    S22 = get_deviation(y,y)
    
    S = np.array([[S11,S12],
                 [S12,S22]])
    
    return S

def get_coef(Sxx,Sxy):
    inv_Sxx = np.array([[Sxx[1][1],-Sxx[0][1]],
                       [-Sxx[0][1],Sxx[0][0]]])   
    beta = np.dot(inv_Sxx,Sxy) / (Sxx[1][1] * Sxx[0][0] - Sxx[0][1] ** 2)
    
    return beta

def calc_eq(ind1,ind2,target):
    N = target.size
    y = np.zeros(N)
    
    Sxx = get_matrix(ind1,ind2)
    S1y = get_matrix(ind1,target)
    #print(Sxx)
    S2y = get_matrix(ind2,target)
    Sxy = np.array([S1y[0][1],S2y[0][1]])
    
    beta = get_coef(Sxx,Sxy)
    beta0 = get_mean(target) - beta[0] * get_mean(ind1) - beta[1] * get_mean(ind2)
    
    print("equation: y = "+'{:.2f}'.format(beta0)+" + "+'{:.4f}'.format(beta[0])+"x1 + "+'{:.4f}'.format(beta[1])+"x2")
    
    for i in range(N):
        y[i] = beta0 + beta[0] * ind1[i] + beta[1] * ind2[i]
        
    return y

if __name__ == '__main__':
    y = calc_eq(area,age,price)




