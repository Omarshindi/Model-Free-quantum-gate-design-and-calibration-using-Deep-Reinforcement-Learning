# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 19:36:28 2021

@author: z5277102
"""

## To generate the initial states 

import numpy as np
from scipy.linalg import expm
import itertools
import math 



class S0_G (object):
    def __init__(self):
        
        super(S0_G, self)               
        self.sx=np.mat(([0,1],[1,0]))
        self.sy=np.mat(([0,-1j],[1j, 0]))
        self.sz=np.mat(([1, 0],[0,-1])) 
        self.H_T=[[(1/np.sqrt(2)), (1/np.sqrt(2))],
                [(1/np.sqrt(2)), -(1/np.sqrt(2))],]  
                       
        #self.st=np.mat(([1,0],[0,np.cos(np.pi/1200)+np.sin(np.pi/1200)*1j]))
        gg=(np.pi*np.random.uniform())
        self.st=np.mat(([1,0],[0,np.cos(gg)+np.sin(gg)*1j]))
        self.cos=np.cos(gg)

    def reset(self, S0):
        self.state0 = S0

    def step(self):
           
        H=np.dot(np.dot(self.H_T,self.st),self.H_T)
        #print(H)
        Final_state= np.dot(self.state0, H)  
        #SS=np.dot(np.transpose(Final_state),Final_state)
        
        #fidelity = np.abs((np.dot(np.transpose(Final_state),self.stateT))**2)
        #fidelity1 = np.abs((np.dot(Final_state,self.stateT))**2)
        #fidelity=fidelity1.item()
        #reward = -math.log(1-fidelity)


        return Final_state,self.cos




