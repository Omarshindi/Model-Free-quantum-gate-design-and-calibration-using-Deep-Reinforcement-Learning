# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 15:59:31 2021

@author: z5277102
"""


## To generate the initial states for testing

import numpy as np
from scipy.linalg import expm
import itertools
import math 



class S0_G_t (object):
    def __init__(self):
        
        super(S0_G_t, self)               
        self.sx=np.mat(([0,1],[1,0]))
        self.sy=np.mat(([0,-1j],[1j, 0]))
        self.sz=np.mat(([1, 0],[0,-1])) 
        self.H_T=[[(1/np.sqrt(2)), (1/np.sqrt(2))],
                [(1/np.sqrt(2)), -(1/np.sqrt(2))],]  
                       
        #self.st=np.mat(([1,0],[0,np.cos(0.01)+np.sin(0.01)*1j]))

    def reset(self, S0):
        self.state0 = S0
        #gg=(np.pi*np.random.uniform())
        gg=(np.random.uniform())
        #gg=(np.pi/4)

        self.st=np.mat(([1,0],[0,np.cos(gg)+np.sin(gg)*1j]))
        self.cos=np.cos(gg)
        
    def step(self):
        u=np.random.randint(-4,4)#np.random.uniform()
        #H=np.dot(np.dot(self.H_T,self.st),self.H_T)
        H0=self.sz        
        H1=u*self.sx
        H=np.array(H0+H1) 
        R=expm(-1j*H*0.05)        
        Final_state =np.dot(self.state0,R)
        #print(H)
        #Final_state= np.dot(self.state0, H)  
        #SS=np.dot(np.transpose(Final_state),Final_state)
        
        #fidelity = np.abs((np.dot(np.transpose(Final_state),self.stateT))**2)
        #fidelity1 = np.abs((np.dot(Final_state,self.stateT))**2)
        #fidelity=fidelity1.item()
        #reward = -math.log(1-fidelity)


        return Final_state,self.cos




