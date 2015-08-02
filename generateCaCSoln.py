# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 09:09:56 2015

@author: bdyer
"""


import DiagenesisMesh
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from numpy import load,linspace,meshgrid
from matplotlib import animation
from pylab import rcParams
from matplotlib import gridspec
import pickle
from matplotlib import rcParams
import pandas as pd
import time
from matplotlib.colors import LinearSegmentedColormap
cm_data = pickle.load( open( "viridis.pkl", "rb" ) )
viridis = LinearSegmentedColormap.from_list('viridis', np.flipud(cm_data))
ArrowCanyon=pd.read_csv('samples_ArrowCanyon.csv')
Leadville=pd.read_csv('samples_B416.csv')
BattleshipWash=pd.read_csv('samples_BattleshipWash.csv')

orgC=-22
rockC=2
injectC=-8.0
rockFrac=(1-(injectC-rockC)/(orgC-rockC))
carbonMass=2.0
caMass=rockFrac*carbonMass/(.012)*.04 #kg
#db=pm.database.pickle.load('6Sections_Diagenesis_ACFIT.pickle')
#db.__dict__
ACvel=0.0029535400741844192
LV1vel=0.049034264297194326

meshY=3
meshX=104
v=np.ones([meshY,meshX])*0.0
u=np.ones([meshY,meshX])*ACvel
aveSpeed=np.sqrt(v.mean()**2+u.mean()**2)
injectionSites=np.zeros([meshY,meshX])
injectionSites=injectionSites>0
injectionSites[:,0]=True  
mesh=DiagenesisMesh.meshRock(meshX,meshY,u,v,2,2,-0.96,3.5703864479316496e-07,injectionSites,[injectC,-10.0,-1.2,0.0],[1.0,1.0,1.0],[[329.0,2.0],[1315.0,889.0],[1096.0,2.0]]) #reaction rate from same sim


#%%
def rockMassFrac(rockCDelta,metCDelta,orgCDelta):
    return 1-(((metCDelta-rockCDelta)/(orgCDelta-rockCDelta)))

carbonMass=np.linspace(.240,240.0,10)  #A_to_R=np.linspace(1.0,1000.0,10)
carbonSolutions=np.zeros((10,10,100,104))
calciumSolutions=np.zeros((10,10,100,104))
     
N=100
injectCa=np.linspace(-1.8,-.9,10)
CaMass=rockMassFrac(2.0,-8.0,-22.0)*carbonMass/12.0*40
mesh=DiagenesisMesh.meshRock(meshX,meshY,u,v,2,2,-1.04,3.5703864479316496e-07,injectionSites,[injectC,-10.0,-1.0,0.0],[1.0,1.0,1.0],[[329.0,2],[960.0,889.0],[800.0,2.5]]) #reaction rate from same sim
        

#%%

for i in range(10):
    for k in range(10):
        mesh=DiagenesisMesh.meshRock(meshX,meshY,u,v,2,2,-1.04,3.5703864479316496e-07,injectionSites,[injectC,-10.0,injectCa[k],0.0],[1.0,1.0,1.0],[[329.0,carbonMass[i]],[1315.0,889.0],[1096.0,CaMass[i]]]) #reaction rate from same sim
        for j in range(N):
            mesh.inject(int(40000/N))
            carbonSolutions[i,k,j,:]=mesh.printBox('rock','d13c')[1,:]
            calciumSolutions[i,k,j,:]=mesh.printBox('rock','d44ca')[1,:]
            
            
#%%
            
pickle.dump( carbonSolutions, open( "d13cAC.pkl", "wb" ) )
pickle.dump( calciumSolutions, open( "d44caAC.pkl", "wb" ) )

