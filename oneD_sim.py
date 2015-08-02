# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 10:55:04 2015

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

meshY=10
meshX=100
v=np.ones([meshY,meshX])*0.0
u=np.ones([meshY,meshX])  #solution from AC test with 2.5+-5ma
multiplier=np.linspace(-1.5,1,meshY-1)
for i in range(1,meshY,1):
    u[i,:]=u[i,:]*10**multiplier[i-1]
u[0,:]=u[1,:]
u[-1,:]=u[-2,:]

aveSpeed=np.sqrt(v.mean()**2+u.mean()**2)
injectionSites=np.zeros([meshY,meshX])
injectionSites=injectionSites>0
injectionSites[:,0]=True  
mesh=DiagenesisMesh.meshRock(meshX,meshY,u,v,2,-1,-1,1e-4,injectionSites,[-8.0,-10.0,-1.5,0.0]) #reaction rate from same sim



#%%
def ratioC(v,R):  #velocities, reaction rates
    return (2*.1*v)/(R*240.0) #porosity, fluid mass, rock mass
    
N=900
rockCarbonBox=np.zeros((N,10,100))
rockOxygenBox=np.zeros((N,10,100))
for i in range(N):
    rockCarbonBox[i,:,:]=mesh.printBox('rock','d13c')
    rockOxygenBox[i,:,:]=mesh.printBox('rock','d18o')
    mesh.inject(100)
#%%
fig = plt.figure(figsize=(8,10))
gs = gridspec.GridSpec(3,2) 
AgePlot = plt.subplot(gs[:,0])
StratPlot = plt.subplot(gs[:2,1])
crossPlot = plt.subplot(gs[2,1])
with plt.style.context('fivethirtyeight'):
    rcParams.update({'figure.autolayout': True})
    rcParams.update({'font.size': 10.0})
    rcParams.update({'figure.subplot.bottom': 0})
    rcParams.update({'figure.subplot.hspace': 0.0})
    rcParams.update({'figure.subplot.left': 0.0})
    rcParams.update({'figure.subplot.right': 1.0})
    rcParams.update({'figure.subplot.top': 1.0})
    rcParams.update({'figure.subplot.wspace': 0.0})
        
    speed = np.sqrt(mesh.u*mesh.u + mesh.v*mesh.v)
    lw = 1.0*speed/speed.max()
    
    yW=100;xW=25;
    x = np.linspace(0,yW,mesh.shape[1])
    y = np.linspace(0,xW,mesh.shape[0])
    X,Y = np.meshgrid(y,x)
            
    im3 = AgePlot.imshow(np.log10((mesh.printBox('fluid','age'))).T, cmap='gist_ncar',aspect='auto',extent=[0,xW,yW,0])
    fig.colorbar(im3, ax=AgePlot, label='log10 fluid age (years)', orientation='horizontal',pad=-.05,shrink=.9)
    qui = AgePlot.streamplot(X, Y, (mesh.v.T), mesh.u.T,color=np.log10(speed.T),cmap=viridis, linewidth=2,density=.5)
    AgePlot.set_xlim([0,25])
    AgePlot.set_ylim([100,1])    
    #grid.cbar_axes[i].colorbar(qui.lines)
    fig.colorbar(qui.lines, ax=AgePlot, label='log10 speed', orientation='horizontal',pad=0.05,shrink=.9)
    AgePlot.grid(None)
    #AgePlot.axis('off')
    specCM=plt.get_cmap(viridis) 
    rockCarbon=mesh.printBox('rock','d13c')
    rockOxygen=mesh.printBox('rock','d18o')
    sections=np.linspace(1,9,9).astype(int)
    for k in sections:
        for i in range(0,N,N-1):StratPlot.plot(rockCarbonBox[i,k,1:],range(99,0,-1),color=specCM(np.abs(-1.5-np.log10(speed.T[2,k]))/2.0),alpha=1.0*i/N,lw=2)
    StratPlot.set_xlim([-10,3])
    StratPlot.set_xlabel('$\delta^{13}$C')
    StratPlot.set_ylabel('Meters')
    for k in sections:
        for i in range(1,N):crossPlot.plot(rockOxygenBox[i-1:i+1,k,1],rockCarbonBox[i-1:i+1,k,1],color=specCM(np.abs(-1.5-np.log10(speed.T[2,k]))/2.5),alpha=1.0*i/N,lw=2)
        crossPlot.plot(rockOxygenBox[-1,k,1],rockCarbonBox[-1,k,1],color=specCM(np.abs(-1.5-np.log10(speed.T[2,k]))/2.5),alpha=1.0,linestyle='none',marker='o',markersize=10,markeredgecolor=[.2,.2,.2])
    
    crossPlot.set_xlim([-12,0])
    crossPlot.set_ylim([-9,4])
    crossPlot.set_ylabel('$\delta^{13}$C')
    crossPlot.set_xlabel('$\delta^{18}$O')
    
    fig.savefig('1D.pdf', format='pdf', dpi=600)
    
    