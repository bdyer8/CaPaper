# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 14:52:01 2015

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

orgC=-25
rockC=2
injectC=-8.0
rockFrac=(1-(injectC-rockC)/(orgC-rockC))
carbonMass=2.0
caMass=rockFrac*carbonMass/(.012)*.04 #kg


meshY=3
meshX=104
v=np.ones([meshY,meshX])*0.0
u=np.ones([meshY,meshX])*0.0025535400741844192  #solution from AC test with 2.5+-5ma
aveSpeed=np.sqrt(v.mean()**2+u.mean()**2)
injectionSites=np.zeros([meshY,meshX])
injectionSites=injectionSites>0
injectionSites[15:35,15:35]=True  

caMass=np.linspace(1.0,70.0,5)
alpha=np.linspace(.9985,1.1,5)
carbonSolutions=np.zeros((30,30,104))
calciumSolutions=np.zeros((5,5,104))
carbonSolutions2=np.zeros((5,5,104))
calciumSolutions2=np.zeros((5,5,104))
mesh=DiagenesisMesh.meshRock(meshX,meshY,u,v,2,2,-1.05,3.5703864479316496e-07,injectionSites,[injectC,-10.0,-1.5,0.0],[1.0,1.0,1.0],[[240.0,carbonMass],[960.0,889.0],[800.0,4.2848305187685645]]) #reaction rate from same sim
mesh.inject(int(3.5e6/mesh.dt))       

#%%
for i in range(5):
    for j in range(5):
        print(str(i)+','+str(j))
        mesh=DiagenesisMesh.meshRock(meshX,meshY,u,v,2,2,-1.04,3.5703864479316496e-07,injectionSites,[injectC,-10.0,-1.0,0.0],[1.0,1.0,alpha[j]],[[240.0,carbonMass],[960.0,889.0],[800.0,caMass[i]]]) #reaction rate from same sim
        mesh.inject(int(2.5e6/mesh.dt))
        carbonSolutions[i,j,:]=mesh.printBox('rock','d13c')[1,:]
        calciumSolutions[i,j,:]=mesh.printBox('rock','d44ca')[1,:]
        mesh.inject(int(2.0e6/mesh.dt))
        carbonSolutions2[i,j,:]=mesh.printBox('rock','d13c')[1,:]
        calciumSolutions2[i,j,:]=mesh.printBox('rock','d44ca')[1,:]


#mesh.inject(int(2.5e6/mesh.dt))

#%%
mesh.inject(int(4.5e6/mesh.dt))

#%%

fig = plt.figure(figsize=(12, 8))
with plt.style.context('fivethirtyeight'):
    rcParams.update({'figure.autolayout': True})
    rcParams.update({'font.size': 8.0})
    rcParams.update({'figure.subplot.bottom': 0})
    rcParams.update({'figure.subplot.hspace': 0.0})
    rcParams.update({'figure.subplot.left': 0.0})
    rcParams.update({'figure.subplot.right': 1.0})
    rcParams.update({'figure.subplot.top': 1.0})
    rcParams.update({'figure.subplot.wspace': 0.0})
    rcParams.update({'axes.color_cycle': [u'#30a2da',
                                u'#fc4f30',
                                u'#e5ae38',
                                u'#6d904f',
                                u'#8b8b8b',
                                '#d33682',
                                '#2aa198']})
    
    gs = gridspec.GridSpec(2, 2) 
    AC = plt.subplot(gs[:,0])
    #BW = plt.subplot(gs[:,3:])
    AC.scatter(ArrowCanyon.d13c,ArrowCanyon.SAMP_HEIGHT,s=10,color=[.7,.7,.7],edgecolor='none')
    AC.set_xlabel('$\delta$13C')
    AC.set_ylabel('height(m)')
    AC.set_ylim([0,120])
    AC.set_xlim([-10,6])
    BattleshipWash.d44ca=BattleshipWash.d44ca[BattleshipWash.d44ca<-.9]
    cmin=round(np.min([ArrowCanyon.d44ca.min(),BattleshipWash.d44ca.min()]),1)
    cmax=round(np.max([ArrowCanyon.d44ca.max(),BattleshipWash.d44ca.max()]),1)
    specCM=plt.get_cmap('Spectral') 
    minCa=-1.4
    rangeCa=.4
    for i in range(meshX-1):
        avCa=np.mean(mesh.printBox('rock','d44ca')[1,i:i+2])
        AC.plot(mesh.printBox('rock','d13c')[1,i:i+2],np.linspace(meshX-(i-1),meshX-i,2),color=specCM(np.abs(minCa-avCa)/rangeCa))
    AC.scatter(BattleshipWash.d13c,BattleshipWash.SAMP_HEIGHT-240.0,s=10,color=[.7,.7,.7],edgecolor='none')
    cmax=-1.0
    d44ca=AC.scatter(ArrowCanyon.d13c,ArrowCanyon.SAMP_HEIGHT,c=ArrowCanyon.d44ca,cmap='Spectral',vmin=cmin,vmax=cmax,s=25,edgecolor=[.2,.2,.2])
    cbar=fig.colorbar(d44ca,ax=AC,label='$\delta$44Ca', orientation='vertical',pad=.05,shrink=.4,ticks=[ round(a, 1) for a in np.linspace(cmin,cmax,7)])
    d44ca=AC.scatter(BattleshipWash.d13c,BattleshipWash.SAMP_HEIGHT-240.0,c=BattleshipWash.d44ca,cmap=viridis,s=25,edgecolor=[.2,.2,.2],vmin=cmin,vmax=cmax)
    #cbar=fig.colorbar(d44ca,ax=AC,label='$\delta$44Ca', orientation='vertical',pad=.01,shrink=.25,ticks=[ round(a, 1) for a in np.linspace(cmin,cmax,7)])
    #fig.savefig('CaData.pdf', format='pdf', dpi=300)
    
    Xplt = plt.subplot(gs[0,1])
    Xplt2 = plt.subplot(gs[1,1])
    Xplt.plot(mesh.printBox('rock','d13c')[1,:],mesh.printBox('rock','d44ca')[1,:],color=plt.rcParams['axes.color_cycle'][0])
    Xplt.plot(ArrowCanyon.d13c,ArrowCanyon.d44ca,markeredgecolor='none',linestyle='none',marker='o',color=plt.rcParams['axes.color_cycle'][1],alpha=.7)
    Xplt.plot([2,-8],[-1,-1.4],markeredgecolor='none',linestyle='none',marker='o',color=plt.rcParams['axes.color_cycle'][4],alpha=1,markersize=10)    
    Xplt2.plot(mesh.printBox('rock','d18o')[1,:],mesh.printBox('rock','d44ca')[1,:],color=plt.rcParams['axes.color_cycle'][0])
    Xplt2.plot(ArrowCanyon.d18o,ArrowCanyon.d44ca,markeredgecolor='none',linestyle='none',marker='o',color=plt.rcParams['axes.color_cycle'][1],alpha=.7)
    Xplt2.plot([-1,-10],[-1,-1.4],markeredgecolor='none',linestyle='none',marker='o',color=plt.rcParams['axes.color_cycle'][4],alpha=1,markersize=10)
    
    Xplt.set_xlabel('$\delta$13C')
    Xplt.set_ylabel('$\delta$44Ca')
    Xplt.set_xlim(-10,3.0)
    Xplt2.set_xlabel('$\delta$18O')
    Xplt2.set_xlim(-12,0.0)
    Xplt2.set_ylabel('$\delta$44Ca')
    

#fig.savefig('crossPlotsModelFit.pdf', format='pdf', dpi=300)


#%%

LV1meters=np.array(Leadville.SAMP_HEIGHT[Leadville.d13c<0]+9.0)
LV1d13c=Leadville.d13c[Leadville.d13c<0]
LV1d44ca=Leadville.d44ca[(Leadville.d13c<0) & (Leadville.SAMP_HEIGHT>20)]
LV1d13c=Leadville.d13c[LV1d44ca.index]
LV1meters=Leadville.SAMP_HEIGHT[LV1d13c.index]
LV1d44cameters=Leadville.SAMP_HEIGHT[Leadville.d44ca[Leadville.d13c<0].index]+9

with plt.style.context('ggplot'):
    rcParams.update({'figure.autolayout': True})
    fig = plt.figure(figsize=(6, 8))
    gs = gridspec.GridSpec(1, 1) 
    AC = plt.subplot(gs[:,:])
    #BW = plt.subplot(gs[:,3:])
    AC.scatter(Leadville.d13c,Leadville.SAMP_HEIGHT,s=10,color=[.7,.7,.7],edgecolor='none')
    AC.set_xlabel('$\delta$13C')
    AC.set_ylabel('height(m)')
    AC.set_ylim([0,120])
    AC.set_xlim([-10,6])
    cmin=LV1d44ca.min()
    cmax=LV1d44ca.max()
    d44ca=AC.scatter(LV1d13c,LV1meters,c=LV1d44ca,cmap='Spectral',vmin=cmin,vmax=cmax,s=25,edgecolor=[.2,.2,.2])
    cbar=fig.colorbar(d44ca,ax=AC,label='$\delta$44Ca', orientation='vertical',pad=.05,shrink=.4,ticks=[ round(a, 1) for a in np.linspace(cmin,cmax,7)])
    #cbar=fig.colorbar(d44ca,ax=AC,label='$\delta$44Ca', orientation='vertical',pad=.01,shrink=.25,ticks=[ round(a, 1) for a in np.linspace(cmin,cmax,7)])
    specCM=plt.get_cmap('Spectral') 
    minCa=cmin
    rangeCa=np.abs(cmin-cmax)
    for i in range(meshX-1):
        avCa=np.mean(mesh.printBox('rock','d44ca')[1,i:i+2])
        AC.plot(mesh.printBox('rock','d13c')[1,i:i+2],np.linspace(meshX-(i-1),meshX-i,2),color=specCM(np.abs(minCa-avCa)/rangeCa))
       
    fig.savefig('LVCaData.pdf', format='pdf', dpi=600)