# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 11:09:29 2015

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

#%%
with plt.style.context('ggplot'):
    rcParams.update({'figure.autolayout': True})
    fig = plt.figure(figsize=(6, 8))
    gs = gridspec.GridSpec(1, 1) 
    AC = plt.subplot(gs[:,:])
    #BW = plt.subplot(gs[:,3:])
    AC.scatter(ArrowCanyon.d13c,ArrowCanyon.SAMP_HEIGHT,s=10,color=[.7,.7,.7],edgecolor='none')
    AC.set_xlabel('$\delta$13C')
    AC.set_ylabel('height(m)')
    AC.set_ylim([0,120])
    AC.set_xlim([-10,6])
    BattleshipWash.d44ca=BattleshipWash.d44ca[BattleshipWash.d44ca<-.9]
    cmin=round(np.min([ArrowCanyon.d44ca.min(),BattleshipWash.d44ca.min()]),1)
    cmax=round(np.max([ArrowCanyon.d44ca.max(),BattleshipWash.d44ca.max()]),1)
    AC.scatter(BattleshipWash.d13c,BattleshipWash.SAMP_HEIGHT-240.0,s=10,color=[.7,.7,.7],edgecolor='none')
    cmax=-1.0
    d44ca=AC.scatter(ArrowCanyon.d13c,ArrowCanyon.SAMP_HEIGHT,c=ArrowCanyon.d44ca,cmap=viridis,vmin=cmin,vmax=cmax,s=25,edgecolor=[.2,.2,.2])
    cbar=fig.colorbar(d44ca,ax=AC,label='$\delta$44Ca', orientation='vertical',pad=.05,shrink=.4,ticks=[ round(a, 1) for a in np.linspace(cmin,cmax,7)])
    d44ca=AC.scatter(BattleshipWash.d13c,BattleshipWash.SAMP_HEIGHT-240.0,c=BattleshipWash.d44ca,cmap=viridis,s=25,edgecolor=[.2,.2,.2],vmin=cmin,vmax=cmax)
    

#%%
meshY=3
meshX=104
v=np.zeros([meshY,meshX])
u=np.ones([meshY,meshX])
aveSpeed=np.sqrt(v.mean()**2+u.mean()**2)

modelData=(ArrowCanyon.d13c[ArrowCanyon.SAMP_HEIGHT<103])
modelData2=(BattleshipWash.d13c[(((BattleshipWash.SAMP_HEIGHT-240)<103) & ((BattleshipWash.SAMP_HEIGHT-240)>0))])
modelData=list(modelData)+list(modelData2)
modelHeight=(ArrowCanyon.SAMP_HEIGHT[ArrowCanyon.SAMP_HEIGHT<103])
modelHeight2=(BattleshipWash.SAMP_HEIGHT[(((BattleshipWash.SAMP_HEIGHT-240)<103) & ((BattleshipWash.SAMP_HEIGHT-240)>0))]-240.0)
modelHeight=list(modelHeight)+list(modelHeight2)
LVData=Leadville.d13c[Leadville.d13c<0]
LVHeight=Leadville.SAMP_HEIGHT[Leadville.d13c<0]+7

injectionSites=np.zeros([meshY,meshX])
injectionSites=injectionSites>0
injectionSites[:,0]=True
N=100
K=20
rmse=np.zeros((K,N))
rmse2=np.zeros((K,N))
mag=np.linspace(4,8.0,K)
age=10000
meshd13c=np.zeros((K,N,meshX))
for j in range(K):
    ACmesh=DiagenesisMesh.meshRock(meshX,meshY,u*.07,v,2,2,-1,aveSpeed/(10**mag[j]),injectionSites) 
    for k in range(N):
        ACmesh.inject(int(int(age/ACmesh.dt)/N))
        meshd13c[j,k,:]=ACmesh.printBox('rock','d13c')[1,:]
        rmse[j,k]=calcRMSE(ACmesh.printBox('rock','d13c')[1,1:],np.linspace(meshX-1,0,meshX-1),modelData,modelHeight)
        rmse2[j,k]=calcRMSE(ACmesh.printBox('rock','d13c')[1,1:],np.linspace(meshX-1,0,meshX-1),LVData,LVHeight)
with plt.style.context('fivethirtyeight'):
    fig=plt.figure(figsize=(16, 5))
    gs = gridspec.GridSpec(1, 2) 
    ac = plt.subplot(gs[0,0])
    lv = plt.subplot(gs[0,1])
    acFig=ac.imshow(rmse,cmap=viridis,interpolation='none',extent=[0,age/(1e6),mag[K-1],mag[0]], aspect='auto'); ac.set_ylabel('Advection to Reaction Ratio (10^N)'); ac.set_xlabel('age of diagenesis (Ma)'); fig.colorbar(acFig, ax=ac, label='rmse', orientation='vertical',pad=.0)
    lvFig=lv.imshow(rmse2,cmap=viridis,interpolation='none',extent=[0,age/(1e6),mag[K-1],mag[0]], aspect='auto'); lv.set_ylabel('Advection to Reaction Ratio (10^N)'); lv.set_xlabel('age of diagenesis (Ma)'); fig.colorbar(lvFig, ax=lv, label='rmse', orientation='vertical',pad=.0)


#np.unravel_index(np.argmin(rmse2),rmse2.shape)

def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

#save_object(rmse, 'ACrmse4ma.pkl')
#save_object(rmse2, 'LVrmse4ma.pkl')
#save_object(meshd13c, 'MeshSolutions_testing.pkl')

#%%  

ACmesh=DiagenesisMesh.meshRock(meshX,meshY,u,v,2,2,-1,aveSpeed/(10**mag[6]),injectionSites) 
ACmesh.inject(int(600000/ACmesh.dt))

with plt.style.context('ggplot'):
    rcParams.update({'figure.autolayout': True})
    fig = plt.figure(figsize=(6, 8))
    gs = gridspec.GridSpec(1, 1) 
    AC = plt.subplot(gs[:,:])
    #BW = plt.subplot(gs[:,3:])
    AC.plot(ACmesh.printBox('rock','d13c')[1,1:],np.linspace(meshX-1,0,meshX-1),lw=2,alpha=.66)
    AC.scatter(ArrowCanyon.d13c,ArrowCanyon.SAMP_HEIGHT,s=10,color=[.7,.7,.7],edgecolor='none')
    AC.set_xlabel('$\delta$13C')
    AC.set_ylabel('height(m)')
    AC.set_ylim([0,120])
    AC.set_xlim([-10,6])
    
    BattleshipWash.d44ca=BattleshipWash.d44ca[BattleshipWash.d44ca<-.9]
    cmin=round(np.min([ArrowCanyon.d44ca.min(),BattleshipWash.d44ca.min()]),1)
    cmax=round(np.max([ArrowCanyon.d44ca.max(),BattleshipWash.d44ca.max()]),1)
    AC.scatter(BattleshipWash.d13c,BattleshipWash.SAMP_HEIGHT-240.0,s=10,color=[.7,.7,.7],edgecolor='none')
    cmax=-1.0
    d44ca=AC.scatter(ArrowCanyon.d13c,ArrowCanyon.SAMP_HEIGHT,c=ArrowCanyon.d44ca,cmap='Spectral',vmin=cmin,vmax=cmax,s=25,edgecolor=[.2,.2,.2])
    cbar=fig.colorbar(d44ca,ax=AC,label='$\delta$44Ca', orientation='vertical',pad=.05,shrink=.4,ticks=[ round(a, 1) for a in np.linspace(cmin,cmax,7)])
    d44ca=AC.scatter(BattleshipWash.d13c,BattleshipWash.SAMP_HEIGHT-240.0,c=BattleshipWash.d44ca,cmap=viridis,s=25,edgecolor=[.2,.2,.2],vmin=cmin,vmax=cmax) 
    


#%% 10300 injections for arrow canyon, now lets try leadville, AtR = 10**4.63
N=20
rmseLV=np.zeros(20)
mag=np.linspace(4,6,20)

for k in range(N):
    meshY=3
    meshX=101
    v=np.zeros([meshY,meshX])
    u=np.ones([meshY,meshX])
    aveSpeed=np.sqrt(v.mean()**2+u.mean()**2)
    AtR=10**(mag[k])
    injectionSites=np.zeros([meshY,meshX])
    injectionSites=injectionSites>0
    injectionSites[:,0]=True
    LVmesh=DiagenesisMesh.meshRock(meshX,meshY,u,v,-1.8,2,-1,aveSpeed/AtR,injectionSites) 
    N=200
    INJECTS=10300
    rmse=np.zeros(N)
    LVData=Leadville.d13c[Leadville.d13c<0]
    LVHeight=Leadville.SAMP_HEIGHT[Leadville.d13c<0]
    LVmesh.inject(int(9270/LVmesh.dt))
    rmseLV[k]=calcRMSE(LVmesh.printBox('rock','d13c')[1,1:],np.linspace(meshX-1,0,meshX-1),LVData,LVHeight)

with plt.style.context('fivethirtyeight'):
    plt.plot(mag,rmseLV) 
#%%
    
meshY=3
meshX=101
v=np.zeros([meshY,meshX])
u=np.ones([meshY,meshX])
aveSpeed=np.sqrt(v.mean()**2+u.mean()**2)
AtR=10**mag[2]  #10^4.63
injectionSites=np.zeros([meshY,meshX])
injectionSites=injectionSites>0
injectionSites[:,0]=True
LVmesh=DiagenesisMesh.meshRock(meshX,meshY,u,v,-1.8,2,-1,aveSpeed/AtR,injectionSites) 
INJECTS=10300
LVData=Leadville.d13c[Leadville.d13c<0]
LVHeight=Leadville.SAMP_HEIGHT[Leadville.d13c<0]
LVmesh.inject(10300)
#rmseLV[k]=calcRMSE(LVmesh.printBox('rock','d13c')[1,1:],np.linspace(meshX-1,0,meshX-1),LVData,LVHeight)
#%%
   
meshY=3
meshX=101
v=np.zeros([meshY,meshX])
u=np.ones([meshY,meshX])
aveSpeed=np.sqrt(v.mean()**2+u.mean()**2)
AtR=10**mag[19]  #10^4.63
injectionSites=np.zeros([meshY,meshX])
injectionSites=injectionSites>0
injectionSites[:,0]=True
LVmesh2=DiagenesisMesh.meshRock(meshX,meshY,u,v,2,2,-1,aveSpeed/AtR,injectionSites) 
LVData=Leadville.d13c[Leadville.d13c<0]
LVHeight=Leadville.SAMP_HEIGHT[Leadville.d13c<0]
LVmesh2.inject(int(600000)/LVmesh2.dt)
#rmseLV[k]=calcRMSE(LVmesh.printBox('rock','d13c')[1,1:],np.linspace(meshX-1,0,meshX-1),LVData,LVHeight)


with plt.style.context('ggplot'):
    rcParams.update({'figure.autolayout': True})
    fig = plt.figure(figsize=(6, 8))
    gs = gridspec.GridSpec(1, 1) 
    LdV = plt.subplot(gs[:,:])
    LdV.plot(LVmesh2.printBox('rock','d13c')[1,1:],np.linspace(meshX-1,0,meshX-1),lw=2,alpha=.66)
    LdV.scatter(LVData,LVHeight,s=10,color=[.7,.7,.7],edgecolor='none')
    LdV.set_xlabel('$\delta$13C')
    LdV.set_ylabel('height(m)')
    LdV.set_ylim([0,100])
    LdV.set_xlim([-10,6])
    

#%%
    
def calcRMSE(modelData,modelHeight,sampData,sampHeights):
    highResModel=scipy.interpolate.interp1d(modelHeight,modelData)
    rmse=np.sqrt((np.sum((highResModel(sampHeights)-sampData)**2))/len(sampData))
    return rmse
    