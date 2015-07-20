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
    d44ca=AC.scatter(ArrowCanyon.d13c,ArrowCanyon.SAMP_HEIGHT,c=ArrowCanyon.d44ca,cmap='Spectral',vmin=cmin,vmax=cmax,s=25,edgecolor=[.2,.2,.2])
    cbar=fig.colorbar(d44ca,ax=AC,label='$\delta$44Ca', orientation='vertical',pad=.05,shrink=.4,ticks=[ round(a, 1) for a in np.linspace(cmin,cmax,7)])
    d44ca=AC.scatter(BattleshipWash.d13c,BattleshipWash.SAMP_HEIGHT-240.0,c=BattleshipWash.d44ca,cmap=viridis,s=25,edgecolor=[.2,.2,.2],vmin=cmin,vmax=cmax)
    #cbar=fig.colorbar(d44ca,ax=AC,label='$\delta$44Ca', orientation='vertical',pad=.01,shrink=.25,ticks=[ round(a, 1) for a in np.linspace(cmin,cmax,7)])
    fig.savefig('CaData.pdf', format='pdf', dpi=300)
    
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
    cmin=round(np.min([ArrowCanyon.d18o.min(),BattleshipWash.d18o.min()]),1)
    cmax=round(np.max([ArrowCanyon.d18o.max(),BattleshipWash.d18o.max()]),1)
    cmax=-4.0
    cmin=-8.0
    AC.scatter(BattleshipWash.d13c,BattleshipWash.SAMP_HEIGHT-240.0,s=10,color=[.7,.7,.7],edgecolor='none')
    d44ca=AC.scatter(ArrowCanyon.d13c,ArrowCanyon.SAMP_HEIGHT,c=ArrowCanyon.d18o,cmap=viridis,vmin=cmin,vmax=cmax,s=25,edgecolor=[.2,.2,.2])
    cbar=fig.colorbar(d44ca,ax=AC,label='$\delta$18O', orientation='vertical',pad=.05,shrink=.4,ticks=[ round(a, 1) for a in np.linspace(cmin,cmax,7)])
    d44ca=AC.scatter(BattleshipWash.d13c,BattleshipWash.SAMP_HEIGHT-240.0,c=BattleshipWash.d18o,cmap=viridis,s=25,edgecolor=[.2,.2,.2],vmin=cmin,vmax=cmax)
    #cbar=fig.colorbar(d44ca,ax=AC,label='$\delta$44Ca', orientation='vertical',pad=.01,shrink=.25,ticks=[ round(a, 1) for a in np.linspace(cmin,cmax,7)])
    fig.savefig('OData.pdf', format='pdf', dpi=600)
    
    