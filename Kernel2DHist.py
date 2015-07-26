# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 14:50:49 2015

@author: bdyer
"""

import pickle
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from numpy import load,linspace,meshgrid
from matplotlib import animation
from pylab import rcParams
from matplotlib import gridspec
from matplotlib import rcParams
import pandas as pd
from scipy import stats
from scipy.stats import norm
with plt.style.context('ggplot'):
    
    delta2=load('mcDelta.npy')
    plat2=load('mcPlatArea.npy')
    delta=delta2[(delta2>0) & (delta2<5.0)]
    plat=plat2[(delta2>0) & (delta2<5.0)]
    m1=delta
    m2=plat
    
    
    xmin = m1.min()
    xmax = m1.max()
    ymin = m2.min()
    ymax = m2.max()
    
    
    
    X, Y = np.mgrid[xmin:xmax:150j, ymin:ymax:150j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([m1, m2])
    kernel = stats.gaussian_kde(values)
    kernel.set_bandwidth(.075)
    #
    Z = np.reshape(kernel.evaluate(positions).T, X.shape)
    
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    
    #ax.imshow(np.rot90(Z), cmap='Reds',
    #           extent=[xmin, xmax, ymin, ymax])
    #ax.plot(m1, m2, 'k.', markersize=2)
    Z2=copy.copy(Z)
    for i in range(Z.T.shape[0]):
        Z2[i,:]=Z.T[i,:]/np.max(Z.T[i,:])
    ax.imshow((np.rot90(Z2)), cmap='seismic', extent=[ymin,ymax,xmin,xmax], aspect=1*10**7,vmin=-1.0*Z2.max(),vmax=Z2.max()*1.0)
    fig.savefig('GaussianHist2d.pdf', format='pdf', dpi=100)
    
    
    plt.show()
    
    #%%
    fig = plt.figure(figsize=(10,10))
    plt.plot(X[:,129],Z[:,129]/(integrate.trapz(Z[:,129],dx=5.0/150)))
    fig.savefig('postHIST.pdf', format='pdf', dpi=100)
    
    #%%
    fig = plt.figure(figsize=(20,4))
    gs = gridspec.GridSpec(1,8) 
    react = plt.subplot(gs[0,0:2])
    timePlot = plt.subplot(gs[0,2:4])
    carbonateW = plt.subplot(gs[0,4:6])
    depthRange = plt.subplot(gs[0,6:])
    n=10000
    react.hist(.13+np.abs(np.random.normal(0,.025,n)),100)
    timePlot.hist((np.random.normal(2,.5,n)),100)
    carbonateW.hist(uniform(0,2.0,n),100)
    depthRange.hist((np.random.normal(55,15,n)),100)
    plt.show()