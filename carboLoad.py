# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 14:49:04 2015

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
import copy
with plt.style.context('ggplot'):
    
    m1=AC.d13c
    m2=AC.age
    
    
    xmin = m1.min()
    xmax = m1.max()
    ymin = m2.min()
    ymax = m2.max()
    
    
    
    X, Y = np.mgrid[xmin:xmax:400j, ymin:ymax:400j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([m1, m2])
    kernel = stats.gaussian_kde(values)
    kernel.set_bandwidth(.1)
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
    ax.imshow((np.rot90(Z2)), cmap='seismic', extent=[ymin,ymax,xmin,xmax], aspect=1*10**0,vmin=-1.0*Z2.max(),vmax=Z2.max()*1.0)
    #fig.savefig('GaussianHist2d.pdf', format='pdf', dpi=100)
    
    
    plt.show()