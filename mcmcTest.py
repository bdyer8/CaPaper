# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 22:59:31 2015

@author: bdyer
"""

from pylab import rcParams
from matplotlib import gridspec
import matplotlib.pyplot as plt
import pymc as pm
import numpy as np
import scipy
import pandas as pd
import pickle
import scipy.interpolate
from pymc.Matplot import plot
import pydot
#%%
err=pm.Uniform('err',0,500)
m=pm.Uniform('slope',-100,100,size=(1000,2),value=np.random.normal(0,1,(1000,2)), observed=True)
b=pm.Uniform('offset',-100,100,size=2)
xData=np.random.uniform(0,1.0,(1000,2))
mR=[3.23,6.75]
bR=[-23.4,12.2]
yData=mR*xData+bR+np.random.normal(0,3,(1000,2))

x=pm.Normal('x',0,1,value=xData,observed=True, size=(1000,2))

@pm.deterministic()
def predY(m=m,b=b,x=x):
    return m*x+b
    
y=pm.Normal('y',predY,err,value=yData,observed=True, size=(1000,2))

model=pm.Model([m,b,x,predY,y])

mcmc=pm.MCMC(model)
#%%
mcmc.sample(10000,0,2)

#%%
for i in range(2):
    colors=['r','b']
    plt.plot(np.sort(x.value[:,i]),sorted(mcmc.trace('predY').stats()['quantiles'][50][:,i]),color=colors[i],lw=2);
    plt.scatter(xData[:,i],yData[:,i],color=colors[i]);
    plt.plot(np.sort(x.value[:,i]),sorted(mcmc.trace('predY').stats()['quantiles'][2.5][:,i]),color=colors[i],lw=2,alpha=.6);
    plt.plot(np.sort(x.value[:,i]),sorted(mcmc.trace('predY').stats()['quantiles'][97.5][:,i]),color=colors[i],lw=2,alpha=.6);