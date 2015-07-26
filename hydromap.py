# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 16:28:14 2015

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
from matplotlib.colors import LinearSegmentedColormap
cm_data = pickle.load( open( "viridis.pkl", "rb" ) )
viridis = LinearSegmentedColormap.from_list('viridis', list(reversed(cm_data)))

#%%
ArrowCanyon=pd.read_csv('samples_ArrowCanyon.csv')
B503=pd.read_csv('samples_B503.csv')
B211=pd.read_csv('samples_B211.csv')
SanAndres=pd.read_csv('samples_SanAndres.csv')
Leadville=pd.read_csv('samples_B416.csv')
Leadville2=pd.read_csv('samples_B417.csv')
BattleshipWash=pd.read_csv('samples_BattleshipWash.csv')
B402=pd.read_csv('samples_B402.csv')
db = pm.database.pickle.load('6Sections_Diagenesis_ACFIT.pickle')

#%%
y=np.array([ArrowCanyon.Lat[0],
    Leadville.Lat[0],
    Leadville2.Lat[0],
    B503.Lat[0],
    B211.Lat[0],
    SanAndres.Lat[0]])
    
x=np.array([ArrowCanyon.Lon[0],
    Leadville.Lon[0],
    Leadville2.Lon[0],
    B503.Lon[0],
    B211.Lon[0],
    SanAndres.Lon[0]])
    
z=np.array([db.trace('velocity1').stats()['quantiles'][50],
    db.trace('velocity2').stats()['quantiles'][50],
    db.trace('velocity3').stats()['quantiles'][50],
    db.trace('velocity4').stats()['quantiles'][50],
    db.trace('velocity6').stats()['quantiles'][50],
    db.trace('velocity7').stats()['quantiles'][50]])
    
#x, y, z = 10 * np.random.random((3,10))
fig=plt.figure(figsize=(8,8))
# Set up a regular grid of interpolation points
xi, yi = np.linspace(x.min()-5, x.max()+5, 100), np.linspace(y.min()-5, y.max()+5, 100)
xi, yi = np.meshgrid(xi, yi)

# Interpolate
rbf = scipy.interpolate.Rbf(x, y, z, function='linear')
zi = rbf(xi, yi)

plt.imshow(zi, vmin=z.min(), vmax=z.max(), origin='lower',
           extent=[xi.min(), xi.max(), yi.min(), yi.max()],cmap=viridis,aspect='auto')
plt.colorbar()
plt.scatter(x, y, c=z,cmap=viridis)

plt.show()

fig.savefig(('hydromap.pdf'), format='pdf', dpi=300)  