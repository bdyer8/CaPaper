
"""
Created on Wed Jul  1 13:09:06 2015

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

#reload(DiagenesisMesh)
     

u=load('uLensSeaWater.npy')[0:-1,0:250]
v=load('vLensSeaWater.npy')[0:-1,0:250]
meshX=u.shape[1]
meshY=u.shape[0]
u=u[:,:]*1.0
v=v[:,:]*-1.0
mesh=DiagenesisMesh.meshRock(meshX,meshY,np.array(list(reversed(u))),np.array(list(reversed(v))),2.0,-1,-1,1.0e-07)  #.01 = 1% per timestep~Ma
#%%
years=10
#temp=time.time()
mesh.inject(int(years/mesh.dt))
#print(time.time()-temp)
mesh.compPlot()

def aniStep(step):
    mesh.inject(1)
    mesh.compPlotAni(fig)


#fig = plt.figure(figsize=(20, 16))
#ani = animation.FuncAnimation(fig, aniStep, frames=500)
#FFwriter = animation.FFMpegWriter()
#ani.save('compPlot_500_realFastAdvect.mp4', dpi=150, writer = FFwriter, fps=30, extra_args=['-vcodec', 'libx264'])
#meshX=60
#meshY=4
#u=np.ones([meshY,meshX])/4.15/100  #50m in 100 yr  .25=.5 m/yr
#v=0*np.ones([meshY,meshX])
#mesh=DiagenesisMesh.meshRock(meshX,meshY,np.array(list(reversed(u))),np.array(list(reversed(v))),2.0,-1,-1,1.0e-05)  #R = % rock per yr
#years=.1
#mesh.inject(int(years/mesh.dt))

#mesh.compPlot()

#
#fig = plt.figure(figsize=(20, 16))
#ani = animation.FuncAnimation(fig, aniStep, frames=300)
#FFwriter = animation.FFMpegWriter()
#ani.save('compPlot_300_lens_rAdvect_r01.mp4', dpi=200, writer = FFwriter, fps=30, extra_args=['-vcodec', 'libx264'])



#
#fig = plt.figure(figsize=(20, 16))
#mesh.inject(1)
#mesh.compPlotAni(fig)
#fig.savefig('ArrowCanyon_real2.pdf', format='pdf', dpi=300)


#%%
#
#A=mesh.printBox('fluid','age')
#rO=mesh.printBox('rock','d18o')
#rC=mesh.printBox('rock','d13c')
#plt.scatter(rO,rC,c=A,cmap='gist_stern_r',s=7,alpha=.8,edgecolors='none',vmin=0, vmax=A.max()*1.1);plt.colorbar()