
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


u=load('uWideLens2.npy')[0:-1:5,0:250:5]
v=load('vWideLens2.npy')[0:-1:5,0:250:5]
meshX=u.shape[1]
meshY=u.shape[0]
u=u[:,:]*1.0/32.0
v=v[:,:]*-1.0/32.0
mesh=DiagenesisMesh.meshRock(meshX,meshY,np.array(list(reversed(u))),np.array(list(reversed(v))),2.0,-1,-1,1.0e-05)  #.01 = 1% per timestep~Ma

years=10000
#temp=time.time()
mesh.inject(int(years/mesh.dt))
#print(time.time()-temp)
mesh.compPlot()
#%%
def aniStep(step):
    mesh.inject(1)
    mesh.compPlotAni(fig)


#fig = plt.figure(figsize=(20, 16))
#ani = animation.FuncAnimation(fig, aniStep, frames=500)
#FFwriter = animation.FFMpegWriter()
#ani.save('compPlot_500_realFastAdvect.mp4', dpi=150, writer = FFwriter, fps=30, extra_args=['-vcodec', 'libx264'])
#meshX=4
#meshY=60
#u=0*np.ones([meshY,meshX])/4.15  #50m in 100 yr  .25=.5 m/yr
#v=np.ones([meshY,meshX])/50.0  #.02 is 50m in 100 yr for 10m grid
#mesh=DiagenesisMesh.meshRock(meshX,meshY,np.array(list(reversed(u))),np.array(list(reversed(v))),2.0,-1,-1,1.0e-05)  #R = % rock per yr
#mesh.inject(1000)
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