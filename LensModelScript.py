# -*- coding: utf-8 -*-
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

reload(DiagenesisMesh)
     
meshX=500
meshY=80
u=load('uLensSeaWater.npy')
v=load('vLensSeaWater.npy')
u=u[:,:]
v=v[:,:]*-1.0
mesh=DiagenesisMesh.meshRock(meshX,meshY,np.array(list(reversed(u))),np.array(list(reversed(v))),2.0,-1,-1,.01)  #.01 = 1% per timestep~Ma

def aniStep(step):
    mesh.inject(1)
    mesh.compPlotAni(fig)


#fig = plt.figure(figsize=(20, 16))
#ani = animation.FuncAnimation(fig, aniStep, frames=500)
#FFwriter = animation.FFMpegWriter()
#ani.save('compPlot_500_realFastAdvect.mp4', dpi=150, writer = FFwriter, fps=30, extra_args=['-vcodec', 'libx264'])
#meshX=50
#meshY=20
#u=np.ones([20,50])*-1
#v=np.ones([20,50])+(range(1,meshX+1)*np.ones([20,50]))/meshX-1.1
#mesh=DiagenesisMesh.meshRock(meshX,meshY,np.array(list(reversed(u))),np.array(list(reversed(v))),2.0,-1,-1,.05)  #.01 = 1% per timestep~Ma
#mesh.inject(10)
#mesh.compPlot()
#
fig = plt.figure(figsize=(20, 16))
ani = animation.FuncAnimation(fig, aniStep, frames=300)
FFwriter = animation.FFMpegWriter()
ani.save('compPlot_300_lens_rAdvect_r01.mp4', dpi=200, writer = FFwriter, fps=30, extra_args=['-vcodec', 'libx264'])



#
#fig = plt.figure(figsize=(20, 16))
#mesh.inject(1)
#mesh.compPlotAni(fig)
fig.savefig('ArrowCanyon_real2.pdf', format='pdf', dpi=300)
