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
u=u[:,:]*100
v=v[:,:]*-100
mesh=DiagenesisMesh.meshRock(meshX,meshY,np.array(list(reversed(u))),np.array(list(reversed(v))),2.0,-1,-1,.025)  #.01 = 1% per timestep~Ma

def aniStep(step):
    mesh.inject(1)
    mesh.compPlotAni(fig)


fig = plt.figure(figsize=(20, 16))
ani = animation.FuncAnimation(fig, aniStep, frames=500)
FFwriter = animation.FFMpegWriter()
ani.save('compPlot_500_fixedA_fixedR_age_realAdvect.mp4', dpi=150, writer = FFwriter, fps=30, extra_args=['-vcodec', 'libx264'])
#meshX=50
#meshY=20
#u=np.ones([20,50])*-10.0*1
#v=np.ones([20,50])*10.0*1*(range(1,meshX+1)*np.ones([20,50]))/25.0
#mesh=DiagenesisMesh.meshRock(meshX,meshY,np.array(list(reversed(u))),np.array(list(reversed(v))),2.0,-1,-1,.25)  #.01 = 1% per timestep~Ma

#mesh.inject(20)
#mesh.compPlot()
#
#fig = plt.figure(figsize=(20, 16))
#mesh.inject(1)
#mesh.compPlotAni(fig)
#fig.savefig('ArrowCanyon.pdf', format='pdf', dpi=100)
