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
     
#meshX=500
#meshY=80
#u=load('uWideLens2.npy')
#v=load('vWideLens2.npy')
#u=u[:,:]*.1
#v=v[:,:]*.1
#mesh=DiagenesisMesh.meshRock(meshX,meshY,np.array(list(reversed(u))),np.array(list(reversed(v))),2.0,-1,-1,.025)  #.01 = 1% per timestep~Ma

def aniStep(step):
    mesh.inject(1)
    mesh.compPlotAni(fig)


#fig = plt.figure(figsize=(20, 16))
#ani = animation.FuncAnimation(fig, aniStep, frames=300)
#FFwriter = animation.FFMpegWriter()
#ani.save('compPlot_1000_fixedA_fixedR_age.mp4', dpi=150, writer = FFwriter, fps=30, extra_args=['-vcodec', 'libx264'])
meshX=50
meshY=20
u=np.ones([20,50])*1.0*.1
v=-1*np.ones([20,50])*1.0*.1
mesh=DiagenesisMesh.meshRock(meshX,meshY,np.array(list(reversed(u))),np.array(list(reversed(v))),2.0,-1,-1,.025)  #.01 = 1% per timestep~Ma

mesh.inject(100)
mesh.compPlot()
#
#fig = plt.figure(figsize=(20, 16))
#mesh.inject(1)
#mesh.compPlotAni(fig)
#fig.savefig('ArrowCanyon.pdf', format='pdf', dpi=100)
