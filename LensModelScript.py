
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
from matplotlib.colors import LinearSegmentedColormap

u=load('uWideLens2.npy')
v=load('vWideLens2.npy')
v=-1*np.flipud(v)
u=np.flipud(u)
u=u[:,250:-20]*.043 #5 boxes = 50 meters, tuned to vertical injection of .5m/yr
v=v[:,250:-20]*.043

meshX=u.shape[1]
meshY=u.shape[0]
injectionSites=np.zeros((meshY,meshX))
injectionSites=injectionSites>0
injectionSites[0:4,:]=True

aveSpeed=np.sqrt(v.mean()**2+u.mean()**2)
AtR=1e5 #12000 seems good?
mesh=DiagenesisMesh.meshRock(meshX,meshY,u,v,5,5,-1,aveSpeed/AtR,injectionSites)  
#mesh.inject(int(3000/mesh.dt))
#mesh.compPlot()
#
def aniStep(step):
    mesh.compPlotAni(fig)    
    mesh.inject(2000)
##
fig = plt.figure(figsize=(20, 16))
ani = animation.FuncAnimation(fig, aniStep, frames=100)
FFwriter = animation.FFMpegWriter()
ani.save('LensTest_AtR_1e5.mp4', dpi=300, writer = FFwriter, fps=30, extra_args=['-vcodec', 'libx264'])
#
fig.savefig('LensTest_AtR_1e5.pdf', format='pdf', dpi=300)
#
def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

save_object(mesh, '2dFluidLensMesh1e5.pkl')