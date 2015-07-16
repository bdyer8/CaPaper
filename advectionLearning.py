# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 09:09:13 2015

@author: bdyer
"""
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import time
import math
import copy
import DiagenesisMesh

###variable declarations


#nt = 400
nt=100
nx=100
ny=100
dx = 1
dy = 1
dt = .01
meshX=u.shape[1]
meshY=u.shape[0]
mesh=DiagenesisMesh.meshRock(meshX,meshY,u,v,5,5,-1,1e-6)
u=mesh.u
v=mesh.v
#cX=mesh.printBox('fluid','d13c')
copy=copy.copy
abso=np.abs
ySign,xSign=np.ones([2,mesh.shape[0],mesh.shape[1]])
test=np.ones((100,100))
for i in range(mesh.shape[0]):
    for k in range(mesh.shape[1]):
        ySign[i,k]=-1*math.copysign(1,v[i,k])
        xSign[i,k]=-1*math.copysign(1,u[i,k])

def timeSavingFunction(cX,cXi,cYi,v,u):
    cXn = copy(cX)    
    cX=   (   cXn
                        - (abso(v) * dt/dx * (cXn - cYi))
                        - (abso(u) * dt/dy * (cXn - cXi))
          )
    return cX

Js=np.ones((ny,nx))
Is=np.ones((ny,nx))
JsN=np.ones((ny,nx))
IsN=np.ones((ny,nx))

for i in range(1,ny-1):
    for j in range(1,nx-1):
        Js[i,j]=j+xSign[i,j]
        JsN[i,j]=j
        Is[i,j]=i+ySign[i,j]
        IsN[i,j]=i

    

 
time=time.time
def anim(t):
    for n in range(nt+1): ##loop across number of time steps
        cX[-3:-1,:]=-2.0
        cXn = cX.copy()
        cX[1:-1,1:-1]=np.array(map(timeSavingFunction,cXn[1:-1,1:-1],cXn[list(IsN[1:-1,1:-1]),list(Js[1:-1,1:-1])],cXn[list(Is[1:-1,1:-1]),list(JsN[1:-1,1:-1])],v[1:-1,1:-1],u[1:-1,1:-1]))    
    
    #fig = plt.figure(figsize=(15, 6))
    gs = gridspec.GridSpec(1,1) 
    ax=plt.subplot(gs[0,0])   
    im=ax.imshow(cX, cmap='nipy_spectral_r',vmin=-2,vmax=2.5,extent=[0,100,100,0],interpolation='none')
    im2=ax.imshow(np.flipud(img),extent=[0,100,100,0],alpha=.2)
    fig.colorbar(im, ax=ax)
    ax.set_xlim([0,nx])
    ax.set_ylim([ny,0])
    ax.axis('off')
    speed = np.sqrt(u*u + v*v)
    Y, X = np.mgrid[0:ny, 0:nx]
    qui = ax.streamplot(X, Y, u, v, density=.9, color=[.9,.9,.9], linewidth=10*speed/speed.max())
    


#fig = plt.figure(figsize=(10, 10))
#ani = animation.FuncAnimation(fig, anim, frames=500)
#FFwriter = animation.FFMpegWriter()
#ani.save('lens_flow_example3.mp4', dpi=150, writer = FFwriter, fps=30, extra_args=['-vcodec', 'libx264'])
cX=np.ones((100,100))*2.0
anim(cX)
