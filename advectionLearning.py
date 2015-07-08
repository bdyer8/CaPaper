# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 09:09:13 2015

@author: bdyer
"""
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


###variable declarations


nt = 400
nx=500
ny=80
dx = 1
dy = 1
dt = .02

u = mesh.u
v = mesh.v
#cX=mesh.printBox('fluid','d13c')
cX=np.ones((80,500))*2.0
abso=np.abs
ySign,xSign=np.ones([2,mesh.shape[0],mesh.shape[1]])
for i in range(mesh.shape[0]):
    for k in range(mesh.shape[1]):
        ySign[i,k]=-1*math.copysign(1,v[i,k])
        xSign[i,k]=-1*math.copysign(1,u[i,k])


for n in range(nt+1): ##loop across number of time steps
    cX[0:5,1:-1:10]=-2
    cXn = cX.copy()
    for i in range(1,ny-1):
        for j in range(1,nx-1):
            cX[i,j]=   (   cXn[i,j]
                        - (abso(v[i,j]) * dt/dx * (cXn[i,j] - cXn[i+ySign[i,j],j]))
                        - (abso(u[i,j]) * dt/dy * (cXn[i,j] - cXn[i,j+xSign[i,j]]))
                       )


fig = plt.figure(figsize=(15, 6))
gs = gridspec.GridSpec(1,1) 
ax=plt.subplot(gs[0,0])   
im=ax.imshow(cX, cmap='nipy_spectral_r',vmin=-2,vmax=2.5)
plt.colorbar(im)
ax.set_xlim([0,nx])
ax.set_ylim([ny,0])

Y, X = np.mgrid[0:ny, 0:nx]
qui = ax.streamplot(X, Y, u, v, density=.8, color=[.9,.9,.9], linewidth=1)



##%%
#def getMatrix(cXn,cYn,u,v,nx,ny):
#testX=np.ones((20,50))
#testY=np.ones((20,50))
#for i in range(20):
#    for k in range(50):
#        test[i,k]=math.copysign(1,u[i,k])
#matrix=np.zeros([1,3,20,50])  #parameters, sum+2d, meshX, meshY         
#for row in range(0,20):
#    for column in range(0,50):
#        for k in range(1):  #k through trackable properties, d13c, d18o, etc
#            if testX[row,column]:
#                matrix[k][1][row][column]=cXn[row][column-1]
#            else:
#                matrix[k][1][row][column]=cXn[row][column+1]
#            if testY[row,column]:
#                matrix[k][2][row][column]=cYn[row-1][column]
#            else:
#                matrix[k][2][row][column]=cYn[row+1][column]