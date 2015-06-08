# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 20:00:49 2015

@author: bdyer
"""

import scipy.integrate as integrate
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation


def rock(M,t):
    dR=M[0]
    dRo=M[2]
    dF=M[1]
    dFo=M[3]
    mr=1
    mrO=mr*4
    mf=M[4] #flux of fluid mass relative to rock per reaction step (r)
    mfO=mf*10000  
    m=mr+mf
    dr=np.zeros(6)
    r=.001
    fIn=r*m
    fOut=fIn
    dr[0]=fIn-fOut
    dr[1]=((fIn*dF)-(fOut*dR))/mr
    dr[2]=-1*((fIn*dF)-(fOut*dR))/mf
    fIn=r*m*4
    fOut=fIn
    dr[3]=((fIn*dFo)-(fOut*dRo))/mrO
    dr[4]=-1*((fIn*dFo)-(fOut*dRo))/mfO
    dr[5]=0
    return dr[1:]
    
    
dinit = 5
t= np.linspace(0, 1e3, 2000)
z = integrate.odeint(rock, [5,-5,1,-5,10], t)
dR,dF,dRo,dFo,flux = z.T

#fig, ax = plt.subplots()
#ax.plot(t,dR)
#ax.plot(t,dF)
#ax.plot(t,dRo)
#ax.plot(t,dFo)
#
#plt.grid(True)
#plt.show()

#%%
t= np.linspace(0, 1, 2)
iters=int(1e2)
rockV=np.ones(iters)*5
rockVo=np.ones(iters)*1
rockV[0]=5
drS=np.zeros(iters)
drS[0]=-5

fig = plt.figure()
ax = plt.axes(xlim=(-7, 6), ylim=(0, iters))
plt.gca().invert_yaxis()
plt.grid(True)
line, = ax.plot([], [], lw=2)
line2, = ax.plot([], [], lw=2)
line3, = ax.plot([], [], lw=2)
line4, = ax.plot([], [], lw=2)



# initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    line2.set_data([], [])
    line3.set_data([], [])
    line4.set_data([], [])
    
    return line, line2, line3, line4

# animation function.  This is called sequentially
def animate(i):
    drS=np.zeros(iters*2)
    drSo=np.zeros(iters*2)
    drS[0]=-5
    drSo[0]=-5.5
    for i in range(1,5+int(np.random.exponential(1)*int(iters/15))):
        z = integrate.odeint(rock, [rockV[i],drS[i-1],rockVo[i],drSo[i-1],10], t)
        dR,dF,dRo,dFo,flux = z.T
        rockV[i]=dR[-1:]
        rockVo[i]=dRo[-1:]
        drS[i]=dF[-1:]
        drSo[i]=dFo[-1:]
            
            
    x = range(1,iters)
    y = rockV[1:]
    y2= rockVo[1:]
    z = drS[1:]
    z = z[z!=0]
    zx = range(0,z.size)
    z2 = drSo[1:]
    z2 = z2[z2!=0]
    zx2 = range(0,z2.size)
    #y = drS[1:]
    line.set_data(y,x)
    line2.set_data(z,zx)
    line3.set_data(y2,x)
    line4.set_data(z2,zx2)
    return line, line2, line3, line4

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=50, interval=1, blit=True, repeat_delay=10)

# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html
#anim.save('basic_animation.mp4', fps=3)

plt.show()

#
#fig, ax = plt.subplots()
#ax.plot(rockV[1:],range(1,iters))
#ax.plot(drS[1:],range(1,iters))
#
#
#plt.gca().invert_yaxis()
#plt.grid(True)
#plt.show()

# %%

#fig, ax = plt.subplots()
#plt.gca().invert_yaxis()
#ax.plot(rockV[1:],range(1,rockV.size))
#plt.grid(True)
#plt.show()


import matplotlib as mpl

# make values from -5 to 5, for this example
rockVdata=np.array(list(reversed(rockV[1:])))
zvals = (rockVdata*np.ones([100,rockV[1:].size])).T

# make a color map of fixed colors
fig = plt.figure(2)
cmap2 = mpl.colors.LinearSegmentedColormap.from_list('my_colormap',
                                           ['green','blue'],
                                           256)
img2 = plt.imshow(zvals,interpolation='nearest',
                    cmap = cmap2,
                    origin='lower')
plt.colorbar(img2,cmap=cmap2)
plt.show()
