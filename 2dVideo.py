import scipy.integrate as integrate
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from numpy import *

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
    r=M[4]/2
    fIn=r*m
    fOut=fIn
    if mr>0:
        dr[0]=fIn-fOut
        dr[1]=((fIn*dF)-(fOut*dR))/mr
        dr[2]=-1*((fIn*dF)-(fOut*dR))/mf
        fIn=r*m*4
        fOut=fIn
        dr[3]=((fIn*dFo)-(fOut*dRo))/mrO
        dr[4]=-1*((fIn*dFo)-(fOut*dRo))/mfO
        dr[5]=0
    else:
        dr[0],dr[1],dr[2],dr[3],dr[4],dr[5],=0
         
    return dr[1:]

def rockBox(d13cInit,d13cI,d18oInit,d18oI,flux):
    t=np.linspace(0, 1, 2)
    z = integrate.odeint(rock, [d13cInit,d13cI,d18oInit,d18oI,flux], t)
    dR,dF,dRo,dFo,flux = z.T
    d13cP=dR[-1:]
    d18oP=dRo[-1:]
    d13cFlF=dF[-1:]
    d18oFlF=dFo[-1:]
    flux=flux[:-1]
    #d13cP,d18oP,d13cFlF,d18oFlF,flux=rockBox(d13cP,d18oP,d13cFlF,d18oFlF,flux)
    return d13cP,d13cFlF,d18oP,d18oFlF,flux

fig, axarr = plt.subplots(1,2)
axarr[0].invert_yaxis()
a,b,c,d,e = np.zeros([5,20,20])
a,b=np.ones([2,20,20])*5
c,d=np.ones([2,20,20])*1
im = axarr[0].imshow(a[3:,3:10], cmap=plt.get_cmap('coolwarm'), vmin=-5, vmax=5)
axarr[1].set_xlim([-7, 6])
axarr[1].set_ylim([0, 16])
axarr[0].set_xlim([0, 6])
axarr[0].set_ylim([16, 0])
line, = axarr[1].plot([], [], lw=2, label='$\delta$13C')
line2, = axarr[1].plot([], [], lw=2, label='$\delta$18O')
fig.colorbar(im, ax=axarr[0], label='$\delta$13C', orientation='horizontal')
a[0,0],b[0,0],c[0,0],d[0,0],e[0,0],=[5,-5,1,-6,10]

# calculate advection field u and v
u,v = np.ones([2,20,20])
u=load('u.npy')
v=load('v.npy')
u[0:2,:]=0
v[0:2,:]=0

x=linspace(0,19,20)
X,Y=meshgrid(x,x)
u=np.array(list(reversed(u)))
v=np.array(list(reversed(v)))
qui = axarr[0].quiver(X[:-1-3,0:7],Y[:-1-3,0:7],u[3:,3:10],v[3:,3:10])
axarr[0].plot(5*np.ones(c[3:,5].size),np.linspace(0,16,c[3:,5].size), lw=2)
v=np.abs(v)*(10/5)
u=np.abs(u)*(10/5)
axarr[1].set_ylabel('meters')
axarr[1].set_xlabel('$\delta$ value')
axarr[1].set_title('stratigraphic section')
#axarr[1].legend(handles=[line, line2], loc=3)
axarr[0].set_title('2d mesh with fluid flow')
axarr[0].axes.get_xaxis().set_visible(False)
axarr[0].axes.get_yaxis().set_visible(False)
    
#runfile('/Users/bdyer/Desktop/LPIA/Talks, Reports, and Posters/MeteoricPaper/Ca Paper/2dAdvection.py', wdir='/Users/bdyer/Desktop/LPIA/Talks, Reports, and Posters/MeteoricPaper/Ca Paper')
def updatefig(frame):
    global a,b,c,d,e
#    for j in range(1,frame):
    aP=a
    cP=c
    
    for k in range(1,20):
         #box to store new values
        b[0:5,:]=-5 #boundary conditions
        d[0:5,:]=-6        
        flux=np.array(u[k,1])+.01
        for i in range(1,20):
            flux=np.array(u[k,i])+.01 #quick fix for no 0 flux
            aP[k,i],b[k,i],cP[k,i],d[k,i],e[k,i],=rockBox(aP[k,i],b[k,i-1],cP[k,i],d[k,i-1],flux)
            
            flux=np.array(v[k,i])+.01 #quick fix for no 0 flux
            aP[k,i],b[k,i],cP[k,i],d[k,i],e[k,i],=rockBox(aP[k,i],b[k-1,i],cP[k,i],d[k-1,i],flux)
            
            
    a=aP        #update the boxes
    c=cP
    a2=a[3:,3:10]
    c2=c[3:,3:10]
    im.set_array(a2)
    line.set_data(np.array(a2[:,5]),np.array(list(reversed(range(0,a2[:,4].size)))))
    line2.set_data(np.array(c2[:,5]),np.array(list(reversed(range(0,c2[:,4].size)))))
    return im, line, line2

ani = animation.FuncAnimation(fig, updatefig, frames=100)

FFwriter = animation.FFMpegWriter()
ani.save('fluidFlowBH.mp4', writer = FFwriter, fps=30, extra_args=['-vcodec', 'libx264'])
#plt.show()