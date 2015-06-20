import scipy.integrate as integrate
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from numpy import *
from pylab import rcParams
from matplotlib import gridspec

rcParams['figure.figsize'] = 15, 12
platformC=2
meteoricFC=-5
def rock(M,t):
    dR=M[0]
    dRo=M[2]
    dF=M[1]
    dFo=M[3]
    mr=1
    mrO=mr*4
    mf=M[4] #flux of fluid mass relative to rock per reaction step (r)
    mfO=mf*444 #number from 2000ppm carbon and 889000 ppm oxygen  
    m=mr+mf
    dr=np.zeros(6)
    r=.0005  #reaction rate
    fIn=r*mf
    fOut=fIn
    if mr>0:
        dr[0]=fIn-fOut
        dr[1]=((fIn*dF)-(fOut*dR))/mr
        dr[2]=-1*((fIn*dF)-(fOut*dR))/mf
        fIn=r*mf*4
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

gridX = 250
gridY = 250
#fig, axarr = plt.subplots(1,2)

fig = plt.figure(figsize=(15, 12)) 
gs = gridspec.GridSpec(1, 2, width_ratios=[5, 1]) 
axarr0 = plt.subplot(gs[0])
axarr1 = plt.subplot(gs[1])
a,b,c,d,e = np.zeros([5,gridX,gridY])
a,b=np.ones([2,gridX,gridY])*platformC
c,d=np.ones([2,gridX,gridY])*1
im = axarr0.imshow(a, cmap=plt.get_cmap('nipy_spectral'), vmin=-6, vmax=2)
#im = axarr0.imshow(a, cmap=plt.get_cmap('prism'), vmin=-5, vmax=5)
axarr1.set_xlim([-10, 10])
axarr1.set_ylim([100, 240])
axarr0.set_xlim([125, 248])
axarr0.set_ylim([140, 10])
line, = axarr1.plot([], [], lw=2, label='$\delta$13C')
line2, = axarr1.plot([], [], lw=2, label='$\delta$18O')
line3, = axarr1.plot([], [], lw=2, label='$\delta$13C')
line4, = axarr1.plot([], [], lw=2, label='$\delta$18O')
fig.colorbar(im, ax=axarr0, label='$\delta$13C', orientation='horizontal')
a[0,0],b[0,0],c[0,0],d[0,0],e[0,0],=[5,-5,1,-6,10]

# calculate advection field u and v
u,v = np.ones([2,gridX,gridY])
#u=load('u.npy')
#v=load('v.npy')
#u[0:2,:]=0
#v[0:2,:]=0
#u=load('uNew2.npy')
#u[np.abs(u)<.001]=0
#v=load('vNew2.npy')
#v[np.abs(v)<.001]=0
u=load('uLensBig.npy')
#u[np.abs(u)<.001]=0
v=load('vLensBig.npy')
#v[np.abs(v)<.001]=0
x=linspace(0,gridX-1,gridX)
X,Y=meshgrid(x,x)
u=np.array(list(reversed(u)))
v=np.array(list(reversed(v)))
#qui = axarr[0].quiver(X[:-1-3,0:7],Y[:-1-3,0:7],u[3:,3:10],v[3:,3:10])
everyN=5
qui = axarr0.quiver(X[::everyN,::everyN],Y[::everyN,::everyN],u[::everyN,::everyN],v[::everyN,::everyN])
axarr0.plot(135*np.ones(c[3:,25].size),np.linspace(0,gridX,c[3:,5].size), lw=2)
axarr0.plot(175*np.ones(c[3:,25].size),np.linspace(0,gridX,c[3:,5].size), lw=2)
axarr0.plot(200*np.ones(c[3:,25].size),np.linspace(0,gridX,c[3:,5].size), lw=2)
axarr0.plot(240*np.ones(c[3:,25].size),np.linspace(0,gridX,c[3:,5].size), lw=2)
#axarr[0].plot(5*np.ones(c[3:,5].size),np.linspace(0,16,c[3:,5].size), lw=2)
#v=np.abs(v)
#u=np.abs(u)
u=u*100
v=v*100
#u[:,:]=0
axarr1.set_ylabel('meters')
axarr1.set_xlabel('$\delta$ value')
axarr1.set_title('stratigraphic section')
#axarr[1].legend(handles=[line, line2], loc=3)
axarr0.set_title('2d mesh with fluid flow + diagenesis')
axarr0.axes.get_xaxis().set_visible(False)
axarr0.axes.get_yaxis().set_visible(False)
global maskSW
maskSW=load('mask250SW.npy')
mask250=load('mask250.npy')
maskSW=mask250-maskSW

def newFluid(xDelta,xFlux,yDelta,yFlux):
    xDelta=float(xDelta)
    fluid=((xDelta*xFlux+yDelta*yFlux)/(xFlux+yFlux))
    return fluid
    
def updatefig(frame):
    global a,b,c,d,e
    aP=a
    bP=b
    bPx=np.zeros([b.shape[0],b.shape[1]])
    bPy=np.zeros([b.shape[0],b.shape[1]])
    cP=c
    dP=d
    dPx=np.zeros([b.shape[0],b.shape[1]])
    dPy=np.zeros([b.shape[0],b.shape[1]])
    
    b[0:2,:]=meteoricFC                     
    d[0:2,:]=-6
    #b[maskSW[:,:,0]]=platformC
    #d[maskSW[:,:,0]]=1
    print(1)
    for k in range((gridX-2),1,-1):  #k is y and i is x, v is y and u is x, positive right and down
        for i in range(gridX-2,1,-1): #we must loop through the whole matrix and gather the fluids from 4 directions             
            F=[0,0,0]
            xFlux=np.array(u[k,i]) 
            if xFlux>0:
                F[1]=xFlux
                bPx[k,i]=b[k,i-1]
                dPx[k,i]=d[k,i-1]
            else:
                F[1]=xFlux
                bPx[k,i]=b[k,i+1]
                dPx[k,i]=d[k,i+1]
                
            yFlux=np.array(v[k,i]) 
            if yFlux>0:
                F[2]=yFlux
                bPy[k,i]=b[k+1,i]
                dPy[k,i]=d[k+1,i]
                 
            else:
                F[2]=yFlux
                bPy[k,i]=b[k-1,i]
                dPy[k,i]=d[k-1,i]
            
            if ((xFlux+yFlux)==0):
                bP[k,i]=a[k,i]
                dP[k,i]=c[k,i]
                aP[k,i]=a[k,i]
                cP[k,i]=c[k,i]
            else:
                bP[k,i]=newFluid(bPx[k,i],np.abs(F[1]),bPy[k,i],np.abs(F[2]))
                dP[k,i]=newFluid(dPx[k,i],np.abs(F[1]),dPy[k,i],np.abs(F[2]))
                aP[k,i],bP[k,i],cP[k,i],dP[k,i],_=rockBox(a[k,i],bP[k,i],c[k,i],dP[k,i],(np.abs(F[1])+np.abs(F[2]))) #react step
                
    a=aP        #update the boxes
    b=bP[:,:]  
    c=cP
    d=dP[:,:]
    
    im.set_array(a)
    line.set_data(np.array(a[:,135]),np.array(list(reversed(range(0,a[:,4].size)))))
    line2.set_data(np.array(a[:,175]),np.array(list(reversed(range(0,c[:,4].size)))))
    line3.set_data(np.array(a[:,200]),np.array(list(reversed(range(0,a[:,4].size)))))
    line4.set_data(np.array(a[:,240]),np.array(list(reversed(range(0,c[:,4].size)))))
    return im, line, line2, line3, line4

#ani = animation.FuncAnimation(fig, updatefig, frames=250)
for i in range(0,10):
    updatefig(1)
#FFwriter = animation.FFMpegWriter()
#ani.save('fluidFlowCarboniferousSW4.mp4', writer = FFwriter, fps=30, extra_args=['-vcodec', 'libx264'])
#fig.savefig('MeteoricCarbon2.pdf', format='pdf')
plt.show()
