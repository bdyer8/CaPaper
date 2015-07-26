import scipy.integrate as integrate
import matplotlib.pyplot as plt
import numpy as np
from numpy import load,linspace,meshgrid
from matplotlib import animation
from pylab import rcParams
from matplotlib import gridspec
import pickle



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
    r=.001  #reaction rate
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


  
rcParams['figure.figsize'] = 15, 12
platformC=2
meteoricFC=-7

gridX = 500
gridY = 80
#fig, axarr = plt.subplots(1,2)

fig = plt.figure(figsize=(15, 12)) 
gs = gridspec.GridSpec(1, 2, width_ratios=[5, 1]) 
axarr0 = plt.subplot(gs[0])
axarr1 = plt.subplot(gs[1])
a,b,c,d,e = np.zeros([5,gridY,gridX])
a,b=np.ones([2,gridY,gridX])*platformC
c,d=np.ones([2,gridY,gridX])*1


fp = open('cmap.pkl', 'rb')
cmapDiag = pickle.load(fp)
fp.close()

im = axarr0.imshow(a, cmap=cmapDiag, vmin=-10, vmax=2)
#im = axarr0.imshow(a, cmap=plt.get_cmap('prism'), vmin=-5, vmax=5)
axarr1.set_xlim([-10, 3])
axarr1.set_ylim([0, 80])
axarr0.set_xlim([5, 495])
axarr0.set_ylim([75, 5])
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
u=load('uWide.npy')
#u[np.abs(u)<.001]=0
v=load('vWide.npy')
#v[np.abs(v)<.001]=0
x=linspace(0,500,gridX)
y=linspace(0,80,gridY)
X,Y=meshgrid(x,y)
u=np.array(list(reversed(u)))
v=np.array(list(reversed(v)))
#qui = axarr[0].quiver(X[:-1-3,0:7],Y[:-1-3,0:7],u[3:,3:10],v[3:,3:10])
everyNX=5
#qui = axarr0.quiver(X,Y,u,v)
qui = axarr0.quiver(
                    X[::everyNX,::everyNX],Y[::everyNX,::everyNX],
                    u[::everyNX,::everyNX],v[::everyNX,::everyNX],
                    width=0.0012, scale=1/2.0
                    )
axarr0.plot(200*np.ones(c[3:,25].size),np.linspace(0,gridX,c[3:,5].size), lw=2)
axarr0.plot(220*np.ones(c[3:,25].size),np.linspace(0,gridX,c[3:,5].size), lw=2)
axarr0.plot(240*np.ones(c[3:,25].size),np.linspace(0,gridX,c[3:,5].size), lw=2)
axarr0.plot(245*np.ones(c[3:,25].size),np.linspace(0,gridX,c[3:,5].size), lw=2)
#axarr[0].plot(5*np.ones(c[3:,5].size),np.linspace(0,16,c[3:,5].size), lw=2)
#v=np.abs(v)
#u=np.abs(u)
u=u*1000
v=v*1000
#u[:,:]=0
axarr1.set_ylabel('meters')
axarr1.set_xlabel('$\delta$ value')
axarr1.set_title('stratigraphic section')
#axarr[1].legend(handles=[line, line2], loc=3)
axarr0.set_title('2d mesh with fluid flow + diagenesis')
axarr0.axes.get_xaxis().set_visible(False)
axarr0.axes.get_yaxis().set_visible(False)
#global maskSW
#maskSW=load('mask250SW.npy')
#mask250=load('mask250.npy')
#maskSW=mask250-maskSW
global positiveX,positiveY,negativeX,negativeY,zeroSum
positiveX=np.array(u)>0
positiveY=np.array(v)>0
negativeX=np.array(u)<0
negativeY=np.array(v)<0
zeroSum=(np.array(v)+np.array(x)==0)

def newFluid(xDelta,xFlux,yDelta,yFlux):    
    fluid=1.0*((1.0*xDelta*xFlux+yDelta*yFlux)/(1.0*xFlux+yFlux))
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
    
    b[0:6,:]=-7                     
    d[0:6,:]=-6
    #b[maskSW[:,:,0]]=platformC
    #d[maskSW[:,:,0]]=1
    bPx[positiveX]=np.roll(b,1,axis=1)[positiveX]  
    bPx[negativeX]=np.roll(b,-1,axis=1)[negativeX]  
    bPy[positiveY]=np.roll(b,-1,axis=0)[positiveY]  
    bPy[negativeY]=np.roll(b,1,axis=0)[negativeY] 
    
    dPx[positiveX]=np.roll(d,1,axis=1)[positiveX]  
    dPx[negativeX]=np.roll(d,-1,axis=1)[negativeX]  
    dPy[positiveY]=np.roll(d,-1,axis=0)[positiveY]  
    dPy[negativeY]=np.roll(d,1,axis=0)[negativeY] 
    
    bP[zeroSum]=a[zeroSum]
    dP[zeroSum]=c[zeroSum]
    aP[zeroSum]=a[zeroSum]
    cP[zeroSum]=c[zeroSum]
            
    #bP[~zeroSum]=newFluid(bPx[~zeroSum],np.abs(u[~zeroSum]),bPy[~zeroSum],np.abs(v[~zeroSum]))
    #dP[~zeroSum]=newFluid(dPx[~zeroSum],np.abs(u[~zeroSum]),dPy[~zeroSum],np.abs(v[~zeroSum]))
    F=(np.abs(u)+np.abs(v))
    
    for k in range(gridY-2,1,-1):  #k is y and i is x, v is y and u is x, positive right and down
        for i in range(gridX-2,1,-1):
            if ((F[k,i])!=0):
                bP[k,i]=newFluid(bPx[k,i],np.abs(u[k,i]),bPy[k,i],np.abs(v[k,i]))
                dP[k,i]=newFluid(dPx[k,i],np.abs(u[k,i]),dPy[k,i],np.abs(v[k,i]))
                aP[k,i],bP[k,i],cP[k,i],dP[k,i],_=rockBox(a[k,i],bP[k,i],c[k,i],dP[k,i],F[k,i])
    
    a=aP        #update the boxes
    b=bP  
    c=cP
    d=dP
    
    im.set_array(a)
    line.set_data(np.array(a[:,200]),np.array(list(reversed(range(0,a[:,4].size)))))
    line2.set_data(np.array(a[:,220]),np.array(list(reversed(range(0,c[:,4].size)))))
    line3.set_data(np.array(a[:,240]),np.array(list(reversed(range(0,a[:,4].size)))))
    line4.set_data(np.array(a[:,245]),np.array(list(reversed(range(0,c[:,4].size)))))
    return im, line, line2, line3, line4

ani = animation.FuncAnimation(fig, updatefig, frames=15)
#for i in range(0,2):
#    updatefig(1)
FFwriter = animation.FFMpegWriter()
ani.save('fluidFlowCarboniferousSW4.mp4', writer = FFwriter, fps=30, extra_args=['-vcodec', 'libx264'])
fig.savefig('MeteoricCarbon4.pdf', format='pdf')
plt.show()