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
    return d13cP,d18oP,d13cFlF,d18oFlF,flux

fig = plt.figure()
a,b,c,d,e = np.zeros([5,20,20])
a=np.ones([20,20])*5
im = plt.imshow(a, cmap=plt.get_cmap('jet'), vmin=-5, vmax=5)
plt.colorbar(im)
a[0,0],b[0,0],c[0,0],d[0,0],e[0,0],=rockBox(5,-5,1,-5.5,10)

def updatefig(frame):
#    for j in range(1,frame):
    for k in range(1,20):
        a[k,0],b[k,0],c[k,0],d[k,0],e[k,0],=rockBox(a[k,0],b[k,0],c[k-1,0],d[k-1,0],e[k-1,0])
        
        for i in range(1,20):
            a[k,i],b[k,i],c[k,i],d[k,i],e[k,i],=rockBox(a[k,i],b[k,i],c[k,i-1],d[k,i-1],e[k,i-1])
            e[k,i]=e[k,i]
            
            
    a2=a[1:,:]
    im.set_array(a2)
    return im

ani = animation.FuncAnimation(fig, updatefig, frames=1, 
                              interval=50)
plt.show()