# -*- coding: utf-8 -*-
# cython: boundscheck=False

"""
Created on Wed Jun 17 11:41:59 2015

@author: bdyer
"""
import math
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from numpy import load,linspace,meshgrid
from matplotlib import animation
from pylab import rcParams
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap
import pickle
from copy import deepcopy
from matplotlib import rcParams
import pandas as pd
cimport cython
cimport numpy as np
from libc.stdlib cimport abs as c_abs

DTYPE = np.int
ctypedef np.int_t DTYPE_t
cdef inline int int_max(int a, int b): return a if a >= b else b
cdef inline int int_min(int a, int b): return a if a <= b else b
       
class meshRock:
    def __init__(self,meshX,meshY,u,v,d13c,d18o,d44ca,reactionRate,injectionSites,boundary=[-8.0,-8.0,-1.0,0.0],alpha=[1.0,1.0,1.0],massRatios=[[329.0,2.0],[1315.0,889.0],[1096.0,10.0]]):
        self.u=u
        self.v=v 
        self.shape=[meshY,meshX]
        self.initialBoundary=[d13c,d18o,d44ca]
        self.size=meshX*meshY
        self.boundary=boundary
        x=linspace(0,meshX,meshX)
        y=linspace(0,meshY,meshY)
        self.X,self.Y=meshgrid(x,y)
        self.rock=[[rock(d13c,d18o,d44ca) for j in range(meshX)] for i in range(meshY)]
        self.fluid=[[fluid(d13c,d18o,d44ca-(alpha[-1]-1)*10**3,u[i,j],v[i,j]) for j in range(meshX)] for i in range(meshY)]
        self.rockInit=deepcopy(self.rock)
        self.fluidInit=deepcopy(self.fluid)
        self.positiveX=np.array(self.u)>0
        self.positiveY=np.array(self.v)<0
        self.negativeX=np.array(self.u)<0
        self.negativeY=np.array(self.v)>0
        self.zeroSum=(np.array(self.v)+np.array(self.u)==0)
        self.flux=np.abs(self.v)+np.abs(self.u)
        self.maxF=np.max(self.flux)
        self.r=reactionRate
        self.injectionAge=0.0
        self.trueAge=0.0
        self.xSize=1.0 #m to box
        self.Js=np.ones((meshY,meshX))
        self.Is=np.ones((meshY,meshX))
        self.JsN=np.ones((meshY,meshX))
        self.IsN=np.ones((meshY,meshX))
        self.injectionSites=injectionSites
        self.alpha=alpha
        self.massRatio=massRatios
        self.d13c=d13c
        self.d18o=d18o
        self.d44ca=d44ca
        ySign,xSign=np.ones([2,self.shape[0],self.shape[1]])
        
        for i in range(self.shape[0]):
            for k in range(self.shape[1]):
                ySign[i,k]=-1*math.copysign(1,self.v[i,k])
                xSign[i,k]=-1*math.copysign(1,self.u[i,k])
        
        for i in range(1,meshY-1):
            for j in range(1,meshX-1):
                self.Js[i,j]=j+xSign[i,j]
                self.JsN[i,j]=j
                self.Is[i,j]=i+ySign[i,j]
                self.IsN[i,j]=i
        self.nt=1  #advection steps per reaction step  
        self.dt=.9/(np.abs(self.u).max()+np.abs(self.v).max()) # yr
        courant=np.abs(self.u).max()*self.dt+np.abs(self.v).max()*self.dt
        if courant>1.0:
            print(courant)
            print('Warning: advecting faster than timestep can resolve, decrease timestep (dt)')
        if ((self.dt*self.r)*massRatios[0][0])>massRatios[0][1]:
            print('Warning: reacting faster than timestep can resolve, decrease timestep (dt)')
    
        
        
        cm_data = pickle.load( open( "viridis.pkl", "rb" ) )
        self.viridis = LinearSegmentedColormap.from_list('viridis', cm_data)

    def compPlot(self):
        fig = plt.figure(figsize=(15, 12))
        self.compPlotAni(fig)
        plt.show()
        
    def hardReset(self):
        self.injectionAge=0.0
        self.trueAge=0.0
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                self.rock[i][j].reset()
                self.fluid[i][j].reset(self.alpha,self.initialBoundary[-1])
            
        
    def compPlotAni(self,fig):
        gs = gridspec.GridSpec(5, 3) 
        d13cRockPlot = plt.subplot(gs[0,:])
        linearPlot = plt.subplot(gs[1,:])
        fluidAge = plt.subplot(gs[2,:])
        fluidSpeed = plt.subplot(gs[3,:])
        crossPlotCOAge = plt.subplot(gs[4,0])
        crossPlotCOSpeed = plt.subplot(gs[4,1])
        ageC = plt.subplot(gs[4,2])
        text=d13cRockPlot.text(.0,.0,(str(int(self.trueAge))+' Years'),family='sans-serif')
        carbonAxes=[-6,6,-5,5]
        oxygenAxes=[-6,6,-5,5] #second set is for cmaps
        calciumAxes=[-1.6,-.8,-2,-1]
        with plt.style.context('ggplot'):
            
            speed = np.sqrt(self.u*self.u + self.v*self.v)
            lw = 1.0*speed/speed.max()
            xW=500;yW=80;
            x = np.linspace(0,xW,self.shape[1])
            y = np.linspace(0,yW,self.shape[0])
            X,Y = np.meshgrid(x,y)
            
            #d13cRockPlot.set_xlim([1,self.shape[1]-1])
            #d13cRockPlot.set_ylim([self.shape[0]-1,1])
            im = d13cRockPlot.imshow(self.printBox('rock','d13c'), cmap=self.viridis, vmin=carbonAxes[2], vmax=carbonAxes[3],aspect='auto',extent=[0,xW,yW,0])   
            fig.colorbar(im, ax=d13cRockPlot, label='$\delta$13C rock', orientation='vertical',pad=.0)
            qui = d13cRockPlot.streamplot(X, Y, self.u, self.v,color='k',linewidth=lw,density=.5)
            d13cRockPlot.grid(None)
            d13cRockPlot.axis('off')
            
            im = linearPlot.imshow(np.abs((self.printBox('rock','d13c')-self.printBox('rock','d18o'))),cmap='ocean_r',vmin=0,vmax=10,aspect='auto',extent=[0,xW,yW,0])   
            fig.colorbar(im, ax=linearPlot, label='d13c-d18o', orientation='vertical',pad=.0)
            qui = linearPlot.streamplot(X, Y, self.u, self.v,color='k',linewidth=lw,density=.5)
            linearPlot.grid(None)
            linearPlot.axis('off')            
            
            #fluidAge.set_xlim([1,self.shape[1]-1])
            #fluidAge.set_ylim([self.shape[0]-1,1])
            im3 = fluidAge.imshow((self.printBox('fluid','age')), cmap='Set1',aspect='auto',extent=[0,xW,yW,0])
            fig.colorbar(im3, ax=fluidAge, label='fluid age (years)', orientation='vertical',pad=.0)
            qui = fluidAge.streamplot(X, Y, (self.u), self.v,color='k',linewidth=lw,density=.5)
            fluidAge.grid(None)
            fluidAge.axis('off')
            
            #fluidSpeed.set_xlim([1,self.shape[1]-1])
            #fluidSpeed.set_ylim([self.shape[0]-1,1])
            im4 = fluidSpeed.imshow(speed, cmap='jet',aspect='auto',extent=[0,xW,yW,0])
            fig.colorbar(im4, ax=fluidSpeed, label='fluid speed (m/yr)', orientation='vertical',pad=.0)
            qui = fluidSpeed.streamplot(X, Y, (self.u), self.v,color='k',linewidth=lw,density=.5)
            fluidSpeed.grid(None)
            fluidSpeed.axis('off')
                       
            fluidA=self.printBox('fluid','age')
            rockCarbon=self.printBox('rock','d13c')
            rockOxygen=self.printBox('rock','d18o')
            fluidCarbon=self.printBox('fluid','d13c')
            fluidOxygen=self.printBox('fluid','d18o')
            
            crossPlotCOAge.set_xlim([oxygenAxes[0], oxygenAxes[1]])
            crossPlotCOAge.set_ylim([carbonAxes[0], carbonAxes[1]])   
            crossPlotCOAge.set_ylabel('$\delta$13C Rock', labelpad=-8)
            crossPlotCOAge.set_xlabel('$\delta$18O Rock', labelpad=0)
            crossRockCO2 = crossPlotCOAge.scatter(rockOxygen,rockCarbon,c=(fluidA), cmap='Set1', s=7,alpha=.8,edgecolors='none')

            crossPlotCOSpeed.set_xlim([oxygenAxes[0], oxygenAxes[1]])
            crossPlotCOSpeed.set_ylim([carbonAxes[0], carbonAxes[1]])   
            crossPlotCOSpeed.set_ylabel('$\delta$13C Rock', labelpad=-8)
            crossPlotCOSpeed.set_xlabel('$\delta$18O Rock', labelpad=0)
            colors=plt.cm.RdYlGn(np.linspace(0,1,self.shape[1]))
            for i in range(self.shape[1]):
                crossRockCO2 = crossPlotCOSpeed.scatter(rockOxygen[:,i],rockCarbon[:,i],color=colors[i], s=7,alpha=.8,edgecolors='none')

            ageC.set_xlim([0, fluidA.max()*1.1])
            ageC.set_xlim([carbonAxes[0], carbonAxes[1]])   
            ageC.set_ylabel('Fluid Age (yr)', labelpad=-8)
            ageC.set_xlabel('$\delta$13C Rock (O) and Fluid (G)', labelpad=0)
            rCvAge = ageC.scatter(rockCarbon,fluidA, s=15,alpha=.8, color=plt.rcParams['axes.color_cycle'][4])
            fCvAge = ageC.scatter(fluidCarbon,fluidA, s=15,alpha=.8, color=plt.rcParams['axes.color_cycle'][5])
            return fig
            
    def plot(self,phase,parameter):
        phase=getattr(self,phase)
        plt.imshow(np.array(list(reversed(
        [[getattr(phase[j][i],parameter) for j in range(self.shape[0])] for i in range(self.shape[1])]
        ))).T)                       
                  
    
    def printBox(self,phase,parameter):
        phase=getattr(self,phase)
        cdef DTYPE_t j,i
        cdef np.ndarray[np.float64_t,ndim=2] box = np.ones((self.shape[0],self.shape[1]))
        for j in xrange(self.shape[0]):
            for i in xrange(self.shape[1]):
                box[j,i]=getattr(phase[j][i],parameter)
        return box
        
    def inject(self,steps):            
        def reactBox(rockDelta,fluidDelta,rr,fr,alpha):           
            rockMass=1.0*(rr)
            porosity=.1
            fluidMass=1.0*fr*porosity
            fIn=(self.r)*self.dt*rockMass     #fixed reaction rate per box       
            fOut=fIn
            reactingDelta=fluidDelta+(np.array(alpha)-1.0)*10*10*10
            rockDelta1=((fIn*reactingDelta)-(fOut*rockDelta))/rockMass
            fluidDelta1=-1*((fIn*reactingDelta)-(fOut*rockDelta))/fluidMass
            return [rockDelta1,fluidDelta1]
            
        cdef DTYPE_t j,n,row,column
        cdef np.ndarray[np.float64_t, ndim=2] newFluid, newRock, dRock, dFluid, onesShape
        delta=['d13c','d18o','d44ca','age'] #properties to track
        boundary=self.boundary #BOUNDARY CONDITIONS
        massRatio=self.massRatio #stiochiometric ratio between elements (r,f) (Ca, 1-37)
        alpha=self.alpha
        nt = self.nt 
        nx=self.shape[1]
        ny=self.shape[0]
        u = np.abs(self.u)
        v = np.abs(self.v)
        loops=len(delta)
        
        for j in xrange(steps):
            def AdvectionStep(delta,boundary): 
                def timeSavingFunction(dt,cXn, cXi,cYi, v, u): #march forward euler upwinding
                    return (cXn - v * dt/1.0 * (cXn - cYi) - u * dt/1.0 * (cXn - cXi))
                    
#                def centralDifference(cX,Is,Js,IsN,JsN): #central diff -- higher accuracy, slower, not implemented yet
#                    cXn = cX.copy()  
#                    cX2 = cX.copy()
#                    cX2[1:-1,1:-1]=timeSavingFunction(tStep,cXn[1:-1,1:-1],cXn[IsN[1:-1,1:-1],Js[1:-1,1:-1]],cXn[Is[1:-1,1:-1],JsN[1:-1,1:-1]],v[Is[1:-1,1:-1],JsN[1:-1,1:-1]],u[IsN[1:-1,1:-1],Js[1:-1,1:-1]])
#                    cXn = cX2.copy()
#                    cX3 = cX3.copy()
#                    cX3[1:-1,1:-1]=timeSavingFunction(tStep,cXn[1:-1,1:-1],cXn[IsN[1:-1,1:-1],Js[1:-1,1:-1]],cXn[Is[1:-1,1:-1],JsN[1:-1,1:-1]],v[Is[1:-1,1:-1],JsN[1:-1,1:-1]],u[IsN[1:-1,1:-1],Js[1:-1,1:-1]])
#                    return (cX3+cX2)/2.0
#                
#                def TVDRungeKutta(dt,cXn, cXi,cYi, v, u): #stable and accurate, not yet implemented
#                    cXi1=(cXn - v * dt/1.0 * (cXn - cYi) - u * dt/1.0 * (cXn - cXi))
#                    cXi2=(cXi1 - v * dt/1.0 * (cXi1 - cYi) - u * dt/1.0 * (cXi1 - cXi))
#                    return (cXn - v * dt/1.0 * (cXn - cYi) - u * dt/1.0 * (cXn - cXi))
                    
                cdef np.ndarray[np.float_t, ndim=3] cX=np.zeros([loops,ny,nx]) 
                tStep=self.dt
                IS=self.injectionSites
                for k in xrange(len(delta)):
                    cX[k][:,:]=self.printBox('fluid',delta[k])    
                    for n in xrange(nt): 
                        cX[k][IS]=boundary[k]
                        #cX[k][-1,2:-1:2]=boundary[k]#BOUNDARY CONDITIONS
                        cXn = cX[k].copy()
                        cX[k][1:-1,1:-1]=timeSavingFunction(tStep,cXn[1:-1,1:-1],cXn[list(self.IsN[1:-1,1:-1]),list(self.Js[1:-1,1:-1])],cXn[list(self.Is[1:-1,1:-1]),list(self.JsN[1:-1,1:-1])],(v[list(self.Is[1:-1,1:-1]),list(self.JsN[1:-1,1:-1])]),(u[list(self.IsN[1:-1,1:-1]),list(self.Js[1:-1,1:-1])]))    
                return cX
            
            self.injectionAge=self.injectionAge+1
            self.trueAge=self.trueAge+self.dt


            cX=AdvectionStep(delta,boundary) 

            for k in xrange(loops-1):
                for row in xrange(1,ny-1):
                    for column in xrange(1,nx-1):
                        setattr(self.fluid[row][column],delta[k],cX[k][row,column])    

            for k in xrange(loops-1):
                R=  self.printBox('rock',delta[k])
                F= self.printBox('fluid',delta[k]) 
                onesShape=np.ones((self.shape[0],self.shape[1]))
                dRock,dFluid=reactBox(R,F,
                                      onesShape*massRatio[k][0], #rock mass 1m^3
                                      onesShape*massRatio[k][1], #fluid mass 1m^3
                                      onesShape*alpha[k])
                newRock=R+dRock
                newFluid=F+dFluid                        
                for row in xrange(1,ny-1):
                    for column in xrange(1,nx-1):
                        setattr(self.rock[row][column],delta[k],newRock[row,column])
                        setattr(self.fluid[row][column],delta[k],newFluid[row,column])    
                        if k==0 and ~self.zeroSum[row][column]:
                            self.fluid[row][column].age=self.dt+cX[-1][row][column]

                for row in xrange(0,ny):
                    setattr(self.fluid[row][0],delta[k],getattr(self.fluid[row][1],delta[k]))
                    setattr(self.fluid[row][-1],delta[k],getattr(self.fluid[row][-2],delta[k]))
                    setattr(self.rock[row][0],delta[k],getattr(self.rock[row][1],delta[k]))
                    setattr(self.rock[row][-1],delta[k],getattr(self.rock[row][-2],delta[k]))
                    
                    

                for column in xrange(0,nx):
                    setattr(self.fluid[0][column],delta[k],getattr(self.fluid[1][column],delta[k]))
                    setattr(self.fluid[-1][column],delta[k],getattr(self.fluid[-2][column],delta[k]))
                    setattr(self.rock[0][column],delta[k],getattr(self.rock[1][column],delta[k]))
                    setattr(self.rock[-1][column],delta[k],getattr(self.rock[-2][column],delta[k]))


            for column in xrange(0,nx):
                setattr(self.fluid[0][column],'age',getattr(self.fluid[1][column],'age'))
                setattr(self.fluid[-1][column],'age',getattr(self.fluid[-2][column],'age'))
            for row in xrange(0,ny):
                setattr(self.fluid[row][0],'age',getattr(self.fluid[row][1],'age'))
                setattr(self.fluid[row][-1],'age',getattr(self.fluid[row][-2],'age'))

class rock:
    def __init__(self, d13c, d18o, d44ca):
        self.d13c = d13c
        self.d18o = d18o
        self.d44ca = d44ca
        self.fluxed = 0.0
        self.d13cI = d13c
        self.d18oI = d18o
        self.d44caI = d44ca
        
    def reset(self):
        self.d13c = self.d13cI
        self.d18o = self.d18oI
        self.d44ca = self.d44caI
        self.fluxed = 0.0
        
class fluid:
    def __init__(self, d13c, d18o, d44ca, u, v):
        self.d13c = d13c
        self.d18o = d18o
        self.d44ca = d44ca
        self.d13cI = d13c
        self.d18oI = d18o
        self.d44caI = d44ca
        self.age = 0.0
        self.u = u
        self.v = v
        self.flux=(np.abs(u)+np.abs(v))  
    
    def reset(self,alpha,CaInit):
        self.d13c = self.d13cI
        self.d18o = self.d18oI
        self.d44ca = CaInit-(alpha[-1]-1)*10**3
        self.age = 0.0

