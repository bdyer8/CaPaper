# -*- coding: utf-8 -*-
# cython: boundscheck=False

"""
Created on Wed Jun 17 11:41:59 2015
2-d mesh solver for fluid rock interactions

to cythonize:  python setup.py build_ext --inplace

author: Blake Dyer: blake.c.dyer@gmail.com
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
    def __init__(self,meshX,meshY,u,v,d13c,d18o,d44ca,reactionRate,injectionSites,
                 track_props=['d13c','d18o','d44ca','age','salinity'],boundary=[-8.0,-8.0,-1.0,0.0,0.0],
                 alpha=[1.0,1.0,1.0],massRatios=[[329.0,2.0],[1315.0,889.0],[1096.0,10.0]],
                 tidalsites=None,seaboundary=[2.0,2.0,0.0,0.0,35.0]):

        self.u=u
        self.v=v
        self.seaboundary=seaboundary
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
        self.track=track_props
        RP=np.array(track_props)
        self.react_props=RP[np.where((RP!='age')&(RP!='salinity'))]
        ySign,xSign=np.ones([2,self.shape[0],self.shape[1]])
        if tidalsites is None:
            self.tidalsites=np.zeros([meshY,meshX],dtype=bool)
        else:
            self.tidalsites=tidalsites
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

    def hardReset(self):
        self.injectionAge=0.0
        self.trueAge=0.0
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                self.rock[i][j].reset()
                self.fluid[i][j].reset(self.alpha,self.initialBoundary[-1])
             
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
        delta=self.track #properties to track
        boundary=self.boundary #BOUNDARY CONDITIONS
        massRatio=self.massRatio #stiochiometric ratio between elements (r,f) (Ca, 1-37)
        alpha=self.alpha
        nt = self.nt 
        nx=self.shape[1]
        ny=self.shape[0]
        u = np.abs(self.u)
        v = np.abs(self.v)
        
        for j in xrange(steps):
            def AdvectionStep(delta,boundary): 
                def timeSavingFunction(dt,cXn, cXi,cYi, v, u): #march forward euler upwinding
                    return (cXn - v * dt/1.0 * (cXn - cYi) - u * dt/1.0 * (cXn - cXi))
                    
                cdef np.ndarray[np.float_t, ndim=3] cX=np.zeros([len(delta),ny,nx]) 
                tStep=self.dt
                IS=self.injectionSites
                TS=self.tidalsites
                for k in xrange(len(delta)):
                    cX[k][:,:]=self.printBox('fluid',delta[k])    
                    for n in xrange(nt): 
                        cX[k][IS]=boundary[k]
                        if delta[k] != 'age':
                            cX[k][TS]=self.seaboundary[k]
                        cXn = cX[k].copy()
                        cX[k][1:-1,1:-1]=timeSavingFunction(tStep,cXn[1:-1,1:-1],cXn[list(self.IsN[1:-1,1:-1]),
                                                            list(self.Js[1:-1,1:-1])],cXn[list(self.Is[1:-1,1:-1]),
                                                            list(self.JsN[1:-1,1:-1])],(v[list(self.Is[1:-1,1:-1]),
                                                            list(self.JsN[1:-1,1:-1])]),(u[list(self.IsN[1:-1,1:-1]),
                                                            list(self.Js[1:-1,1:-1])]))    
                return cX
            
            self.injectionAge=self.injectionAge+1
            self.trueAge=self.trueAge+self.dt
            cX=AdvectionStep(delta,boundary) 

            for k,prop in enumerate(delta):
                for row in xrange(1,ny-1):
                    for column in xrange(1,nx-1):
                        if prop == 'age':
                            setattr(self.fluid[row][column],prop,cX[k][row,column]+self.dt)    
                        else:
                            setattr(self.fluid[row][column],prop,cX[k][row,column])    
            for k,prop in enumerate(self.react_props):
                R=self.printBox('rock',prop)
                F=self.printBox('fluid',prop) 
                onesShape=np.ones((self.shape[0],self.shape[1]))
                dRock,dFluid=reactBox(R,F,
                                      onesShape*massRatio[k][0], #rock mass 1m^3
                                      onesShape*massRatio[k][1], #fluid mass 1m^3
                                      onesShape*alpha[k])
                newRock=R+dRock
                newFluid=F+dFluid                        
                for row in xrange(1,ny-1):
                    for column in xrange(1,nx-1):
                        setattr(self.rock[row][column],prop,newRock[row,column])
                        setattr(self.fluid[row][column],prop,newFluid[row,column])    
                        #if k==0 and ~self.zeroSum[row][column]:
                        #    self.fluid[row][column].age=self.dt+cX[-1][row][column]

                for row in xrange(0,ny):
                    setattr(self.fluid[row][0],prop,getattr(self.fluid[row][1],prop))
                    setattr(self.fluid[row][-1],prop,getattr(self.fluid[row][-2],prop))
                    setattr(self.rock[row][0],prop,getattr(self.rock[row][1],prop))
                    setattr(self.rock[row][-1],prop,getattr(self.rock[row][-2],prop))
                    
                    

                for column in xrange(0,nx):
                    setattr(self.fluid[0][column],prop,getattr(self.fluid[1][column],prop))
                    setattr(self.fluid[-1][column],prop,getattr(self.fluid[-2][column],prop))
                    setattr(self.rock[0][column],prop,getattr(self.rock[1][column],prop))
                    setattr(self.rock[-1][column],prop,getattr(self.rock[-2][column],prop))

            if 'age' in self.track:
                for column in xrange(0,nx):
                    setattr(self.fluid[0][column],'age',getattr(self.fluid[1][column],'age'))
                    setattr(self.fluid[-1][column],'age',getattr(self.fluid[-2][column],'age'))
                for row in xrange(0,ny):
                    setattr(self.fluid[row][0],'age',getattr(self.fluid[row][1],'age'))
                    setattr(self.fluid[row][-1],'age',getattr(self.fluid[row][-2],'age'))
           
            if 'salinity' in self.track:
                for column in xrange(0,nx):
                    setattr(self.fluid[0][column],'salinity',getattr(self.fluid[1][column],'salinity'))
                    setattr(self.fluid[-1][column],'salinity',getattr(self.fluid[-2][column],'salinity'))
                for row in xrange(0,ny):
                    setattr(self.fluid[row][0],'salinity',getattr(self.fluid[row][1],'salinity'))
                    setattr(self.fluid[row][-1],'salinity',getattr(self.fluid[row][-2],'salinity'))
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
        self.salinity = 35.0
        self.flux=(np.abs(u)+np.abs(v))  
    
    def reset(self,alpha,CaInit):
        self.d13c = self.d13cI
        self.d18o = self.d18oI
        self.d44ca = CaInit-(alpha[-1]-1)*10**3
        self.age = 0.0
        self.salinity = 35.0

