# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 11:41:59 2015

@author: bdyer
"""

import scipy.integrate as integrate
import matplotlib.pyplot as plt
import numpy as np
from numpy import load,linspace,meshgrid
from matplotlib import animation
from pylab import rcParams
from matplotlib import gridspec
import pickle
        
        
class meshRock:
    def __init__(self,meshX,meshY,u,v,d13c,d18o,reactionRate):
        self.u=u
        self.v=v 
        self.shape=[meshY,meshX]
        self.size=meshX*meshY
        x=linspace(0,meshX,meshX)
        y=linspace(0,meshY,meshY)
        self.X,self.Y=meshgrid(x,y)
        self.rock=[[rock(d13c,d18o) for j in range(meshX)] for i in range(meshY)]
        self.fluid=[[fluid(d13c,d18o,u[i,j],v[i,j]) for j in range(meshX)] for i in range(meshY)]
        self.positiveX=np.array(self.u)>0
        self.positiveY=np.array(self.v)<0
        self.negativeX=np.array(self.u)<0
        self.negativeY=np.array(self.v)>0
        self.zeroSum=(np.array(self.v)+np.array(self.u)==0)
        self.flux=np.abs(self.v)+np.abs(self.u)
        self.r=reactionRate
    
    def compPlot(self):
        fig = plt.figure(figsize=(15, 12))
        gs = gridspec.GridSpec(4, 4) 
        d13cRockPlot = plt.subplot(gs[0,0:3])
        d13cFluidPlot = plt.subplot(gs[1,0:3])
        rockFluxed = plt.subplot(gs[2,0:3])
        covarPlot = plt.subplot(gs[3,0:3])
        stratPlot = plt.subplot(gs[:,3])
        
        sections=[35,400]
        everyNX=5

        
        fp = open('cmap.pkl', 'rb')
        cmapDiag = pickle.load(fp)
        fp.close()
        
        d13cRockPlot.set_xlim([1,self.shape[1]-1])
        d13cRockPlot.set_ylim([self.shape[0]-1,1])
        im = d13cRockPlot.imshow(self.printBox('rock','d13c'), cmap=cmapDiag) # , vmin=-10, vmax=2  
        fig.colorbar(im, ax=d13cRockPlot, label='$\delta$13C rock', orientation='horizontal',shrink=.5,pad=.2)
        d13cRockPlot.plot(sections[0]*np.ones(100),np.linspace(0,self.shape[1],100), lw=1)
        d13cRockPlot.plot(sections[1]*np.ones(100),np.linspace(0,self.shape[1],100), lw=1)
        qui = d13cRockPlot.quiver(
                    self.X[::everyNX,::everyNX],self.Y[::everyNX,::everyNX],
                    np.array(list((self.u[::everyNX,::everyNX]))),np.array(list((self.v[::everyNX,::everyNX]))),
                    width=0.0005
                    )
        
                    
        
        d13cFluidPlot.set_xlim([1,self.shape[1]-1])
        d13cFluidPlot.set_ylim([self.shape[0]-1,1])
        im2 = d13cFluidPlot.imshow(self.printBox('fluid','d13c'), cmap=cmapDiag) # , vmin=-10, vmax=2  
        fig.colorbar(im2, ax=d13cFluidPlot, label='$\delta$13C fluid', orientation='horizontal',shrink=.5,pad=.2)
        d13cFluidPlot.plot(sections[0]*np.ones(100),np.linspace(0,self.shape[1],100), lw=1)
        d13cFluidPlot.plot(sections[1]*np.ones(100),np.linspace(0,self.shape[1],100), lw=1)
        qui = d13cFluidPlot.quiver(
                    self.X[::everyNX,::everyNX],self.Y[::everyNX,::everyNX],
                    np.array(list((self.u[::everyNX,::everyNX]))),np.array(list((self.v[::everyNX,::everyNX]))),
                    width=0.0005
                    )
        
        rockFluxed.set_xlim([1,self.shape[1]-1])
        rockFluxed.set_ylim([self.shape[0]-1,1])
        im3 = rockFluxed.imshow(self.printBox('rock','fluxed'), cmap='jet')
        fig.colorbar(im3, ax=rockFluxed, label='total fluid flux (mass C in Fluid to mass C in Rock)', orientation='horizontal',shrink=.5,pad=.2)
        rockFluxed.plot(sections[0]*np.ones(100),np.linspace(0,self.shape[1],100), lw=1)
        rockFluxed.plot(sections[1]*np.ones(100),np.linspace(0,self.shape[1],100), lw=1)
        qui = rockFluxed.quiver(
                    self.X[::everyNX,::everyNX],self.Y[::everyNX,::everyNX],
                    np.array(list((self.u[::everyNX,::everyNX]))),np.array(list((self.v[::everyNX,::everyNX]))),
                    width=0.0005
                    )
        
        covarPlot.set_xlim([1,self.shape[1]-1])
        covarPlot.set_ylim([self.shape[0]-1,1])
        im4 = covarPlot.imshow(np.array(self.printBox('rock','d13c')/self.printBox('rock','d18o')), cmap='jet')
        fig.colorbar(im4, ax=covarPlot, label='$\delta$13C/$\delta$18O rock', orientation='horizontal',shrink=.5,pad=.2)
        covarPlot.plot(sections[0]*np.ones(100),np.linspace(0,self.shape[1],100), lw=1)
        covarPlot.plot(sections[1]*np.ones(100),np.linspace(0,self.shape[1],100), lw=1)
        qui = covarPlot.quiver(
                    self.X[::everyNX,::everyNX],self.Y[::everyNX,::everyNX],
                    np.array(list((self.u[::everyNX,::everyNX]))),np.array(list((self.v[::everyNX,::everyNX]))),
                    width=0.0005
                    )
        
        stratPlot.set_xlim([-10, 3])
        stratPlot.set_ylim([self.shape[0]-2,2])   
        rockCarbon=self.printBox('rock','d13c')
        strat = stratPlot.plot(rockCarbon[:,sections[0]],np.linspace(0,self.shape[0],len(rockCarbon[:,sections[0]])), lw=2, label='$\delta$13C')
        strat2 = stratPlot.plot(rockCarbon[:,sections[1]],np.linspace(0,self.shape[0],len(rockCarbon[:,sections[1]])), lw=2, label='$\delta$13C')
        
        plt.show()
        
    def plot(self,phase,parameter):
        phase=getattr(self,phase)
        plt.imshow(np.array(list(reversed(
        [[getattr(phase[j][i],parameter) for j in range(self.shape[0])] for i in range(self.shape[1])]
        ))).T)

    def flowMatrix(self,phase,parameters):  #positive('fluid','d13c',x)
        def newFluid(xDelta,xFlux,yDelta,yFlux):  
            fluid=1.0*((1.0*xDelta*xFlux+yDelta*yFlux)/(1.0*xFlux+yFlux))
            return fluid
                    
        matrix=np.zeros([len(parameters),3,self.shape[0],self.shape[1]])  #parameters, sum+2d, meshX, meshY      
        phase=getattr(self,phase)        
        for row in range(1,self.shape[0]-1):
            for column in range(1,self.shape[1]-1):
                for k in range(len(parameters)):  #k through trackable properties, d13c, d18o, etc
                    if self.positiveX[row][column]:
                        matrix[k][1][row][column]=getattr(phase[row][column-1],parameters[k])
                    else:
                        matrix[k][1][row][column]=getattr(phase[row][column+1],parameters[k])
                    if self.positiveY[row][column]:
                        matrix[k][2][row][column]=getattr(phase[row-1][column],parameters[k])
                    else:
                        matrix[k][2][row][column]=getattr(phase[row+1][column],parameters[k])

        
         
        for k in range(len(parameters)):
            for row in range(self.shape[0]):
                for column in range(self.shape[1]):
                    if (self.flux[row][column]!=0):
                        matrix[k][0][row][column]=newFluid(matrix[k][1][row][column],
                        np.abs(self.u[row][column]),matrix[k][2][row][column],
                        np.abs(self.v[row][column]))
                    else:
                        matrix[k][0][row][column]=getattr(phase[row][column],parameters[k])

        return matrix
                        
                
    def printBox(self,phase,parameter):
        phase=getattr(self,phase)
        return np.array(list((
        [[getattr(phase[j][i],parameter) for j in range(self.shape[0])] for i in range(self.shape[1])]
        ))).T
    
    def inject(self,steps):
        def react(M,t):  #redefine as rock, fluid?
            rockDelta,fluidDelta,flux,rr,fr=M
            #parameter table for fluid composition
            #        
            rockMass=1.0*rr
            fluidMass=flux*fr
            reactingMass=rockMass+fluidMass
            
            fIn=self.r*flux #flux = scaled to carbon, this forces reactions at right stiochiometric ratio
            fOut=fIn
        
            rockDelta1=((fIn*fluidDelta)-(fOut*rockDelta))/rockMass
            fluidDelta1=-1*((fIn*fluidDelta)-(fOut*rockDelta))/fluidMass               
            flux1=0
        
            return [rockDelta1,fluidDelta1,flux1,0,0]
            
        # define boundary injection (convert this to pull from call)
        for _ in range(steps):
#            for x in range(self.shape[0]):
#                for y in range(self.shape[1]):
#                    self.fluid[x][y].d13c=self.rock[x][y].d13c
#                    self.fluid[x][y].d18o=self.rock[x][y].d18o
#            for x in range(self.shape[1]):
#                for y in range(7):
#                    self.fluid[y][x].d13c=-7.0
#                    self.fluid[y][x].d18o=-10.0
        
            for i in range(90,95):
                for k in range(25,30):
                    self.fluid[k][i].d13c=-7
                    self.fluid[k][i].d18o=-10
            

                
                    #box.age=0
            
            delta=['d13c','d18o']  #what do you want to track?
            massRatio=[[1,1],[4,444]] #stiochiometric ratio between elements (r,f)
            
            M=self.flowMatrix('fluid',delta)
            
            t=np.linspace(0, 1, 2) 
        
            for k in range(len(delta)):
                for row in range(1,self.shape[0]-1):
                    for column in range(1,self.shape[1]-1):
                        if self.flux[row][column]!=0:
                            z=integrate.odeint(react,
                                                 [getattr(self.rock[row][column],delta[k]),
                                                 M[k][0][row][column],
                                                 self.flux[row][column],
                                                 massRatio[k][0],massRatio[k][1]],
                                                 t)
                                                
                            r,f,F,_,_ = z.T
                            setattr(self.fluid[row][column],delta[k],f[-1])
                            setattr(self.rock[row][column],delta[k],r[-1])                    
                            
                            if k==0:
                                self.fluid[row][column].age=self.fluid[row][column].age+1.0
                                self.rock[row][column].fluxed=self.rock[row][column].fluxed+self.flux[row][column]
                            
            
    



class rock:
    def __init__(self, d13c, d18o):
        self.d13c = d13c
        self.d18o = d18o
        self.fluxed = 0.0

        
class fluid:
    def __init__(self, d13c, d18o, u, v):
        self.d13c = d13c
        self.d18o = d18o
        self.age = 0
        self.u = u
        self.v = v
        self.flux=(np.abs(u)+np.abs(v))        
        
meshX=500
meshY=80
u=load('uWide.npy')
v=load('vWide.npy')
u=u
v=v
mesh=meshRock(meshX,meshY,np.array(list(reversed(u))),np.array(list(reversed(v))),2.0,2.0,.001)  #.01 = 1% per timestep~Ma
mesh.inject(50)
plt.style.use('ggplot')
mesh.compPlot()


