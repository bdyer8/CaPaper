# -*- coding: utf-8 -*-
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
import pickle
from matplotlib import rcParams
import pandas as pd
import copy
        
class meshRock:
    def __init__(self,meshX,meshY,u,v,d13c,d18o,d44ca,reactionRate):
        self.u=u
        self.v=v 
        self.shape=[meshY,meshX]
        self.size=meshX*meshY
        x=linspace(0,meshX,meshX)
        y=linspace(0,meshY,meshY)
        self.X,self.Y=meshgrid(x,y)
        self.rock=[[rock(d13c,d18o,d44ca) for j in range(meshX)] for i in range(meshY)]
        self.fluid=[[fluid(d13c,d18o,d44ca-(0.9995-1)*10**3,u[i,j],v[i,j]) for j in range(meshX)] for i in range(meshY)]
        self.positiveX=np.array(self.u)>0
        self.positiveY=np.array(self.v)<0
        self.negativeX=np.array(self.u)<0
        self.negativeY=np.array(self.v)>0
        self.zeroSum=(np.array(self.v)+np.array(self.u)==0)
        self.flux=np.abs(self.v)+np.abs(self.u)
        self.maxF=np.max(self.flux)
        self.r=reactionRate
        self.injectionAge=0.0
    
    def compPlot(self):
        fig = plt.figure(figsize=(15, 12))
        self.compPlotAni(fig)
        plt.show()
    
    def compPlotAni(self,fig):
        gs = gridspec.GridSpec(4, 5) 
        d13cRockPlot = plt.subplot(gs[0,0:3])
        d13cFluidPlot = plt.subplot(gs[1,0:3])
        rockFluxed = plt.subplot(gs[2,0:3])
        covarPlot = plt.subplot(gs[3,0:3])
        stratPlot = plt.subplot(gs[0:2,3])
        crossPlotCO = plt.subplot(gs[2:3,3])
        crossPlotCCa = plt.subplot(gs[3:,3])
        crossPlotCaO = plt.subplot(gs[2:3,4])
        crossPlotCaFlux = plt.subplot(gs[3:,4])
        stratPlotCa = plt.subplot(gs[0:2,4])
        carbonAxes=[-8.5,3.0,-10,2]
        oxygenAxes=[-6.5,-.5,-6,0]
        calciumAxes=[-1.6,-.8,-2,0]
        sections=[10,40]
        everyNX=2
        with plt.style.context('ggplot'):
        
            fp = open('cmap.pkl', 'rb')
            cmapDiag = pickle.load(fp)
            fp.close()
            
            speed = np.sqrt(self.u*self.u + self.v*self.v)
            lw = 5*speed/speed.max()
            Y, X = np.mgrid[0:self.shape[0], 0:self.shape[1]]
            d13cRockPlot.set_xlim([1,self.shape[1]-1])
            d13cRockPlot.set_ylim([self.shape[0]-1,1])
            im = d13cRockPlot.imshow(self.printBox('rock','d13c'), cmap='cubehelix', vmin=carbonAxes[2], vmax=carbonAxes[3])   
            fig.colorbar(im, ax=d13cRockPlot, label='$\delta$13C rock', orientation='horizontal',shrink=.5,pad=.0)
            d13cRockPlot.plot(sections[0]*np.ones(100),np.linspace(0,self.shape[1],100), lw=1.5, color=plt.rcParams['axes.color_cycle'][4])
            d13cRockPlot.plot(sections[1]*np.ones(100),np.linspace(0,self.shape[1],100), lw=1.5, color=plt.rcParams['axes.color_cycle'][1])
            qui = d13cRockPlot.streamplot(X, Y, self.u, self.v, density=.5, color='k', linewidth=lw*.1)
            d13cRockPlot.grid(None)
            d13cRockPlot.axis('off')
            
                        
            
            d13cFluidPlot.set_xlim([1,self.shape[1]-1])
            d13cFluidPlot.set_ylim([self.shape[0]-1,1])
            im2 = d13cFluidPlot.imshow(self.printBox('rock','d44ca'), cmap='seismic_r', vmin=calciumAxes[2], vmax=calciumAxes[3])
            fig.colorbar(im2, ax=d13cFluidPlot, label='$\delta$44Ca rock', orientation='horizontal',shrink=.5,pad=.0)
            d13cFluidPlot.plot(sections[0]*np.ones(100),np.linspace(0,self.shape[1],100), lw=1.5, color=plt.rcParams['axes.color_cycle'][4])
            d13cFluidPlot.plot(sections[1]*np.ones(100),np.linspace(0,self.shape[1],100), lw=1.5, color=plt.rcParams['axes.color_cycle'][1])
            qui = d13cFluidPlot.streamplot(X, Y, self.u, self.v, density=.5, color='k', linewidth=lw*.1)
            d13cFluidPlot.grid(None)
            d13cFluidPlot.axis('off')
            
            rockFluxed.set_xlim([1,self.shape[1]-1])
            rockFluxed.set_ylim([self.shape[0]-1,1])
            im3 = rockFluxed.imshow(self.printBox('fluid','age'), cmap='cubehelix')
            fig.colorbar(im3, ax=rockFluxed, label='fluid age', orientation='horizontal',shrink=.5,pad=.0)
            rockFluxed.plot(sections[0]*np.ones(100),np.linspace(0,self.shape[1],100), lw=1.5, color=plt.rcParams['axes.color_cycle'][4])
            rockFluxed.plot(sections[1]*np.ones(100),np.linspace(0,self.shape[1],100), lw=1.5, color=plt.rcParams['axes.color_cycle'][1])
            qui = rockFluxed.streamplot(X, Y, self.u, self.v, density=.5, color='k', linewidth=lw*.1)
            rockFluxed.grid(None)
            rockFluxed.axis('off')
            
            covarPlot.set_xlim([1,self.shape[1]-1])
            covarPlot.set_ylim([self.shape[0]-1,1])
            ratio=np.abs(self.printBox('rock','d13c'))/np.abs(self.printBox('rock','d18o'))+2
            im4 = covarPlot.imshow(self.printBox('rock','d18o'), cmap='cubehelix', vmin=-6, vmax=-1)
            fig.colorbar(im4, ax=covarPlot, label='$\delta$18O rock', orientation='horizontal',shrink=.5,pad=.0)
            covarPlot.plot(sections[0]*np.ones(100),np.linspace(0,self.shape[1],100), lw=1.5, color=plt.rcParams['axes.color_cycle'][4])
            covarPlot.plot(sections[1]*np.ones(100),np.linspace(0,self.shape[1],100), lw=1.5, color=plt.rcParams['axes.color_cycle'][1])
            qui = covarPlot.streamplot(X, Y, self.u, self.v, density=.5, color='k', linewidth=lw*.1)
            covarPlot.grid(None)
            covarPlot.axis('off')
            
            stratPlot.set_xlim([carbonAxes[0], carbonAxes[1]])
            stratPlot.set_ylim([self.shape[0]-1,1])   
            stratPlot.set_xlabel('$\delta$13C and $\delta$18O Rock', labelpad=0)
            stratPlot.set_ylabel('Depth (meters)', labelpad=0)
            rockCarbon=self.printBox('rock','d13c')
            rockOxygen=self.printBox('rock','d18o')
            strat = stratPlot.plot(rockCarbon[:,sections[0]],np.linspace(0,self.shape[0],len(rockCarbon[:,sections[0]])), color=plt.rcParams['axes.color_cycle'][4], lw=2, label='$\delta$13C')
            strat2 = stratPlot.plot(rockCarbon[:,sections[1]],np.linspace(0,self.shape[0],len(rockCarbon[:,sections[1]])), color=plt.rcParams['axes.color_cycle'][1],lw=2, label='$\delta$13C')
            strat3 = stratPlot.plot(rockOxygen[:,sections[0]],np.linspace(0,self.shape[0],len(rockOxygen[:,sections[0]])), color=plt.rcParams['axes.color_cycle'][4], linestyle='--',lw=2, label='$\delta$18O')
            strat4 = stratPlot.plot(rockOxygen[:,sections[1]],np.linspace(0,self.shape[0],len(rockOxygen[:,sections[1]])), color=plt.rcParams['axes.color_cycle'][1], linestyle='--',lw=2, label='$\delta$18O')
            
            stratPlotCa.set_xlim([calciumAxes[0], calciumAxes[1]])
            stratPlotCa.set_ylim([self.shape[0]-1,1])     
            stratPlotCa.set_xlabel('$\delta$44Ca Rock', labelpad=0)
            stratPlotCa.set_ylabel('Depth (meters)', labelpad=0)
            rockCalcium=self.printBox('rock','d44ca')
            strat = stratPlotCa.plot(rockCalcium[:,sections[0]],np.linspace(0,self.shape[0],len(rockCalcium[:,sections[0]])), color=plt.rcParams['axes.color_cycle'][4], lw=2, label='$\delta$44Ca')
            strat2 = stratPlotCa.plot(rockCalcium[:,sections[1]],np.linspace(0,self.shape[0],len(rockCalcium[:,sections[1]])), color=plt.rcParams['axes.color_cycle'][1], lw=2, label='$\delta$44Ca')
                        
            
            crossPlotCO.set_xlim([oxygenAxes[0], oxygenAxes[1]])
            crossPlotCO.set_ylim([carbonAxes[0], carbonAxes[1]])   
            crossPlotCO.set_ylabel('$\delta$13C Rock', labelpad=-8)
            crossPlotCO.set_xlabel('$\delta$18O Rock', labelpad=0)
            #crossRockCO4 = crossPlotCO.scatter(rockOxygen[:,49],rockCarbon[:,49],color=plt.rcParams['axes.color_cycle'][6], s=15, alpha=.8)            
            #crossRockCO3 = crossPlotCO.scatter(rockOxygen[:,100],rockCarbon[:,100],color=plt.rcParams['axes.color_cycle'][4], s=15, alpha=.8)
            crossRockCO2 = crossPlotCO.scatter(rockOxygen[:,:],rockCarbon[:,:],color=plt.rcParams['axes.color_cycle'][1], s=1, alpha=.3)
            
            crossPlotCCa.set_xlim([calciumAxes[0], calciumAxes[1]])
            crossPlotCCa.set_ylim([carbonAxes[0], carbonAxes[1]])
            crossPlotCCa.set_ylabel('$\delta$13C Rock', labelpad=-8)
            crossPlotCCa.set_xlabel('$\delta$44Ca Rock', labelpad=0)            
            crossRockCCa2 = crossPlotCCa.scatter(rockCalcium,rockCarbon,color=plt.rcParams['axes.color_cycle'][5], s=1, alpha=.5)
            
            crossPlotCaO.set_xlim([calciumAxes[0], calciumAxes[1]])
            crossPlotCaO.set_ylim([oxygenAxes[0], oxygenAxes[1]])
            crossPlotCaO.set_ylabel('$\delta$18O Rock', labelpad=-5)
            crossPlotCaO.set_xlabel('$\delta$44Ca Rock', labelpad=0)            
            crossRockCaO2 = crossPlotCaO.scatter(rockCalcium,rockOxygen,color=plt.rcParams['axes.color_cycle'][6], s=1, alpha=.5)
            
            crossPlotCaFlux.set_xlim([calciumAxes[0], calciumAxes[1]])
            #crossPlotCaFlux.set_ylim([-1, 5])
            rockFluxed=self.printBox('rock','fluxed')
            crossPlotCaFlux.set_ylabel('Fluid Age', labelpad=-5)
            crossPlotCaFlux.set_xlabel('$\delta$44Ca Rock', labelpad=0)            
            crossRockCaFlux = crossPlotCaFlux.scatter(rockCalcium,self.printBox('fluid','age'),color=plt.rcParams['axes.color_cycle'][3], s=1, alpha=.7)
          
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
            rockDelta,fluidDelta,flux,rr,fr,alpha=M     
            rockMass=1.0*rr
            fluidMass=flux*fr
            reactingMass=rockMass+fluidMass
            fIn=self.r*rockMass     #fixed reaction rate per box       
            fOut=fIn
            reactingDelta=fluidDelta+(alpha-1.0)*10**3
            if fIn>rockMass:
                print('Warning: Flux in is '+ str(fIn) +' and rock mass is ' + str(rockMass) +' for single reaction step, consider scaling flow field down')
            rockDelta1=((fIn*reactingDelta)-(fOut*rockDelta))/rockMass
            fluidDelta1=-1*((fIn*reactingDelta)-(fOut*rockDelta))/fluidMass               
            flux1=0
        
            return [rockDelta1,fluidDelta1,flux1,0,0,0]
            

        for _ in range(steps):
            def AdvectionStep(delta,boundary):         
                nt = 5 #time steps
                nx=self.shape[1]
                ny=self.shape[0]
                dx = 1
                dy = 1
                dt = .01
                u = self.u
                v = self.v
                cX=np.zeros([len(delta),self.shape[0],self.shape[1]]) 
                abso=np.abs
                ySign,xSign=np.ones([2,ny,nx])
                for i in range(ny):
                    for k in range(nx):
                        ySign[i,k]=-1*math.copysign(1,v[i,k])
                        xSign[i,k]=-1*math.copysign(1,u[i,k])
                
                for k in range(len(delta)):
                    cX[k][:,:]=self.printBox('fluid',delta[k])    
                    for n in range(nt+1): ##loop across number of time steps
                        cX[k][6:9,5:-5:6]=boundary[k]
                        cXn = cX[k].copy()
                        for i in range(1,ny-1):
                            for j in range(1,nx-1):
                                cX[k][i,j]=   (   cXn[i,j]
                                            - (abso(u[i,j]) * dt/dx * (cXn[i,j] - cXn[i+ySign[i,j],j]))
                                            - (abso(v[i,-j]) * dt/dy * (cXn[i,j] - cXn[i,j+xSign[i,j]]))
                                           )
                    
                return cX
            
            self.injectionAge=self.injectionAge+1
            delta=['d13c','d18o','d44ca','age'] #properties to track
            boundary=[-7.0,-6.0,-1.0,0.0]
            massRatio=[[1.0,1.0],[4.0,444.0],[3.33,2.0]] #stiochiometric ratio between elements (r,f) (Ca, 1-37)
            alpha=[1.0,1.0,0.9995]
            #get a matrix of new fluid values here:
            cX=AdvectionStep(delta,boundary)
            ## below here we react the new fluid  
            t=np.linspace(0, 1, 2) 
            for k in range(len(delta)-1):
                for row in range(1,self.shape[0]-1):
                    for column in range(1,self.shape[1]-1):
                        if self.flux[row][column]!=0:
                            z=integrate.odeint(react,
                                                 [getattr(self.rock[row][column],delta[k]),
                                                 cX[k][row,column],
                                                 self.flux[row][column],
                                                 massRatio[k][0],massRatio[k][1],
                                                 alpha[k]],
                                                 t)
                                                
                            r,f,F,_,_,_ = z.T
                            setattr(self.fluid[row][column],delta[k],f[-1])
                            setattr(self.rock[row][column],delta[k],r[-1])                    
                            
                            if k==0:
                                self.fluid[row][column].age=1+cX[-1][row][column]
#                                self.rock[row][column].fluxed=self.rock[row][column].fluxed+self.flux[row][column]
#            
            
#            for row in range(self.shape[0]):
#                for column in range(self.shape[1]):
#                    self.fluid[row][column].age=self.fluid[row][column].age+1
#           # clean up edges 
            for row in range(0,self.shape[0]):
                self.fluid[row][0]=copy.copy(self.fluid[row][1])
                self.fluid[row][-1]=copy.copy(self.fluid[row][-2])
                self.rock[row][0]=copy.copy(self.rock[row][1])
                self.rock[row][-1]=copy.copy(self.rock[row][-2])
            for column in range(0,self.shape[1]):
                self.fluid[0][column]=copy.copy(self.fluid[1][column])
                self.fluid[-1][column]=copy.copy(self.fluid[-2][column])
                self.rock[0][column]=copy.copy(self.rock[1][column])
                self.rock[-1][column]=copy.copy(self.rock[-2][column])
            
    



class rock:
    def __init__(self, d13c, d18o, d44ca):
        self.d13c = d13c
        self.d18o = d18o
        self.d44ca = d44ca
        self.fluxed = 0.0

        
class fluid:
    def __init__(self, d13c, d18o, d44ca, u, v):
        self.d13c = d13c
        self.d18o = d18o
        self.d44ca = d44ca
        self.age = 0
        self.u = u
        self.v = v
        self.flux=(np.abs(u)+np.abs(v))        

