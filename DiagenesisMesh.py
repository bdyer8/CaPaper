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
import math
import time
        
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
        
        self.Js=np.ones((meshY,meshX))
        self.Is=np.ones((meshY,meshX))
        self.JsN=np.ones((meshY,meshX))
        self.IsN=np.ones((meshY,meshX))
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
        self.dt=.09  # yr
        courant=np.abs(self.u).max()*self.dt+np.abs(self.v).max()*self.dt
        if courant>1.0:
            print(courant)
            print('Warning: advecting faster than timestep can resolve, decrease timestep (dt)')

    
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
        sections=[2,2]
        everyNX=2
        with plt.style.context('ggplot'):
        
            fp = open('cmap.pkl', 'rb')
            cmapDiag = pickle.load(fp)
            fp.close()
            
            speed = np.sqrt(self.u*self.u + self.v*self.v)
            lw = 10*speed/speed.max()
            Y, X = np.mgrid[0:self.shape[0], 0:self.shape[1]]
            d13cRockPlot.set_xlim([1,self.shape[1]-1])
            d13cRockPlot.set_ylim([self.shape[0]-1,1])
            im = d13cRockPlot.imshow(self.printBox('rock','d13c'), cmap='cubehelix', vmin=carbonAxes[2], vmax=carbonAxes[3])   
            fig.colorbar(im, ax=d13cRockPlot, label='$\delta$13C rock', orientation='horizontal',shrink=.5,pad=.0)
            d13cRockPlot.plot(sections[0]*np.ones(100),np.linspace(0,self.shape[1],100), lw=1.5, color=plt.rcParams['axes.color_cycle'][4])
            d13cRockPlot.plot(sections[1]*np.ones(100),np.linspace(0,self.shape[1],100), lw=1.5, color=plt.rcParams['axes.color_cycle'][1])
            qui = d13cRockPlot.streamplot(X, Y, self.u, self.v, density=.5, color='k', linewidth=lw)
            d13cRockPlot.grid(None)
            d13cRockPlot.axis('off')
            
                        
            
            d13cFluidPlot.set_xlim([1,self.shape[1]-1])
            d13cFluidPlot.set_ylim([self.shape[0]-1,1])
            im2 = d13cFluidPlot.imshow(self.printBox('rock','d44ca'), cmap='seismic_r', vmin=calciumAxes[2], vmax=calciumAxes[3])
            fig.colorbar(im2, ax=d13cFluidPlot, label='$\delta$44Ca rock', orientation='horizontal',shrink=.5,pad=.0)
            d13cFluidPlot.plot(sections[0]*np.ones(100),np.linspace(0,self.shape[1],100), lw=1.5, color=plt.rcParams['axes.color_cycle'][4])
            d13cFluidPlot.plot(sections[1]*np.ones(100),np.linspace(0,self.shape[1],100), lw=1.5, color=plt.rcParams['axes.color_cycle'][1])
            qui = d13cFluidPlot.streamplot(X, Y, self.u, self.v, density=.5, color='k', linewidth=lw)
            d13cFluidPlot.grid(None)
            d13cFluidPlot.axis('off')
            
            rockFluxed.set_xlim([1,self.shape[1]-1])
            rockFluxed.set_ylim([self.shape[0]-1,1])
            im3 = rockFluxed.imshow(self.printBox('fluid','age'), cmap='gist_stern_r', vmin=0, vmax=self.printBox('fluid','age').max()*1.1)
            fig.colorbar(im3, ax=rockFluxed, label='fluid age', orientation='horizontal',shrink=.5,pad=.0)
            rockFluxed.plot(sections[0]*np.ones(100),np.linspace(0,self.shape[1],100), lw=1.5, color=plt.rcParams['axes.color_cycle'][4])
            rockFluxed.plot(sections[1]*np.ones(100),np.linspace(0,self.shape[1],100), lw=1.5, color=plt.rcParams['axes.color_cycle'][1])
            qui = rockFluxed.streamplot(X, Y, self.u, self.v, density=.5, color='k', linewidth=lw)
            rockFluxed.grid(None)
            rockFluxed.axis('off')
            
            covarPlot.set_xlim([1,self.shape[1]-1])
            covarPlot.set_ylim([self.shape[0]-1,1])
            misfit=(self.printBox('rock','d18o')-self.printBox('rock','d13c')*((2.0--7.0)/(-1.0--6.0))+4.6)
            #ratio=(2-np.abs(self.printBox('rock','d13c')-2))/(-1-(self.printBox('rock','d18o')+1))
            #ratio[np.isnan(ratio)]=2.0
            im4 = covarPlot.imshow(misfit, cmap='ocean_r', vmin=0,vmax=5)
            fig.colorbar(im4, ax=covarPlot, label='$\Delta\delta$13C \ $\Delta\delta$18O rock', orientation='horizontal',shrink=.5,pad=.0)
            covarPlot.plot(sections[0]*np.ones(100),np.linspace(0,self.shape[1],100), lw=1.5, color=plt.rcParams['axes.color_cycle'][4])
            covarPlot.plot(sections[1]*np.ones(100),np.linspace(0,self.shape[1],100), lw=1.5, color=plt.rcParams['axes.color_cycle'][1])
            qui = covarPlot.streamplot(X, Y, self.u, self.v, density=.5, color='k', linewidth=lw)
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
                        
            fluidA=self.printBox('fluid','age')
            crossPlotCO.set_xlim([oxygenAxes[0], oxygenAxes[1]])
            crossPlotCO.set_ylim([carbonAxes[0], carbonAxes[1]])   
            crossPlotCO.set_ylabel('$\delta$13C Rock', labelpad=-8)
            crossPlotCO.set_xlabel('$\delta$18O Rock', labelpad=0)
            #crossRockCO4 = crossPlotCO.scatter(rockOxygen[:,49],rockCarbon[:,49],color=plt.rcParams['axes.color_cycle'][6], s=15, alpha=.8)            
            #crossRockCO3 = crossPlotCO.scatter(rockOxygen[:,100],rockCarbon[:,100],color=plt.rcParams['axes.color_cycle'][4], s=15, alpha=.8)
            crossRockCO2 = crossPlotCO.scatter(rockOxygen,rockCarbon,c=fluidA, cmap='gist_stern_r', s=4,alpha=.8,edgecolors='none',vmin=0, vmax=fluidA.max()*1.1)
            
            crossPlotCCa.set_xlim([calciumAxes[0], calciumAxes[1]])
            crossPlotCCa.set_ylim([carbonAxes[0], carbonAxes[1]])
            crossPlotCCa.set_ylabel('$\delta$13C Rock', labelpad=-8)
            crossPlotCCa.set_xlabel('$\delta$44Ca Rock', labelpad=0)            
            crossRockCCa2 = crossPlotCCa.scatter(rockCalcium,rockCarbon,c=fluidA, cmap='gist_stern_r', s=4,alpha=.8,edgecolors='none',vmin=0, vmax=fluidA.max()*1.1)
            
            crossPlotCaO.set_xlim([calciumAxes[0], calciumAxes[1]])
            crossPlotCaO.set_ylim([oxygenAxes[0], oxygenAxes[1]])
            crossPlotCaO.set_ylabel('$\delta$18O Rock', labelpad=-5)
            crossPlotCaO.set_xlabel('$\delta$44Ca Rock', labelpad=0)            
            crossRockCaO2 = crossPlotCaO.scatter(rockCalcium,rockOxygen,c=fluidA, cmap='gist_stern_r', s=4,alpha=.8,edgecolors='none',vmin=0, vmax=fluidA.max()*1.1)
            
            crossPlotCaFlux.set_xlim([calciumAxes[0], calciumAxes[1]])
            crossPlotCaFlux.set_ylim([0, self.printBox('fluid','age').max()*1.1])
            rockFluxed=self.printBox('rock','fluxed')
            crossPlotCaFlux.set_ylabel('Fluid Age', labelpad=-5)
            crossPlotCaFlux.set_xlabel('$\delta$44Ca Rock', labelpad=0)            
            crossRockCaFlux = crossPlotCaFlux.scatter(rockCalcium,self.printBox('fluid','age'),color=plt.rcParams['axes.color_cycle'][3], s=1, alpha=.7)
            crossFluidCaFlux = crossPlotCaFlux.scatter(self.printBox('fluid','d44ca')-.5,self.printBox('fluid','age'),color=plt.rcParams['axes.color_cycle'][1], s=1, alpha=.7)
          
    def plot(self,phase,parameter):
        phase=getattr(self,phase)
        plt.imshow(np.array(list(reversed(
        [[getattr(phase[j][i],parameter) for j in range(self.shape[0])] for i in range(self.shape[1])]
        ))).T)                       
                
    def printBox(self,phase,parameter):
        phase=getattr(self,phase)
        return np.array(list((
        [[getattr(phase[j][i],parameter) for j in range(self.shape[0])] for i in range(self.shape[1])]
        ))).T    
        
    def inject(self,steps):            
        def reactBox(rockDelta,fluidDelta,rr,fr,alpha):
            rockMass=1.0*np.array(rr)
            porosity=.05
            fluidMass=1.0*np.array(fr)*porosity
            fIn=np.float64(self.r)*self.dt*rockMass     #fixed reaction rate per box       
            fOut=fIn
            reactingDelta=fluidDelta+(np.array(alpha)-1.0)*10**3
            rockDelta1=((fIn*reactingDelta)-(fOut*rockDelta))/rockMass
            fluidDelta1=-1*((fIn*reactingDelta)-(fOut*rockDelta))/fluidMass
            return [rockDelta1,fluidDelta1]
            

        for _ in range(steps):
            def AdvectionStep(delta,boundary): 
                def timeSavingFunction(cXn,cXi,cYi,v,u):
                    cXn=cXn - v * self.dt/1.0 * (cXn - cYi) - u * self.dt/1.0 * (cXn - cXi)
                    return cXn
                nt = self.nt #dt time steps per reaction
                nx=self.shape[1]
                ny=self.shape[0]
                u = self.u
                v = self.v
                cX=np.zeros([len(delta),self.shape[0],self.shape[1]]) 
                abso=np.abs

                for k in range(len(delta)):
                    cX[k][:,:]=self.printBox('fluid',delta[k])    
                    for n in range(nt+1): 
                        cX[k][0:3,:]=boundary[k]  #BOUNDARY CONDITIONS
                        cXn = cX[k].copy()
                        cX[k][1:-1,1:-1]=np.array(map(timeSavingFunction,cXn[1:-1,1:-1],cXn[list(self.IsN[1:-1,1:-1]),list(self.Js[1:-1,1:-1])],cXn[list(self.Is[1:-1,1:-1]),list(self.JsN[1:-1,1:-1])],abso(v[1:-1,1:-1]),abso(u[1:-1,1:-1])))    
                return cX
            
            self.injectionAge=self.injectionAge+1
           
            delta=['d13c','d18o','d44ca','age'] #properties to track
            boundary=[-7.0,-6.0,-1.0,0.0] #BOUNDARY CONDITIONS
            massRatio=[[240.0,2.0],[960.0,889.0],[800.0,70.0]] #stiochiometric ratio between elements (r,f) (Ca, 1-37)
            alpha=[1.0,1.0,0.9995]

            #timeInt=time.time()
            cX=AdvectionStep(delta,boundary) #fast, scales linearly at number of things you track in delta
            #print(time.time()-timeInt)
            for k in range(len(delta)-1):
                for row in range(1,self.shape[0]-1):
                    for column in range(1,self.shape[1]-1):
                        setattr(self.fluid[row][column],delta[k],cX[k][row,column])    
            #print(time.time()-timeInt)
            for k in range(len(delta)-1):
                R=  self.printBox('rock',delta[k])
                F= self.printBox('fluid',delta[k]) 
                onesShape=np.ones((self.shape[0],self.shape[1]))
                dRock,dFluid=reactBox(R,F,
                                      onesShape*massRatio[k][0], #rock mass 1m^3
                                      onesShape*massRatio[k][1], #fluid mass 1m^3
                                      onesShape*alpha[k])
                newRock=R+dRock
                newFluid=F+dFluid                        
                for row in range(1,self.shape[0]-1):
                    for column in range(1,self.shape[1]-1):
                        setattr(self.rock[row][column],delta[k],newRock[row,column])
                        setattr(self.fluid[row][column],delta[k],newFluid[row,column])    
                        if k==0:
                            self.fluid[row][column].age=self.dt+cX[-1][row][column]
            #print(time.time()-timeInt)                
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
            #print(time.time()-timeInt)



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

