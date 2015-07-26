# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 10:01:05 2015

@author: bdyer
"""
import pickle
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from numpy import load,linspace,meshgrid
from matplotlib import gridspec
import time

def carbon1BoxModel(M,t):  #redefine as rock, fluid?
    
    Mass=M[0]
    deltaCarbonate=M[1]
    pco2=M[2]
    meteoricFlux=0
    cWeatherExtra=0
    deltaFOW=0
    if (t>3000) & (t<6000):
        #if t%400.0>200:
            meteoricFlux=M[3]
            cWeatherExtra=M[4]
            #deltaFOW=M[3]/5.0
            
    PO4=.25
    
    fOW=10000+deltaFOW

    deltaB=(159.5*PO4+38.39)/(0.034*pco2)-33


    #weathering flux in
    fluxOrganicWeathering=fOW-meteoricFlux
    deltaOrganicWeathering=-22
    fluxSilicateWeathering=6000
    deltaSilicateWeathering=-5
    fluxCarbonateWeathering=34000
    deltaCarbonateWeathering=0
    fluxCarbonatePlatformWeathering=cWeatherExtra
    deltaCarbonatePlatformWeathering=M[5] #arrow canyon number
    #burial flux
    fluxCarbonateBurial=38000+deltaFOW*(1)+((1)*((cWeatherExtra-meteoricFlux)))
    deltaCarbonateBurial=deltaCarbonate
    fluxOrganicBurial=12000+deltaFOW*(.0)
    deltaOrganicBurial=deltaCarbonate+(deltaB)



    Massdt=(fluxOrganicWeathering + fluxSilicateWeathering
            + fluxCarbonateWeathering + fluxCarbonatePlatformWeathering
            - fluxCarbonateBurial - fluxOrganicBurial)
            
    massDeltaDT =(fluxOrganicWeathering * (deltaOrganicWeathering)
                  + fluxSilicateWeathering * (deltaSilicateWeathering)
                  + fluxCarbonateWeathering * (deltaCarbonateWeathering)
                  + fluxCarbonatePlatformWeathering * (deltaCarbonatePlatformWeathering)
                  - fluxOrganicBurial * (deltaOrganicBurial)
                  - fluxCarbonateBurial * (deltaCarbonateBurial))
               
    deltaCarbonatedt=(massDeltaDT-deltaCarbonate*Massdt)/Mass;
    pco2dt=Massdt*pco2/Mass;
    
    
        
    return Massdt,deltaCarbonatedt,pco2dt,0,0,0
    
with plt.style.context('ggplot'):
    t=np.linspace(0, 6000.0, 800) 
#    z=integrate.odeint(carbon1BoxModel,
#                       [3.8*10**6,0.0,570.0,2707.0,2707.0,2],
#                       t, hmax=1)
#    fig = plt.figure(figsize=(15, 5))
#    gs = gridspec.GridSpec(1, 3) 
#    deltaPlot = plt.subplot(gs[0,:2])
#    deltaPlot.set_xlabel('Time (ky)')
#    deltaPlot.set_ylabel('$\delta$13C of the Ocean')
#    #MassPlot = plt.subplot(gs[0,0]) 
#    crossPlot2 = plt.subplot(gs[0,2]) 
#    crossPlot2.set_xlabel('Global Carbonate Platform Area (10^6 km^2)')
#    crossPlot2.set_ylabel('$\Delta\delta$13C')
#    #pco2Plot = plt.subplot(gs[2,0])
#    
#    #MassPlot.plot(t,z[:,0]/10**6, lw=2)
#    #pco2Plot.plot(t,z[:,2], lw=2)
#    reverseW=[0,1,2]
#    z=integrate.odeint(carbon1BoxModel,
#                       [3.8*10**6,0.0,570.0,2707.0,2707.0*reverseW[0],2],
#                       t, hmax=1)
#    #MassPlot.plot(t,z[:,0]/10**6, lw=2)
#    deltaPlot.plot(t,z[:,1], lw=2)
#    #pco2Plot.plot(t,z[:,2], lw=2)
#    
#    z=integrate.odeint(carbon1BoxModel,
#                       [3.8*10**6,0.0,570.0,2707.0,2707.0*reverseW[1],2],
#                       t, hmax=1)
#    #MassPlot.plot(t,z[:,0]/10**6, lw=2)    
#    deltaPlot.plot(t,z[:,1], lw=2)
#    #pco2Plot.plot(t,z[:,2], lw=2)
#    
#    z=integrate.odeint(carbon1BoxModel,
#                       [3.8*10**6,0.0,570.0,2707.0,2707.0*reverseW[2],2],
#                       t, hmax=1)
#    #MassPlot.plot(t,z[:,0]/10**6, lw=2)    
#    deltaPlot.plot(t,z[:,1], lw=2)
    #pco2Plot.plot(t,z[:,2], lw=2)
    uniform=np.random.uniform
    linspace=np.linspace
    number=100000
    Delta=np.zeros(number)
    platformArea=np.zeros(number)
    rPercent=np.zeros(number)
    timeSpan=np.zeros(number)
    reverseW=np.zeros(number)
    depthRange=np.zeros(number)
    timestamper=time.time
    normalRand=np.random.normal
    cDens=12.011/(40.078+12.011+3*15.999)
    lsDens=2100.0
    earthArea=510072000.0 #surface area of earth km^2
for i in range(number):
            #tstamp=timestamper()
            rPercent[i]=.13+abs(normalRand(0,.025,1))
            timeSpan[i]=normalRand(2,.5,1)
            reverseW[i]=uniform(0,2.0,1)
            depthRange[i]=normalRand(55,15,1)
            
            platformArea[i]=uniform(1*10**6,45*10**6,1)
            platformAreaCarbonate=1.0*platformArea[i] #surface area of carbonate platforms km^2 
            platformVolume=platformAreaCarbonate*10**6*depthRange[i]  #volume in m^3
            platformMass=lsDens*platformVolume #kg
            platformCMass=platformMass*cDens
            platformCMol=(platformCMass*1000)/12.011
            reactedPlatCMol=rPercent[i]*platformCMol
            rCmolPerKY=reactedPlatCMol/(timeSpan[i]*10**3)/10**12
            z=integrate.odeint(carbon1BoxModel,
                               [3.8*10**6,0.0,570.0,rCmolPerKY,rCmolPerKY*reverseW[i],2],
                               t, hmax=10)
            Delta[i]=z[500,1]-z[200,1]
            
            #crossPlot2.scatter(platformArea[i]/(10.0**6),Delta[i],color=plt.rcParams['axes.color_cycle'][1])
            #print(time.time()-tstamp)
#plt.show()  
pickle.dump( platformArea, open( "mcPlatArea2.npy", "wb" ) )
pickle.dump( Delta, open( "mcDelta2.npy", "wb" ) )
pickle.dump( rPercent, open( "mcRpercent2.npy", "wb" ) )
pickle.dump( timeSpan, open( "mcTimeSpan2.npy", "wb" ) )
pickle.dump( reverseW, open( "mcCarbWeather2.npy", "wb" ) )



