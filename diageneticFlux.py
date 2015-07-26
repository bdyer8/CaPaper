# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 10:25:25 2015

@author: bdyer
"""
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from numpy import load,linspace,meshgrid
from matplotlib import gridspec
timeSpanIter=np.linspace(2,4,3)
with plt.style.context('ggplot'):

    fig = plt.figure(figsize=(10, 4))
    gs = gridspec.GridSpec(1,2)
    crossPlot = plt.subplot(gs[0,0]) 
    crossPlot2 = plt.subplot(gs[0,1])
    for i in timeSpanIter:
        rPercent=.15
        timeSpan=i
        
        depthRange=np.linspace(1,180,100)
        depthRange=90
        
        
        platformArea=np.linspace(1*10**6,50*10**6,100)
        platformAreaCarbonate=.503*platformArea #surface area of carbonate platforms km^2 
        
            
        earthArea=510072000.0 #surface area of earth km^2
        platformVolume=platformAreaCarbonate*10**6*depthRange  #volume in m^3
        lsDens=2100 #kg/m^3
        platformMass=lsDens*platformVolume #kg
        cDens=12.011/(40.078+12.011+3*15.999)
        platformCMass=platformMass*cDens
        platformCMol=(platformCMass*1000)/12.011
        reactedPlatCMol=rPercent*platformCMol
        rCmolPerKY=reactedPlatCMol/(timeSpan*10**3)/10**12
        
     
        crossPlot.plot(platformArea/(10.0**6),rCmolPerKY)  #2-7Ma interval of diagenesis for 90m for 1-50km^2
        
        
    for i in timeSpanIter:
            rPercent=.15
            timeSpan=i
            
            depthRange=np.linspace(1,180,100)
            depthRange=40
            
            
            platformArea=np.linspace(1*10**6,50*10**6,100)
            platformAreaCarbonate=.503*platformArea #surface area of carbonate platforms km^2 
            
                
            earthArea=510072000.0 #surface area of earth km^2
            platformVolume=platformAreaCarbonate*10**6*depthRange  #volume in m^3
            lsDens=2100 #kg/m^3
            platformMass=lsDens*platformVolume #kg
            cDens=12.011/(40.078+12.011+3*15.999)
            platformCMass=platformMass*cDens
            platformCMol=(platformCMass*1000)/12.011
            reactedPlatCMol=rPercent*platformCMol
            rCmolPerKY=reactedPlatCMol/(timeSpan*10**3)/10**12
            
         
            crossPlot.plot(platformArea/(10.0**6),rCmolPerKY,linestyle='--',lw=1,color=plt.rcParams['axes.color_cycle'][int(i-2)])  #2-7Ma interval of diagenesis for 45m for 1-50km^2
            
    for i in range(8000):
            rPercent=.15
            timeSpan=np.random.uniform(2,4,1)
            
            depthRange=np.linspace(1,180,100)
            depthRange=np.random.uniform(45,130,1)
            
            
            platformArea=np.random.uniform(1*10**6,45*10**6,1)
            platformAreaCarbonate=.503*platformArea #surface area of carbonate platforms km^2 
            
                
            earthArea=510072000.0 #surface area of earth km^2
            platformVolume=platformAreaCarbonate*10**6*depthRange  #volume in m^3
            lsDens=2100 #kg/m^3
            platformMass=lsDens*platformVolume #kg
            cDens=12.011/(40.078+12.011+3*15.999)
            platformCMass=platformMass*cDens
            platformCMol=(platformCMass*1000)/12.011
            reactedPlatCMol=rPercent*platformCMol
            rCmolPerKY=reactedPlatCMol/(timeSpan*10**3)/10**12
            
         
            crossPlot2.scatter(platformArea/(10.0**6),rCmolPerKY,linestyle='--',lw=1,color=plt.rcParams['axes.color_cycle'][1])  #2-7Ma interval of diagenesis for 90m for 1-50km^2
        

#rCmolPerKY=99.8810 for pliest
#rCmolPerKY=1391.4 for miss