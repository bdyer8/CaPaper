# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 10:37:20 2015

@author: bdyer
"""
import DiagenesisMesh
from pylab import rcParams
from matplotlib import gridspec
import matplotlib.pyplot as plt
import pymc as pm
import numpy as np
import scipy
import pandas as pd
import pickle
import scipy.interpolate
from pymc.Matplot import plot
import pydot
interp1d=scipy.interpolate.interp1d
meshY=3
meshX=11 #solving 10m boxes
v=np.ones([meshY,meshX])*0.0
u=np.ones([meshY,meshX])*(0.002553540074184419) #decameters/100yr
aveSpeed=np.sqrt(v.mean()**2+u.mean()**2)
injectionSites=np.zeros([meshY,meshX])
injectionSites=injectionSites>0
injectionSites[:,0]=True  



def rockMassFrac(rockCDelta,metCDelta,orgCDelta):
    return 1-(((metCDelta-rockCDelta)/(orgCDelta-rockCDelta)))

ArrowCanyon=pd.read_csv('samples_ArrowCanyon.csv')
BattleshipWash=pd.read_csv('samples_BattleshipWash.csv')

#arrow canyon
ACd13c=(ArrowCanyon.d13c[ArrowCanyon.SAMP_HEIGHT<103])
BWd13c=(BattleshipWash.d13c[(((BattleshipWash.SAMP_HEIGHT-240)<103) & ((BattleshipWash.SAMP_HEIGHT-240)>0))])
ACd13c=list(ACd13c)+list(BWd13c)
ACmeters=(ArrowCanyon.SAMP_HEIGHT[ArrowCanyon.SAMP_HEIGHT<103])
BWmeters=(BattleshipWash.SAMP_HEIGHT[(((BattleshipWash.SAMP_HEIGHT-240)<103) & ((BattleshipWash.SAMP_HEIGHT-240)>0))]-240.0)
ACmeters=list(ACmeters)+list(BWmeters)


ACd44ca=ArrowCanyon.d44ca[(ArrowCanyon.SAMP_HEIGHT<103)].dropna()
BWd44ca=(BattleshipWash.d44ca[(((BattleshipWash.SAMP_HEIGHT-240)<103) & ((BattleshipWash.SAMP_HEIGHT-240)>0))]).dropna()
ACmeters2=ArrowCanyon.SAMP_HEIGHT[ACd44ca.index]
BWmeters2=(BattleshipWash.SAMP_HEIGHT[BWd44ca.index]-240.0)
ACmetersCa=list(ACmeters2)+list(BWmeters2)
ACd44ca=list(ACd44ca)+list(BWd44ca)

Leadville=pd.read_csv('samples_B416.csv')

LVd13c=Leadville.d13c[Leadville.d13c<0]
LVd13c=LVd13c.dropna()
LVd44ca=Leadville.d44ca[(Leadville.SAMP_HEIGHT>20) & (Leadville.SAMP_HEIGHT<93.0)]
LVd44ca=LVd44ca.dropna()
LVmetersCa=Leadville.SAMP_HEIGHT[LVd44ca.index]+9.0
LVmeters=Leadville.SAMP_HEIGHT[LVd13c.index]+9.0

#errCa = pm.Uniform("errCa", 0.0, 1500) #uncertainty on d13c values, flat prior
#errC = pm.Uniform("errC", 0.0, 1500) #uncertainty on d44ca values, flat prior
carbonMass = pm.Uniform("carbonMass",.1,25,value=4.0)
caMass = pm.Uniform("caMass",.1,1.3666666666666667*carbonMass)
caOffset = pm.Uniform('caOffset',-.05,0.05)


Age=np.random.normal(2.5e6,.5,10000)
#set pymc observations from data
heights1=pm.Normal("ACmeters", 0, 1, value=LVmeters.values, observed=True)
heights2=pm.Normal("ACmetersCa", 0, 1, value=LVmetersCa.values, observed=True)
#base=pm.Uniform('baseCa',-0.2,0.2)
#d13cbase=pm.Uniform('baseC',-1.0,1.0)
CaInject=pm.Uniform('CaInject',-1.6,-0.9)
CaAlpha=pm.Uniform('CaAlpha',.999,1.0,value=1.0,observed=True)
mesh=DiagenesisMesh.meshRock(meshX,3,u*.1,v,2,2,-0.9,3.5703864479316496e-06,injectionSites,[-8.0,-10.0,CaInject.value,0.0],[1.0,1.0,CaAlpha.value],[[329.0,carbonMass.value],[1315.0,889.0],[1096.0,caMass.value]]) #reaction rate from same sim    
@pm.deterministic
def predACd13c(carbonMass=carbonMass,age=Age,heights1=heights1,mesh=mesh):
    mesh.hardReset()
    mesh.massRatio[0][1]=carbonMass*100.0 
    mesh.dt=.7/(np.abs(mesh.u).max())
    if ((mesh.massRatio[0][1]*.1)<=(mesh.r*mesh.massRatio[0][0]*mesh.dt)):
            mesh.dt=.9*(mesh.massRatio[0][1]*.1)/(mesh.r*mesh.massRatio[0][0])
    mesh.inject(int((age[np.random.choice(len(age),1)]*.1)/mesh.dt)) 
    interpValC=interp1d(np.linspace(103,0,10),mesh.printBox('rock','d13c')[1,1:],kind='quadratic')
    return [interpValC(heights1)]
    
@pm.deterministic
def predACd44ca(caOffset=caOffset,caMass=caMass,age=Age,heights2=heights2,mesh=mesh,CaInject=CaInject,CaAlpha=CaAlpha):
    #mesh.alpha[-1]=CaAlpha    
    mesh.hardReset()
    mesh.massRatio[2][1]=caMass*100.0
    mesh.boundary[2]=CaInject  
    mesh.dt=.7/(np.abs(mesh.u).max())
    if ((mesh.massRatio[2][1]*.1)<=(mesh.r*mesh.massRatio[2][0]*mesh.dt)):
            mesh.dt=.9*(mesh.massRatio[2][1]*.1)/(mesh.r*mesh.massRatio[2][0])
    mesh.inject(int((age[np.random.choice(len(age),1)]*.1)/mesh.dt)) 
    interpValCa=interp1d(np.linspace(103,0,10),mesh.printBox('rock','d44ca')[1,1:],kind='quadratic') 
    return (interpValCa(heights2)+caOffset)

obs1 = pm.Normal("ACd13c", predACd13c, 0.1, value=np.array(LVd13c.values), observed=True)
obs2 = pm.Normal("ACd44ca", predACd44ca, 0.1, value=np.array(LVd44ca.values), observed=True)
#@pm.deterministic
#def predACd13c(d13cbase=d13cbase, alpha=alphaDist, caMass=caMassDist, heights=heights1,carbonSolutions=carbonSolutions):
#    return predFunct(alpha,caMass,heights,carbonSolutions)+d13cbase
#    
#obs2 = pm.Normal("ACd13c", predACd13c, errC, value=ACd13c, observed=True)
#
#model = pm.Model([errCa,base,errC,alphaDist,caMassDist,heights1,heights2,predACd13c,predACd44Ca,obs1,obs2])
model = pm.Model([predACd44ca,predACd13c,obs1,obs2,carbonMass,caMass,CaInject,caOffset])

map_start=pm.MAP(model)
map_start.fit()
mcmc=pm.MCMC(model,db='pickle', dbname=('LVFullModelCa_workingC.pickle'))
#mcmc=pm.MCMC(model)
mcmc.sample(100)

#%%
N=100
carbonFits=np.zeros((N,heights1.value.size))
for i in range(N):
    carbonFits[i,:]=mcmc.trace('predACd13c')[i]
    
caFits=np.zeros((N,heights2.value.size))
for i in range(N):
    caFits[i,:]=mcmc.trace('predACd44ca')[i]
with plt.style.context('fivethirtyeight'):
    
    rcParams.update({'figure.autolayout': True})
    rcParams.update({'font.size': 8.0})
    rcParams.update({'figure.subplot.bottom': 0})
    rcParams.update({'figure.subplot.hspace': 0.0})
    rcParams.update({'figure.subplot.left': 0.0})
    rcParams.update({'figure.subplot.right': 1.0})
    rcParams.update({'figure.subplot.top': 1.0})
    rcParams.update({'figure.subplot.wspace': 0.0})
    rcParams.update({'axes.color_cycle': [u'#30a2da',
                                u'#fc4f30',
                                u'#e5ae38',
                                u'#6d904f',
                                u'#8b8b8b',
                                '#d33682',
                                '#2aa198']})

    
    fig=plt.figure(figsize=(8, 5.5))   
    cFit=np.percentile(carbonFits,50,axis=0)
    caFit=np.percentile(caFits,50,axis=0)
    cFit2=np.percentile(carbonFits,2.5,axis=0)
    caFit2=np.percentile(caFits,2.5,axis=0)
    cFit9=np.percentile(carbonFits,97.5,axis=0)
    caFit9=np.percentile(caFits,97.5,axis=0)
    A=plt.subplot(1,2,1)
    B=plt.subplot(1,2,2)
    A.scatter(caFit,heights2.value)
    A.scatter(caFit2,heights2.value,alpha=.5)
    A.scatter(caFit9,heights2.value,alpha=.5)
    A.scatter(obs2.value,heights2.value,color='r')
    B.scatter(cFit,heights1.value)
    B.scatter(cFit2,heights1.value,alpha=.5)
    B.scatter(cFit9,heights1.value,alpha=.5)
    B.scatter(obs1.value,heights1.value,color='r')

#%%
with plt.style.context('fivethirtyeight'):
    fig=plt.figure(figsize=(8, 8.5)) 
    def a_r(carbon):
        return(mesh.dt*carbon*.1*mesh.u.max())/(240*mesh.dt*mesh.r)
    
    A=plt.subplot(5,1,1)
    B=plt.subplot(5,1,2)
    C=plt.subplot(5,1,3)
    D=plt.subplot(5,1,4)
    E=plt.subplot(5,1,5)

    
    A.plot(range(N),(mcmc.trace('caMass')[:]))
    B.plot(range(N),(mcmc.trace('CaInject')[:]))
    #C.plot(range(N),(mcmc.trace('Age')[:]/1e6))
    D.plot(range(N),a_r(mcmc.trace('carbonMass')[:]*100))
    #E.plot(range(N),(mcmc.trace('errCa')[:]))
    
    
    

#%%

mcmc.db.close()