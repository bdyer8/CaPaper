# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 12:54:59 2015

@author: bdyer
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 13:41:42 2015

@author: bdyer
"""
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

#%%
interp1d=scipy.interpolate.interp1d
def bilinear_interpolation(x, y, points):
    
    points = sorted(points)               # order points by x, then by y
    (x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = points

    if x1 != _x1 or x2 != _x2 or y1 != _y1 or y2 != _y2:
        raise ValueError('points do not form a rectangle')
    if not x1 <= x <= x2 or not y1 <= y <= y2:
        raise ValueError(str(x)+','+str(y)+'(x, y) not within the rectangle:'+str(x1)+','+str(y1)+'-'+str(x2)+','+str(y2))

    return  ((x2 - x) * (y2 - y) / ((x2 - x1) * (y2 - y1) + 0.0),
            (x - x1) * (y2 - y) / ((x2 - x1) * (y2 - y1) + 0.0),
            (x2 - x) * (y - y1) / ((x2 - x1) * (y2 - y1) + 0.0),
            (x - x1) * (y - y1) / ((x2 - x1) * (y2 - y1) + 0.0))
            
def synthData(a, b, m):
    idxA=np.abs((a2r-a)).argmin()
    if a2r[idxA]>a:
        idxA_2=idxA.copy()
        idxA=idxA-1
    else:
        idxA_2=idxA+1
    idxT=np.abs((t-b)).argmin()
    if t[idxT]>b:
        idxT_2=idxT.copy()
        idxT=idxT-1
    else:
        idxT_2=idxT+1
    if idxT_2>=100:
        idxT_2=99
        idxT=98
        b=t[-1]

    highResModel1=interp1d(np.linspace(103,0,104),modelSolutions[idxA,idxT,:])
    highResModel2=interp1d(np.linspace(103,0,104),modelSolutions[idxA_2,idxT,:])
    highResModel3=interp1d(np.linspace(103,0,104),modelSolutions[idxA,idxT_2,:])
    highResModel4=interp1d(np.linspace(103,0,104),modelSolutions[idxA_2,idxT_2,:])    
    data=np.array([highResModel1(m),highResModel2(m),highResModel3(m),highResModel4(m)]) 
    points= [(a2r[idxA], t[idxT], 0),
             (a2r[idxA_2], t[idxT], 0),
             (a2r[idxA], t[idxT_2], 0),
             (a2r[idxA_2], t[idxT_2], 0)]
    a,b,c,d=bilinear_interpolation(a,b,points)
    return data.T.dot([a,b,c,d]).flatten()
            
a2r=np.linspace(2.5,8.0,20) 
t=np.linspace(0,500000.0,100) 
h=np.linspace(103,0,104)

sectionName='Synthetic2'
modelSolutions=pickle.load(open('MeshSolutions_2_5to8_0_500k.pkl','rb'))

#generate perfect synthetic data
AgeSynth=np.random.uniform(1e6,3e6)
AgeSynth=0.5e6
RRSynth=9e-6
velocitiesSynth=np.random.uniform(.01,2.0,5)
velocitiesSynth=np.linspace(.01,.4,N)


testmeters1=sorted(np.floor(np.random.uniform(0,100,45)))
sAR=np.log10(velocitiesSynth[0]/RRSynth)
print(sAR)
sInject=(AgeSynth*velocitiesSynth[0])/.9
testd13c1=synthData(sAR,sInject,np.sort(np.array(testmeters1).astype(int)))+np.random.normal(0,.2,45)

testmeters2=sorted(np.floor(np.random.uniform(0,100,45)))
sAR=np.log10(velocitiesSynth[1]/RRSynth)
print(sAR)
sInject=(AgeSynth*velocitiesSynth[1])/.9
testd13c2=synthData(sAR,sInject,np.sort(np.array(testmeters2).astype(int)))+np.random.normal(0,.2,45)

testmeters3=sorted(np.floor(np.random.uniform(0,100,45)))
sAR=np.log10(velocitiesSynth[2]/RRSynth)
print(sAR)
sInject=(AgeSynth*velocitiesSynth[2])/.9
testd13c3=synthData(sAR,sInject,np.sort(np.array(testmeters3).astype(int)))+np.random.normal(0,.2,45)

testmeters4=sorted(np.floor(np.random.uniform(0,100,45)))
sAR=np.log10(velocitiesSynth[3]/RRSynth)
print(sAR)
sInject=(AgeSynth*velocitiesSynth[3])/.9
testd13c4=synthData(sAR,sInject,np.sort(np.array(testmeters4).astype(int)))+np.random.normal(0,.2,45)

testmeters5=sorted(np.floor(np.random.uniform(0,100,45)))
sAR=np.log10(velocitiesSynth[4]/RRSynth)
print(sAR)
sInject=(AgeSynth*velocitiesSynth[4])/.9
testd13c5=synthData(sAR,sInject,np.sort(np.array(testmeters5).astype(int)))+np.random.normal(0,.2,45)



#%%
allData=np.array([testd13c1,testd13c2,testd13c3,testd13c4,testd13c5]).T
allMeters=np.array([testmeters1,testmeters2,testmeters3,testmeters4,testmeters5]).T
#initialize pymc stochastic variables

N=5
err = pm.Uniform("err", 0, 500) #uncertainty on d13c values, flat prior

A_to_R=pm.Uniform('A_to_R',2.5,8.0,size=N)
Age=pm.Uniform('Age',.01e6,4e6)
RR=pm.Uniform('RR',1e-8,1e-5) 
#set pymc observations from data
metersModel=pm.Normal("meters", 0, 100, value=allMeters, observed=True, size=(45,N))

@pm.deterministic
def velocity(A_to_R=A_to_R, RR=RR):              
    return RR*(10**A_to_R)

@pm.deterministic
def model_iterations(velocity=velocity, Age=Age):              
    return (Age*velocity)/.9  #.07 comes from the scheme i solved with for the model solutions


#model prediction function from model results (should be able to interpolate in 3d to make this continuous)    
@pm.deterministic
def predd13c(A_to_R=A_to_R, model_iterations=model_iterations, metersModel=metersModel):
    idxA=np.zeros((len(A_to_R)))
    idxA_2=np.zeros((len(A_to_R)))
    idxT=np.zeros((len(A_to_R)))
    idxT_2=np.zeros((len(A_to_R)))
    results=np.zeros((len(A_to_R),len(metersModel)))
    
    for j in range(len(A_to_R)):   
        idxA[j]=np.abs((a2r-A_to_R[j])).argmin()
        if a2r[idxA[j]]>A_to_R[j]:
            idxA_2[j]=idxA[j].copy()
            idxA[j]=idxA[j]-1
        else:
            idxA_2[j]=idxA[j]+1
        idxT[j]=np.abs((t-model_iterations[j])).argmin()
        if t[idxT[j]]>model_iterations[j]:
            idxT_2[j]=idxT[j].copy()
            idxT[j]=idxT[j]-1
        else:
            idxT_2[j]=idxT[j]+1
        injections=model_iterations[j]
        if idxT_2[j]>=100:
            idxT_2[j]=99
            idxT[j]=98
            injections=t[-1]

        
        highResModel1=interp1d(np.linspace(103,0,104),modelSolutions[idxA[j],idxT[j],:])
        highResModel2=interp1d(np.linspace(103,0,104),modelSolutions[idxA_2[j],idxT[j],:])
        highResModel3=interp1d(np.linspace(103,0,104),modelSolutions[idxA[j],idxT_2[j],:])
        highResModel4=interp1d(np.linspace(103,0,104),modelSolutions[idxA_2[j],idxT_2[j],:])    
        data=np.array([highResModel1(metersModel[:,j]),highResModel2(metersModel[:,j]),highResModel3(metersModel[:,j]),highResModel4(metersModel[:,j])]) 
        points= [(a2r[idxA[j]], t[idxT[j]], 0),
                 (a2r[idxA_2[j]], t[idxT[j]], 0),
                 (a2r[idxA[j]], t[idxT_2[j]], 0),
                 (a2r[idxA_2[j]], t[idxT_2[j]], 0)]
        if model_iterations[j]>500000.0:
            data=data/data*-7.0
        a,b,c,d=bilinear_interpolation(A_to_R[j],injections,points)
        results[j,:]=data.T.dot([a,b,c,d]).flatten()
    return results.T

    
obs = pm.Normal("obs", predd13c, err, value=allData, observed=True, size=(45,N))

   
#set up pymc model with all variables, observations, and deterministics
model = pm.Model([predd13c,err,obs,metersModel, A_to_R,velocity,RR,model_iterations,Age])

mcmc=pm.MCMC(model,db='pickle', dbname=('Synthetic_5_diagenesis.pickle'))

#%%
mcmc.sample(200000, 100000,5) #N, burn, thin


#%%
N=5
with plt.style.context('fivethirtyeight'):
    
    rcParams.update({'figure.autolayout': True})
    fig=plt.figure(figsize=(5*N, 9))
    gs = gridspec.GridSpec(3, 1*N+1) 
    for i in range(N):

        timeofd = plt.subplot(gs[2,i])
        strat = plt.subplot(gs[:2,i])
        

        density = scipy.stats.gaussian_kde(mcmc.trace('velocity')[:,i])
        xs = np.linspace(0,.5,200)
        density.covariance_factor = lambda : .4
        density._compute_covariance()
        timeofd.plot(xs,density(xs)/np.trapz(density(xs)),color=plt.rcParams['axes.color_cycle'][3],lw=3)
        timeofd.set_xlabel('Vertical Velocity (m/yr)')
        timeofd.plot([velocitiesSynth[i],velocitiesSynth[i]],[0,(density(xs)/np.trapz(density(xs))).max()*1.1],color=plt.rcParams['axes.color_cycle'][1],lw=2)
        timeofd.set_ylabel('Probability')
        timeofd.set_xlim([0,.5])
        
        
          
        strat.plot(np.array(sorted(predd13c.stats()['quantiles'][50][:,i],reverse=True)),np.sort(metersModel.value[:,i]),color=plt.rcParams['axes.color_cycle'][2],lw=2)
        strat.plot(np.array(sorted(predd13c.stats()['quantiles'][2.5][:,i],reverse=True)),np.sort(metersModel.value[:,i]),color=plt.rcParams['axes.color_cycle'][2],lw=2,alpha=.3) 
        strat.plot(np.array(sorted(predd13c.stats()['quantiles'][97.5][:,i],reverse=True)),np.sort(metersModel.value[:,i]),color=plt.rcParams['axes.color_cycle'][2],lw=2,alpha=.3) 
        strat.plot(np.array(allData[:,i]),np.array(allMeters[:,i]),linestyle="None",marker='.',markeredgecolor='none',markersize=10,alpha=.8)          
        strat.set_xlabel('$\delta^{13}$C')
        strat.set_ylabel('Meters')
        strat.set_title(sectionName+': A_to_R='+str(int(velocitiesSynth[i]/RRSynth)))

ator = plt.subplot(gs[1,-1])
RRate = plt.subplot(gs[2,-1])     
errPlot = plt.subplot(gs[0,-1]) 

density = scipy.stats.gaussian_kde(np.sqrt(1/mcmc.trace('err')[:]))
xs = np.linspace(0,1,200)
density.covariance_factor = lambda : .4
density._compute_covariance()
errPlot.plot(xs,density(xs)/np.trapz(density(xs)),color=plt.rcParams['axes.color_cycle'][3],lw=3)
errPlot.plot([0.2,0.2],[0,(density(xs)/np.trapz(density(xs))).max()*1.1],color=plt.rcParams['axes.color_cycle'][1],lw=2)
errPlot.set_xlabel('$\sigma$ error')
errPlot.set_ylabel('Probability')
errPlot.set_xlim([0,1])



density = scipy.stats.gaussian_kde(mcmc.trace('RR')[:]*1e6)
xs = np.linspace(1e-1,2e1,200)
density.covariance_factor = lambda : .4
density._compute_covariance()
ator.plot(xs,density(xs)/np.trapz(density(xs)),color=plt.rcParams['axes.color_cycle'][3],lw=3)
ator.plot([RRSynth*1e6,RRSynth*1e6],[0,(density(xs)/np.trapz(density(xs))).max()*1.1],color=plt.rcParams['axes.color_cycle'][1],lw=2)
ator.set_xlabel('% Rock Reacted per Ma')
ator.set_ylabel('Probability')
ator.set_xlim([1e-2,2e1])


density = scipy.stats.gaussian_kde(mcmc.trace('Age')[:]/1e6)
xs = np.linspace(0,4.0,200)
density.covariance_factor = lambda : .4
density._compute_covariance()
RRate.plot(xs,density(xs)/np.trapz(density(xs)),color=plt.rcParams['axes.color_cycle'][3],lw=3)
RRate.set_xlabel('Duration of Diagenesis (Ma)')
RRate.plot([AgeSynth/1e6,AgeSynth/1e6],[0,(density(xs)/np.trapz(density(xs))).max()*1.1],color=plt.rcParams['axes.color_cycle'][1],lw=2)
RRate.set_ylabel('Probability')
RRate.set_xlim([0,4.0])   
    
fig.savefig((sectionName+'200ktest_2noise.pdf'), format='pdf', dpi=300)  





mcmc.db.close()

#db = pymc.database.pickle.load('Disaster.pickle')