

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
            
            
a2r=np.linspace(2.5,8.0,20) 
t=np.linspace(0,500000.0,100) 
h=np.linspace(103,0,104)


#load datasets and clean
ArrowCanyon=pd.read_csv('samples_ArrowCanyon.csv')
BattleshipWash=pd.read_csv('samples_BattleshipWash.csv')

#arrow canyon
ACd13c=(ArrowCanyon.d13c[ArrowCanyon.SAMP_HEIGHT<103])
BWd13c=(BattleshipWash.d13c[(((BattleshipWash.SAMP_HEIGHT-240)<103) & ((BattleshipWash.SAMP_HEIGHT-240)>0))])
ACd13c=list(ACd13c)+list(BWd13c)
ACmeters=(ArrowCanyon.SAMP_HEIGHT[ArrowCanyon.SAMP_HEIGHT<103])
BWmeters=(BattleshipWash.SAMP_HEIGHT[(((BattleshipWash.SAMP_HEIGHT-240)<103) & ((BattleshipWash.SAMP_HEIGHT-240)>0))]-240.0)
ACmeters=list(ACmeters)+list(BWmeters)

modelSolutions=pickle.load(open('MeshSolutions_2_5to8_0_500k.pkl','rb'))

def predFunct(A_to_R, model_iterations, metersModel):
    if model_iterations<0:
        model_iterations=0
    idxA=np.abs((a2r-A_to_R)).argmin()
    if a2r[idxA]>A_to_R:
        idxA_2=idxA.copy()
        idxA=idxA-1
    else:
        idxA_2=idxA+1
    idxT=np.abs((t-model_iterations)).argmin()
    if t[idxT]>model_iterations:
        idxT_2=idxT.copy()
        idxT=idxT-1
    else:
        idxT_2=idxT+1
    injections=model_iterations
    if idxT_2>=100:
        idxT_2=99
        idxT=98
        injections=t[-1]

    
    highResModel1=interp1d(np.linspace(103,0,104),modelSolutions[idxA,idxT,:])
    highResModel2=interp1d(np.linspace(103,0,104),modelSolutions[idxA_2,idxT,:])
    highResModel3=interp1d(np.linspace(103,0,104),modelSolutions[idxA,idxT_2,:])
    highResModel4=interp1d(np.linspace(103,0,104),modelSolutions[idxA_2,idxT_2,:])    
    data=np.array([highResModel1(metersModel),highResModel2(metersModel),highResModel3(metersModel),highResModel4(metersModel)]) 
    points= [(a2r[idxA], t[idxT], 0),
             (a2r[idxA_2], t[idxT], 0),
             (a2r[idxA], t[idxT_2], 0),
             (a2r[idxA_2], t[idxT_2], 0)]
    if model_iterations>500000.0:
        data=data/data*-7.0
    a,b,c,d=bilinear_interpolation(A_to_R,injections,points)
    results=data.T.dot([a,b,c,d]).flatten()
    return results.T

#%%

#initialize pymc stochastic variables

N=1
names=['AC']
err1 = pm.Uniform("err1", 0, 500) #uncertainty on d13c values, flat prior
Age=pm.Normal('Age',2.5e6,4e-12)
RR=pm.Uniform('RR',1e-8,1e-5) 

AR1=pm.Uniform('ACAR',2.5,8.0)


#set pymc observations from data
heights1=pm.Normal("ACmeters", 0, 100, value=ACmeters, observed=True)

@pm.deterministic
def velocity1(AR=AR1, RR=RR):              
    return RR*(10**AR)

@pm.deterministic
def model_iterations1(V=velocity1, Age=Age):              
    return (Age*V)/.9  #.07 comes from the scheme i solved with for the model solutions

@pm.deterministic
def predACd13c(AR=AR1, injections=model_iterations1, heights=heights1):
    return predFunct(AR,injections,heights)
#model prediction function from model results (should be able to interpolate in 3d to make this continuous)    
    
obs1 = pm.Normal("ACd13c", predACd13c, err1, value=ACd13c, observed=True)
   
#set up pymc model with all variables, observations, and deterministics
model = pm.Model([Age,RR,err1,
                  predACd13c,
                  obs1,
                  heights1,
                  AR1,
                  velocity1,
                  model_iterations1])




map_start=pm.MAP(model)
map_start.fit()
mcmc=pm.MCMC(model,db='pickle', dbname=('ArrowCanyon_diagenesis.pickle'))
#mcmc=pm.MCMC(model)
#mcmc.use_step_method(pm.Metropolis, Age, proposal_distribution='Normal', proposal_sd=1e6)
#mcmc.use_step_method(pm.AdaptiveMetropolis, [RR,Age],delay=20000,interval=20000,shrink_if_necessary=True)
#mcmc.use_step_method(pm.AdaptiveMetropolis, [AR1,AR2,AR3,AR4,AR5,AR6,AR7])


#mcmc.use_step_method(pm.AdaptiveMetropolis, [RR])

#%%
mcmc.sample(10000,0,1) #N, burn, thin


#%%
N=1
bandwidth=.1
extraburn=0
thin=10
with plt.style.context('fivethirtyeight'):
    
    rcParams.update({'figure.autolayout': True})
    rcParams.update({'font.size': 8.0})
    rcParams.update({'figure.subplot.bottom': 0})
    rcParams.update({'figure.subplot.hspace': 0.0})
    rcParams.update({'figure.subplot.left': 0.0})
    rcParams.update({'figure.subplot.right': 1.0})
    rcParams.update({'figure.subplot.top': 1.0})
    rcParams.update({'figure.subplot.wspace': 0.0})
    
    fig=plt.figure(figsize=(7, 5.5))
    gs = gridspec.GridSpec(3, 1*N+2) 
    allData=[[ACd13c]]
    allHeights=[[ACmeters]]
    fits_50=[[np.array(sorted(np.percentile(mcmc.trace('predACd13c')[extraburn::thin,:],50,axis=0),reverse=True))]]
                   
                   
                
    fits_2_5=[[np.array(sorted(np.percentile(mcmc.trace('predACd13c')[extraburn::thin,:],2.5,axis=0),reverse=True))]]     
     
    fits_97_5=[[np.array(sorted(np.percentile(mcmc.trace('predACd13c')[extraburn::thin,:],97.5,axis=0),reverse=True))]]    
                  
                   
    heights_50=[[np.sort(heights1.value)]]
    
    heights_50=heights_50[:N]
    
    for i in range(N):

        timeofd = plt.subplot(gs[2,i])
        strat = plt.subplot(gs[:2,i])
        numbers=[1,2,3,6,7]
        density = scipy.stats.gaussian_kde(mcmc.trace('velocity'+str(numbers[i]))[extraburn::thin])
        xs = np.linspace(0,mcmc.trace('velocity'+str(numbers[i]))[extraburn::thin].max()*1.5,1000)
        density.covariance_factor = lambda : bandwidth
        density._compute_covariance()
        timeofd.plot(xs,density(xs)/np.trapz(density(xs)),color=plt.rcParams['axes.color_cycle'][i],lw=3)
        timeofd.set_xlabel('Velocity (m yr$^{-1}$)')
        timeofd.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
        timeofd.set_ylabel('Probability')
        
        xmax=[150,500,500,150,300]
        #timeofd.set_xlim([0,xmax[i]])
        timeofd.locator_params(axis = 'x',nbins=4)
        timeofd.set_ylim([0,(density(xs)/np.trapz(density(xs))).max()*1.1])
         
        strat.plot(np.array(fits_50[i][0]),np.array(heights_50[i][0]),color='#6c71c4',lw=2)
        strat.plot(np.array(fits_2_5[i][0]),np.array(heights_50[i][0]),color='#6c71c4',lw=2,alpha=.3) 
        strat.plot(np.array(fits_97_5[i][0]),np.array(heights_50[i][0]),color='#6c71c4',lw=2,alpha=.3) 
        strat.plot(np.array(allData[i][0]),np.array(allHeights[i][0]),linestyle="None",marker='o',markeredgecolor='none',alpha=.4,markersize=7,color=plt.rcParams['axes.color_cycle'][i])          
        strat.set_xlabel('$\delta^{13}$C')
        strat.set_ylabel('Meters')
        strat.set_title(names[i])
        strat.set_xlim([-8,3])
        strat.set_ylim([0,120])
        
    ator = plt.subplot(gs[1,-2])
    RRate = plt.subplot(gs[2,-2]) 
    errPlot = plt.subplot(gs[0,-2:]) 
    
    for k in range(N):
        numbers=[1,2,3,6,7]
        density = scipy.stats.gaussian_kde(np.sqrt(1/mcmc.trace('err'+str(numbers[k]))[extraburn::thin]))
        xs = np.linspace(0,np.sqrt(1/(mcmc.trace('err'+str(numbers[k]))[extraburn::thin])).max(),200)
        density.covariance_factor = lambda : bandwidth
        density._compute_covariance()
        errPlot.plot(xs,density(xs)/np.trapz(density(xs)),color=plt.rcParams['axes.color_cycle'][k],lw=2)
        errPlot.set_xlabel('$\sigma$ error')
        errPlot.set_ylabel('Probability')
        #errPlot.set_ylim([0,(density(xs)/np.trapz(density(xs))).max()*1.1])
    
    
    density = scipy.stats.gaussian_kde(mcmc.trace('RR')[extraburn::thin])
    xs = np.linspace(0,(mcmc.trace('RR')[extraburn::thin]).max()*1.1,200)
    rrDensity=density
    rrxs=xs
    density.covariance_factor = lambda : bandwidth
    density._compute_covariance()
    ator.plot(xs,density(xs)/np.trapz(density(xs)),color='#6c71c4',lw=3)
    ator.set_xlabel('Reaction Rate yr$^{-1}$')
    ator.set_ylabel('Probability')
    ator.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
    ator.locator_params(axis = 'x',nbins=4)
    #ator.set_xlim([0,2e1])
    ator.set_ylim([0,(density(xs)/np.trapz(density(xs))).max()*1.1])
    
    
    density = scipy.stats.gaussian_kde(mcmc.trace('Age')[extraburn::thin])
    xs = np.linspace(0,mcmc.trace('Age')[extraburn::thin].max()*1.1,200)
    ageDensity=density
    agexs=xs
    density.covariance_factor = lambda : bandwidth
    density._compute_covariance()
    RRate.plot(xs,density(xs)/np.trapz(density(xs)),color='#6c71c4',lw=3)
    RRate.set_xlabel('Duration (yr)')
    RRate.set_ylabel('Probability')
    RRate.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
    RRate.locator_params(axis = 'x',nbins=4)
    RRate.set_xlim([0,5.0e6])  
    RRate.set_ylim([0,(density(xs)/np.trapz(density(xs))).max()*1.1])  
    
    atorP = plt.subplot(gs[1,-1])
    xs = np.linspace(1e-8,1e-5,200)
    atorP.plot(np.log10(xs),np.ones(200)*.006,color='#cb4b16',lw=3)
    atorP.set_xlabel('Log Reaction Rate yr$^{-1}$')
    atorP.set_ylabel('Probability')
    atorP.set_title('Priors')
    atorP.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
    atorP.locator_params(axis = 'x',nbins=4)
    atorP.plot(np.log10(rrxs),rrDensity(rrxs)/np.trapz(rrDensity(rrxs)),color='#6c71c4',lw=2)
    atorP.set_xlim(np.log10([1e-8,1e-5]))  
    atorP.set_ylim([0,.05])  
    
    RRateP = plt.subplot(gs[2,-1]) 
    density = scipy.stats.gaussian_kde(np.random.normal(2.5e6,.5e6,10000))
    xs = np.linspace(.1,5e6,200)
    density.covariance_factor = lambda : .5    
    density._compute_covariance()
    RRateP.plot(xs,density(xs)/np.trapz(density(xs)),color='#cb4b16',lw=3)
    RRateP.set_xlabel('Duration (yr)')
    RRateP.set_ylabel('Probability')
    RRateP.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
    RRateP.plot(agexs,ageDensity(agexs)/np.trapz(ageDensity(agexs)),color='#6c71c4',lw=2)
    RRateP.locator_params(axis = 'x',nbins=4)
    RRateP.set_xlim([0,5e6])  
    RRateP.set_ylim([0,(density(xs)/np.trapz(density(xs))).max()*1.1])  




#    
fig.savefig(('AC_tset.pdf'), format='pdf', dpi=300)  
#
#
#
#
#
mcmc.db.close()

#db = pymc.database.pickle.load('Disaster.pickle')

#%%
#orange=#cb4b16
#magenta=#d33682
def tracePlot(mcmc,extraburn=0,thin=1):
    with plt.style.context('fivethirtyeight'):        
        rcParams.update({'figure.autolayout': True})
        rcParams.update({'font.size': 10.0})
        rcParams.update({'figure.subplot.bottom': 0})
        rcParams.update({'figure.subplot.hspace': 0.0})
        rcParams.update({'figure.subplot.left': 0.0})
        rcParams.update({'figure.subplot.right': 1.0})
        rcParams.update({'figure.subplot.top': 1.0})
        rcParams.update({'figure.subplot.wspace': 0.0})
        fig=plt.figure(figsize=(16, 8.5))
        gs = gridspec.GridSpec(4,1)
        listTraces=[['RR','Age','ACAR','err1']]
        for k in range(4):
                tracePlot=plt.subplot(gs[k,0])
                trace=mcmc.trace(listTraces[0][k])[extraburn::thin]
                tracePlot.plot(range(trace.size),trace,lw=1)
                tracePlot.set_title(listTraces[0][k])
        fig.savefig(('AC_tset_traceFig.pdf'), format='pdf', dpi=300)  
        
#%%
variables=['Age','RR','ACAR','err1']
for i in variables:
    burn=0
    thin=1
    trace=mcmc.trace(i)[burn::thin]
    test=pm.geweke(trace, intervals=20)
    pm.Matplot.geweke_plot(test,'test',lw=1)
    
