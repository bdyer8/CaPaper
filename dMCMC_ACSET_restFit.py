

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
B503=pd.read_csv('samples_B503.csv')
B211=pd.read_csv('samples_B211.csv')
SanAndres=pd.read_csv('samples_SanAndres.csv')
Leadville=pd.read_csv('samples_B416.csv')
Leadville2=pd.read_csv('samples_B417.csv')
BattleshipWash=pd.read_csv('samples_BattleshipWash.csv')
B402=pd.read_csv('samples_B402.csv')

#B402 (Strawberry Creek)
B402meters=np.array(B402.SAMP_HEIGHT+6.0)
B402d13c=np.array(B402.d13c-1.0)

#B503 (CrazyWoman)
B503meters=np.array(B503.SAMP_HEIGHT-27.0)
B503d13c=np.array(B503.d13c)

#B211 (Clark's Fork)
B211meters=np.array(B211.SAMP_HEIGHT-124.0)
B211d13c=np.array(B211.d13c)

#leadville 1
LV1meters=np.array(Leadville.SAMP_HEIGHT[Leadville.d13c<0]+9.0)
LV1d13c=np.array(Leadville.d13c[Leadville.d13c<0])

#Leadville 2
LV2meters=np.array(Leadville2.SAMP_HEIGHT+18)
LV2d13c=np.array(Leadville2.d13c)

#arrow canyon
ACd13c=(ArrowCanyon.d13c[ArrowCanyon.SAMP_HEIGHT<103])
BWd13c=(BattleshipWash.d13c[(((BattleshipWash.SAMP_HEIGHT-240)<103) & ((BattleshipWash.SAMP_HEIGHT-240)>0))])
ACd13c=list(ACd13c)+list(BWd13c)
ACmeters=(ArrowCanyon.SAMP_HEIGHT[ArrowCanyon.SAMP_HEIGHT<103])
BWmeters=(BattleshipWash.SAMP_HEIGHT[(((BattleshipWash.SAMP_HEIGHT-240)<103) & ((BattleshipWash.SAMP_HEIGHT-240)>0))]-240.0)
ACmeters=list(ACmeters)+list(BWmeters)

#san andres
SAmeters=np.array(SanAndres.SAMP_HEIGHT[SanAndres.SAMP_HEIGHT>240.1]-240.0)
SAd13c=np.array(SanAndres.d13c[SanAndres.SAMP_HEIGHT>240.1])

sectionName='Synthetic2'
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

N=6
names=['AC','LV1','LV2','B503','B211','SA']
err1 = pm.Uniform("err1", 0, 500) #uncertainty on d13c values, flat prior
err2 = pm.Uniform("err2", 0, 500) #uncertainty on d13c values, flat prior
err3 = pm.Uniform("err3", 0, 500) #uncertainty on d13c values, flat prior
err4 = pm.Uniform("err4", 0, 500) #uncertainty on d13c values, flat prior
err5 = pm.Uniform("err5", 0, 500) #uncertainty on d13c values, flat prior
err6 = pm.Uniform("err6", 0, 500) #uncertainty on d13c values, flat prior
err7 = pm.Uniform("err7", 0, 500) #uncertainty on d13c values, flat prior
Age1=pm.Normal('Age1',2.5e6,4e-12)
Age2=pm.Uniform('Age2',.1e6,40e6)
Age3=Age2
Age4=pm.Uniform('Age4',.1e6,40e6)
Age5=pm.Uniform('Age5',.1e6,40e6)
Age6=pm.Uniform('Age6',.1e6,40e6)
Age7=pm.Uniform('Age7',.1e6,40e6)
RR=pm.Normal('RR',3.5703864479316496e-07,68683619419949.766,value=3.5703864479316496e-07,observed=True) 

AR1=pm.Uniform('ACAR',2.5,8.0)
AR2=pm.Uniform('LV1AR',2.5,8.0)
AR3=pm.Uniform('LV2AR',2.5,8.0)
AR4=pm.Uniform('B503AR',2.5,8.0)
AR5=pm.Uniform('B402AR',2.5,8.0)
AR6=pm.Uniform('B211AR',2.5,8.0)
AR7=pm.Uniform('SAAR',2.5,8.0)


#set pymc observations from data
heights1=pm.Normal("ACmeters", 0, 100, value=ACmeters, observed=True)
heights2=pm.Normal("LV1meters", 0, 100, value=LV1meters, observed=True)
heights3=pm.Normal("LV2meters", 0, 100, value=LV2meters, observed=True)
heights4=pm.Normal("B503meters", 0, 100, value=B503meters, observed=True)
heights5=pm.Normal("B402meters", 0, 100, value=B402meters, observed=True)
heights6=pm.Normal("B211meters", 0, 100, value=B211meters, observed=True)
heights7=pm.Normal("SAmeters", 0, 100, value=SAmeters, observed=True)

@pm.deterministic
def velocity1(AR=AR1, RR=RR):              
    return np.abs(RR)*(10**AR)
@pm.deterministic
def velocity2(AR=AR2, RR=RR):              
    return np.abs(RR)*(10**AR)
@pm.deterministic
def velocity3(AR=AR3, RR=RR):              
    return np.abs(RR)*(10**AR)
@pm.deterministic
def velocity4(AR=AR4, RR=RR):              
    return np.abs(RR)*(10**AR)
@pm.deterministic
def velocity5(AR=AR5, RR=RR):              
    return np.abs(RR)*(10**AR)
@pm.deterministic
def velocity6(AR=AR6, RR=RR):              
    return np.abs(RR)*(10**AR)
@pm.deterministic
def velocity7(AR=AR7, RR=RR):              
    return np.abs(RR)*(10**AR)

@pm.deterministic
def model_iterations1(V=velocity1, Age=Age1):              
    return (Age*V)/.9  #.07 comes from the scheme i solved with for the model solutions
@pm.deterministic
def model_iterations2(V=velocity2, Age=Age2):              
    return (Age*V)/.9  #.07 comes from the scheme i solved with for the model solutions
@pm.deterministic
def model_iterations3(V=velocity3, Age=Age3):              
    return (Age*V)/.9  #.07 comes from the scheme i solved with for the model solutions
@pm.deterministic
def model_iterations4(V=velocity4, Age=Age4):              
    return (Age*V)/.9  #.07 comes from the scheme i solved with for the model solutions
@pm.deterministic
def model_iterations5(V=velocity5, Age=Age5):              
    return (Age*V)/.9  #.07 comes from the scheme i solved with for the model solutions
@pm.deterministic
def model_iterations6(V=velocity6, Age=Age6):              
    return (Age*V)/.9  #.07 comes from the scheme i solved with for the model solutions
@pm.deterministic
def model_iterations7(V=velocity7, Age=Age7):              
    return (Age*V)/.9  #.07 comes from the scheme i solved with for the model solutions

@pm.deterministic
def predACd13c(AR=AR1, injections=model_iterations1, heights=heights1):
    return predFunct(AR,injections,heights)
@pm.deterministic
def predLV1d13c(AR=AR2, injections=model_iterations2, heights=heights2):
    return predFunct(AR,injections,heights)
@pm.deterministic
def predLV2d13c(AR=AR3, injections=model_iterations3, heights=heights3):
    return predFunct(AR,injections,heights)
@pm.deterministic
def predB503d13c(AR=AR4, injections=model_iterations4, heights=heights4):
    return predFunct(AR,injections,heights)
@pm.deterministic
def predB402d13c(AR=AR5, injections=model_iterations5, heights=heights5):
    return predFunct(AR,injections,heights)
@pm.deterministic
def predB211d13c(AR=AR6, injections=model_iterations6, heights=heights6):
    return predFunct(AR,injections,heights)
@pm.deterministic
def predSAd13c(AR=AR7, injections=model_iterations7, heights=heights7):
    return predFunct(AR,injections,heights)
#model prediction function from model results (should be able to interpolate in 3d to make this continuous)    
    
obs1 = pm.Normal("ACd13c", predACd13c, err1, value=ACd13c, observed=True)
obs2 = pm.Normal("LV1d13c", predLV1d13c, err2, value=LV1d13c, observed=True)
obs3 = pm.Normal("LV2d13c", predLV2d13c, err3, value=LV2d13c, observed=True)
obs4 = pm.Normal("B503d13c", predB503d13c, err4, value=B503d13c, observed=True)
obs5 = pm.Normal("B402d13c", predB402d13c, err5, value=B402d13c, observed=True)
obs6 = pm.Normal("B211d13c", predB211d13c, err6, value=B211d13c, observed=True)
obs7 = pm.Normal("SAd13c", predSAd13c, err7, value=SAd13c, observed=True)

   
#set up pymc model with all variables, observations, and deterministics
model = pm.Model([Age1,Age2,Age3,Age4,Age6,Age7,RR,err1,err2,err3,err4,err5,err6,err7,
                  predACd13c,predLV1d13c,predLV2d13c,predB503d13c,predB211d13c,predSAd13c,
                  obs1,obs2,obs3,obs4,obs6,obs7,
                  heights1,heights2,heights3,heights4,heights6,heights7, 
                  AR1,AR2,AR3,AR4,AR6,AR7,
                  velocity1,velocity2,velocity3,velocity4,velocity6,velocity7,
                  model_iterations1,model_iterations2,model_iterations3,model_iterations4,model_iterations6,model_iterations7])

#model = pm.Model([Age,RR,err1,err2,err3,err6,
#                  predACd13c,predLV1d13c,predLV2d13c,predB211d13c,
#                  obs1,obs2,obs3,obs6,
#                  heights1,heights2,heights3,heights6, 
#                  AR1,AR2,AR3,AR6,
#                  velocity1,velocity2,velocity3,velocity6,
#                  model_iterations1,model_iterations2,model_iterations3,model_iterations6])




map_start=pm.MAP(model)
map_start.fit()
mcmc=pm.MCMC(model,db='pickle', dbname=('6Sections_Diagenesis_ACFIT.pickle'))
#mcmc=pm.MCMC(model)
#mcmc.use_step_method(pm.Metropolis, Age, proposal_distribution='Normal', proposal_sd=.5e6)
#mcmc.use_step_method(pm.AdaptiveMetropolis, [RR,Age],delay=20000,interval=20000,shrink_if_necessary=True)
#mcmc.use_step_method(pm.AdaptiveMetropolis, [Age2,AR2])


#mcmc.use_step_method(pm.AdaptiveMetropolis, [RR])

#%%
mcmc.sample(100000,0,100) #N, burn, thin


#%%
N=6
bandwidth=.1
extraburn=0
thin=1
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

    
    fig=plt.figure(figsize=(16, 5.5))
    gs = gridspec.GridSpec(3, 1*N+2) 
    allData=[[ACd13c],[LV1d13c],[LV2d13c],[B503d13c],[B211d13c],[SAd13c]]
    allHeights=[[ACmeters],[LV1meters],[LV2meters],[B503meters],[B211meters],[SAmeters]]
    fits_50=[[np.array(sorted(np.percentile(mcmc.trace('predACd13c')[extraburn::thin,:],50,axis=0),reverse=True))],
              [np.array(sorted(np.percentile(mcmc.trace('predLV1d13c')[extraburn::thin,:],50,axis=0),reverse=True))],
               [np.array(sorted(np.percentile(mcmc.trace('predLV2d13c')[extraburn::thin,:],50,axis=0),reverse=True))],
                [np.array(sorted(np.percentile(mcmc.trace('predB503d13c')[extraburn::thin,:],50,axis=0),reverse=True))],
                  [np.array(sorted(np.percentile(mcmc.trace('predB211d13c')[extraburn::thin,:],50,axis=0),reverse=True))],
                   [np.array(sorted(np.percentile(mcmc.trace('predSAd13c')[extraburn::thin,:],50,axis=0),reverse=True))]]
                   
                   
                
    fits_2_5=[[np.array(sorted(np.percentile(mcmc.trace('predACd13c')[extraburn::thin,:],2.5,axis=0),reverse=True))],
              [np.array(sorted(np.percentile(mcmc.trace('predLV1d13c')[extraburn::thin,:],2.5,axis=0),reverse=True))],
               [np.array(sorted(np.percentile(mcmc.trace('predLV2d13c')[extraburn::thin,:],2.5,axis=0),reverse=True))],
                [np.array(sorted(np.percentile(mcmc.trace('predB503d13c')[extraburn::thin,:],2.5,axis=0),reverse=True))],
                  [np.array(sorted(np.percentile(mcmc.trace('predB211d13c')[extraburn::thin,:],2.5,axis=0),reverse=True))],
                   [np.array(sorted(np.percentile(mcmc.trace('predSAd13c')[extraburn::thin,:],2.5,axis=0),reverse=True))]]     
     
    fits_97_5=[[np.array(sorted(np.percentile(mcmc.trace('predACd13c')[extraburn::thin,:],97.5,axis=0),reverse=True))],
              [np.array(sorted(np.percentile(mcmc.trace('predLV1d13c')[extraburn::thin,:],97.5,axis=0),reverse=True))],
               [np.array(sorted(np.percentile(mcmc.trace('predLV2d13c')[extraburn::thin,:],97.5,axis=0),reverse=True))],
                [np.array(sorted(np.percentile(mcmc.trace('predB503d13c')[extraburn::thin,:],97.5,axis=0),reverse=True))],
                  [np.array(sorted(np.percentile(mcmc.trace('predB211d13c')[extraburn::thin,:],97.5,axis=0),reverse=True))],
                   [np.array(sorted(np.percentile(mcmc.trace('predSAd13c')[extraburn::thin,:],97.5,axis=0),reverse=True))]]    
                  
                   
    heights_50=[[np.sort(heights1.value)],
                 [np.sort(heights2.value)],
                 [np.sort(heights3.value)],
                  [np.sort(heights4.value)],
                 [np.sort(heights6.value)],
                  [np.sort(heights7.value)]]
    
    heights_50=heights_50[:N]
    
    for i in range(N):

        timeofd = plt.subplot(gs[2,i])
        strat = plt.subplot(gs[:2,i])
        numbers=[1,2,3,4,6,7]
        density = scipy.stats.gaussian_kde(mcmc.trace('velocity'+str(numbers[i]))[extraburn::thin])
        xs = np.linspace(mcmc.trace('velocity'+str(numbers[i]))[extraburn::thin].min(),mcmc.trace('velocity'+str(numbers[i]))[extraburn::thin].max()*.95,1000)
        density.covariance_factor = lambda : bandwidth
        density._compute_covariance()
        timeofd.plot(xs,density(xs)/np.trapz(density(xs)),color=plt.rcParams['axes.color_cycle'][i],lw=3)
        timeofd.set_xlabel('Velocity (m yr$^{-1}$)')
        timeofd.ticklabel_format(axis='x', style='sci', scilimits=(0,2))
        timeofd.locator_params(axis = 'x',nbins=4)
        timeofd.set_ylabel('Probability')
        
        timeofd.set_ylim([0,.015])
        if i==5:
            timeofd.set_xlim([0,.05])
         
        strat.plot(np.array(fits_50[i][0]),np.array(heights_50[i][0]),color='#6c71c4',lw=2)
        strat.plot(np.array(fits_2_5[i][0]),np.array(heights_50[i][0]),color='#6c71c4',lw=2,alpha=.3) 
        strat.plot(np.array(fits_97_5[i][0]),np.array(heights_50[i][0]),color='#6c71c4',lw=2,alpha=.3) 
        strat.plot(np.array(allData[i][0]),np.array(allHeights[i][0]),linestyle="None",marker='o',markeredgecolor='none',alpha=.4,markersize=7,color=plt.rcParams['axes.color_cycle'][i])          
        strat.set_xlabel('$\delta^{13}$C')
        strat.set_ylabel('Meters')
        strat.set_title(names[i])
        strat.set_xlim([-8,4])
        strat.set_ylim([0,120])
        
    #ator = plt.subplot(gs[1,-2])
    RRate = plt.subplot(gs[2,-2:]) 
    errPlot = plt.subplot(gs[1,-2:]) 
    #    
    #    density = scipy.stats.gaussian_kde(np.sqrt(1/mcmc.trace('err')[extraburn:]))
    #    xs = np.linspace(0,np.sqrt(1/mcmc.trace('err')[extraburn:]).max(),200)
    #    density.covariance_factor = lambda : bandwidth
    #    density._compute_covariance()
    #    errPlot.plot(xs,density(xs)/np.trapz(density(xs)),color=plt.rcParams['axes.color_cycle'][3],lw=3)
    #    errPlot.set_xlabel('$\sigma$ error')
    #    errPlot.set_ylabel('Probability')
    #    errPlot.set_ylim([0,(density(xs)/np.trapz(density(xs))).max()*1.1])
    #    #errPlot.set_xlim([0,1])
    
    for k in range(N):
        numbers=[1,2,3,4,6,7]
        density = scipy.stats.gaussian_kde(np.sqrt(1/mcmc.trace('err'+str(numbers[k]))[extraburn::thin]))
        xs = np.linspace(0,np.sqrt(1/(mcmc.trace('err'+str(numbers[k]))[extraburn::thin])).max(),200)
        density.covariance_factor = lambda : bandwidth
        density._compute_covariance()
        errPlot.plot(xs,density(xs)/np.trapz(density(xs)),color=plt.rcParams['axes.color_cycle'][k],lw=2)
        errPlot.set_xlabel('$\sigma$ error')
        errPlot.set_ylabel('Probability')
        
        numbers=[1,2,2,4,6,7]
        density = scipy.stats.gaussian_kde((mcmc.trace('Age'+str(numbers[k]))[extraburn::thin]/1e6))
        xs = np.linspace(0,((mcmc.trace('Age'+str(numbers[k]))[extraburn::thin])/1e6).max(),200)
        density.covariance_factor = lambda : bandwidth
        density._compute_covariance()
        RRate.plot(xs,density(xs)/np.trapz(density(xs)),color=plt.rcParams['axes.color_cycle'][k],lw=2)
        RRate.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
        RRate.locator_params(axis = 'x',nbins=5)
        RRate.set_xlabel('Duration of Diagenesis (Ma)')
        RRate.set_ylabel('Probability')
        RRate.set_xlim([0,10])
        #errPlot.set_ylim([0,(density(xs)/np.trapz(density(xs))).max()*1.1])
    
    
#    density = scipy.stats.gaussian_kde(mcmc.trace('RR')[extraburn::thin])
#    xs = np.linspace(0,(mcmc.trace('RR')[extraburn::thin]).max()*1.1,200)
#    rrDensity=density
#    rrxs=xs
#    density.covariance_factor = lambda : bandwidth
#    density._compute_covariance()
#    ator.plot(xs,density(xs)/np.trapz(density(xs)),color='#6c71c4',lw=3)
#    ator.set_xlabel('Reaction Rate yr$^{-1}$')
#    ator.set_ylabel('Probability')
#    ator.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
#    ator.locator_params(axis = 'x',nbins=4)
#    #ator.set_xlim([0,2e1])
#    ator.set_ylim([0,(density(xs)/np.trapz(density(xs))).max()*1.1])
    
    
#    density = scipy.stats.gaussian_kde(mcmc.trace('Age')[extraburn::thin])
#    xs = np.linspace(0,mcmc.trace('Age')[extraburn::thin].max()*1.1,200)
#    ageDensity=density
#    agexs=xs
#    density.covariance_factor = lambda : bandwidth
#    density._compute_covariance()
#    RRate.plot(xs,density(xs)/np.trapz(density(xs)),color='#6c71c4',lw=3)
#    RRate.set_xlabel('Duration (yr)')
#    RRate.set_ylabel('Probability')
#    RRate.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
#    RRate.locator_params(axis = 'x',nbins=4)
    #RRate.set_xlim([0,2.0])  
    #RRate.set_ylim([0,(density(xs)/np.trapz(density(xs))).max()*1.1])  
    
#    atorP = plt.subplot(gs[1,-1])
#    xs = np.linspace(1e-8,1e-5,200)
#    atorP.plot(xs,np.ones(200)*.006,color='#cb4b16',lw=3)
#    atorP.set_xlabel('Reaction Rate yr$^{-1}$')
#    atorP.set_ylabel('Probability')
#    atorP.set_title('Priors')
#    atorP.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
#    atorP.locator_params(axis = 'x',nbins=4)
#    atorP.plot(rrxs,rrDensity(rrxs)/np.trapz(rrDensity(rrxs)),color='#6c71c4',lw=2)
#    atorP.set_xlim([1e-8,1e-5])  
#    atorP.set_ylim([0,.05])  
    
#    RRateP = plt.subplot(gs[2,-1]) 
#    density = scipy.stats.gaussian_kde(np.random.normal(2.5e6,.5e6,10000))
#    xs = np.linspace(.1,5e6,200)
#    density.covariance_factor = lambda : .5    
#    density._compute_covariance()
#    RRateP.plot(xs,density(xs)/np.trapz(density(xs)),color='#cb4b16',lw=3)
#    RRateP.set_xlabel('Duration (yr)')
#    RRateP.set_ylabel('Probability')
#    RRateP.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
#    RRateP.plot(agexs,ageDensity(agexs)/np.trapz(ageDensity(agexs)),color='#6c71c4',lw=2)
#    RRateP.locator_params(axis = 'x',nbins=4)
#    RRateP.set_xlim([0,5e6])  
#    RRateP.set_ylim([0,(density(xs)/np.trapz(density(xs))).max()*1.5])  


#errPlotP = plt.subplot(gs[0,-1]) 


#    
fig.savefig(('6SectionsData_tset_ACSET.pdf'), format='pdf', dpi=300)  
#
#
#
#
#
#mcmc.db.close()

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
        fig=plt.figure(figsize=(21, 10.5))
        gs = gridspec.GridSpec(6,3)
        listTraces=[['Age1','Age2','Age2','Age4','Age6','Age7'],['ACAR','LV1AR','LV2AR','B211AR','B503AR','SAAR'],['err1','err2','err3','err4','err6','err7']]
        for k in range(6):
            for i in range(3):
                tracePlot=plt.subplot(gs[k,i])
                trace=mcmc.trace(listTraces[i][k])[extraburn::thin]
                tracePlot.plot(range(trace.size),trace,lw=1)
                tracePlot.set_title(listTraces[i][k])
        fig.savefig(('6SectionsData_tset_traceFig_ACSET.pdf'), format='pdf', dpi=300)  
        
#%%
variables=['velocity1','velocity2','velocity3','velocity6','velocity7','Age','RR','err1']
for i in variables:
    burn=0
    thin=1
    trace=mcmc.trace(i)[burn::thin]
    test=pm.geweke(trace, intervals=20)
    pm.Matplot.geweke_plot(test,'test',lw=1)
    
#%%
class HRAM(pm.Gibbs):
    def __init__(self, stochastic, proposal_sd=None, verbose=None):
        pm.Metropolis.__init__(self, stochastic, proposal_sd=proposal_sd,
                            verbose=verbose, tally=False)
        self.proposal_tau = self.proposal_sd**-2.
        self.n = 0
        self.N = 11
        self.value = pm.rnormal(self.stochastic.value, self.proposal_tau, size=tuple([self.N] + list(self.stochastic.value.shape)))
 
    def step(self):
        x0 = self.value[self.n]
        u = pm.rnormal(np.zeros(self.N), 1.)
        dx = np.dot(u, self.value)
 
        self.stochastic.value = x0
        logp = [self.logp_plus_loglike]
        x_prime = [x0]
 
        for direction in [-1, 1]:
            for i in xrange(25):
                delta = direction*np.exp(.1*i)*dx
                try:
                    self.stochastic.value = x0 + delta
                    logp.append(self.logp_plus_loglike)
                    x_prime.append(x0 + delta)
                except pm.ZeroProbability:
                    self.stochastic.value = x0
 
        i = pm.rcategorical(np.exp(np.array(logp) - pm.flib.logsum(logp)))
        self.value[self.n] = x_prime[i]
        self.stochastic.value = x_prime[i]
 
        if i == 0:
            self.rejected += 1
        else:
            self.accepted += 1
 
        self.n += 1
        if self.n == self.N:
            self.n = 0    