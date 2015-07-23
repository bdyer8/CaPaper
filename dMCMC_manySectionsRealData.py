

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
            
            
a2r=np.linspace(2.845,6.845,20) 
t=np.linspace(0,311111,100) 
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
modelSolutions=pickle.load(open('MeshSolutions_4ma.pkl','rb'))



def predFunct(A_to_R, model_iterations, metersModel):
 
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
    if model_iterations>311111.0:
        data=data/data*2000.0
    a,b,c,d=bilinear_interpolation(A_to_R,injections,points)
    results=data.T.dot([a,b,c,d]).flatten()
    return results.T

#%%

#initialize pymc stochastic variables

N=4
names=['AC','LV1','LV2','B503','B402','B211','SA']
err = pm.Uniform("err", 0, 500) #uncertainty on d13c values, flat prior
Age=pm.Uniform('Age',.01e6,4e6)
RR=pm.Uniform('RR',1e-8,1e-5) 

AR1=pm.Uniform('ACAR',2.845,6.845)
AR2=pm.Uniform('LV1AR',2.845,6.845)
AR3=pm.Uniform('LV2AR',2.845,6.845)
AR4=pm.Uniform('B503AR',2.845,6.845)
AR5=pm.Uniform('B402AR',2.845,6.845)
AR6=pm.Uniform('B211AR',2.845,6.845)
AR7=pm.Uniform('SAAR',2.845,6.845)


#set pymc observations from data
heights1=pm.Normal("ACmeters", 0, 10, value=ACmeters, observed=True)
heights2=pm.Normal("LV1meters", 0, 10, value=LV1meters, observed=True)
heights3=pm.Normal("LV2meters", 0, 10, value=LV2meters, observed=True)
heights4=pm.Normal("B503meters", 0, 10, value=B503meters, observed=True)
heights5=pm.Normal("B402meters", 0, 10, value=B402meters, observed=True)
heights6=pm.Normal("B211meters", 0, 10, value=B211meters, observed=True)
heights7=pm.Normal("SAmeters", 0, 10, value=SAmeters, observed=True)

@pm.deterministic
def velocity1(AR=AR1, RR=RR):              
    return RR*(10**AR)
@pm.deterministic
def velocity2(AR=AR2, RR=RR):              
    return RR*(10**AR)
@pm.deterministic
def velocity3(AR=AR3, RR=RR):              
    return RR*(10**AR)
@pm.deterministic
def velocity4(AR=AR4, RR=RR):              
    return RR*(10**AR)
@pm.deterministic
def velocity5(AR=AR5, RR=RR):              
    return RR*(10**AR)
@pm.deterministic
def velocity6(AR=AR6, RR=RR):              
    return RR*(10**AR)
@pm.deterministic
def velocity7(AR=AR7, RR=RR):              
    return RR*(10**AR)

@pm.deterministic
def model_iterations1(V=velocity1, Age=Age):              
    return (Age*V)/.9  #.07 comes from the scheme i solved with for the model solutions
@pm.deterministic
def model_iterations2(V=velocity2, Age=Age):              
    return (Age*V)/.9  #.07 comes from the scheme i solved with for the model solutions
@pm.deterministic
def model_iterations3(V=velocity3, Age=Age):              
    return (Age*V)/.9  #.07 comes from the scheme i solved with for the model solutions
@pm.deterministic
def model_iterations4(V=velocity4, Age=Age):              
    return (Age*V)/.9  #.07 comes from the scheme i solved with for the model solutions
@pm.deterministic
def model_iterations5(V=velocity5, Age=Age):              
    return (Age*V)/.9  #.07 comes from the scheme i solved with for the model solutions
@pm.deterministic
def model_iterations6(V=velocity6, Age=Age):              
    return (Age*V)/.9  #.07 comes from the scheme i solved with for the model solutions
@pm.deterministic
def model_iterations7(V=velocity7, Age=Age):              
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
    
obs1 = pm.Normal("ACd13c", predACd13c, err, value=ACd13c, observed=True)
obs2 = pm.Normal("LV1d13c", predLV1d13c, err, value=LV1d13c, observed=True)
obs3 = pm.Normal("LV2d13c", predLV2d13c, err, value=LV2d13c, observed=True)
obs4 = pm.Normal("B503d13c", predB503d13c, err, value=B503d13c, observed=True)
obs5 = pm.Normal("B402d13c", predB402d13c, err, value=B402d13c, observed=True)
obs6 = pm.Normal("B211d13c", predB211d13c, err, value=B211d13c, observed=True)
obs7 = pm.Normal("SAd13c", predSAd13c, err, value=SAd13c, observed=True)

   
#set up pymc model with all variables, observations, and deterministics
model = pm.Model([Age,RR,err,
                  predACd13c,predLV1d13c,predLV2d13c,predB503d13c,predB402d13c,predB211d13c,predSAd13c,
                  obs1,obs2,obs3,obs4,obs5,obs6,obs7,
                  heights1,heights2,heights3,heights4,heights5,heights6,heights7, 
                  AR1,AR2,AR3,AR4,AR5,AR6,AR7,
                  velocity1,velocity2,velocity3,velocity4,velocity5,velocity6,velocity7,
                  model_iterations1,model_iterations2,model_iterations3,model_iterations4,model_iterations5,model_iterations6,model_iterations7])




map_start=pm.MAP(model)
map_start.fit()
mcmc=pm.MCMC(model,db='pickle', dbname=('AllSectionsDiagenesis.pickle'))

#%%
mcmc.sample(100000, 50000,3) #N, burn, thin


#%%
N=7
with plt.style.context('fivethirtyeight'):
    
    rcParams.update({'figure.autolayout': True})
    fig=plt.figure(figsize=(5*N, 9))
    gs = gridspec.GridSpec(3, 1*N+1) 
    allData=[[ACd13c],[LV1d13c],[LV2d13c],[B503d13c],[B402d13c],[B211d13c],[SAd13c]]
    allHeights=[[ACmeters],[LV1meters],[LV2meters],[B503meters],[B402meters],[B211meters],[SAmeters]]
    fits_50=[[np.array(sorted(predACd13c.stats()['quantiles'][50][:],reverse=True))],
              [np.array(sorted(predLV1d13c.stats()['quantiles'][50][:],reverse=True))],
               [np.array(sorted(predLV2d13c.stats()['quantiles'][50][:],reverse=True))],
                [np.array(sorted(predB503d13c.stats()['quantiles'][50][:],reverse=True))],
                 [np.array(sorted(predB402d13c.stats()['quantiles'][50][:],reverse=True))],
                  [np.array(sorted(predB211d13c.stats()['quantiles'][50][:],reverse=True))],
                   [np.array(sorted(predSAd13c.stats()['quantiles'][50][:],reverse=True))]]
                   
                
    fits_2_5=[[np.array(sorted(predACd13c.stats()['quantiles'][2.5][:],reverse=True))],
              [np.array(sorted(predLV1d13c.stats()['quantiles'][2.5][:],reverse=True))],
               [np.array(sorted(predLV2d13c.stats()['quantiles'][2.5][:],reverse=True))],
                [np.array(sorted(predB503d13c.stats()['quantiles'][2.5][:],reverse=True))],
                 [np.array(sorted(predB402d13c.stats()['quantiles'][2.5][:],reverse=True))],
                  [np.array(sorted(predB211d13c.stats()['quantiles'][2.5][:],reverse=True))],
                   [np.array(sorted(predSAd13c.stats()['quantiles'][2.5][:],reverse=True))]]
     
    fits_97_5=[[np.array(sorted(predACd13c.stats()['quantiles'][97.5][:],reverse=True))],
              [np.array(sorted(predLV1d13c.stats()['quantiles'][97.5][:],reverse=True))],
               [np.array(sorted(predLV2d13c.stats()['quantiles'][97.5][:],reverse=True))],
                [np.array(sorted(predB503d13c.stats()['quantiles'][97.5][:],reverse=True))],
                 [np.array(sorted(predB402d13c.stats()['quantiles'][97.5][:],reverse=True))],
                  [np.array(sorted(predB211d13c.stats()['quantiles'][97.5][:],reverse=True))],
                   [np.array(sorted(predSAd13c.stats()['quantiles'][97.5][:],reverse=True))]]
                  
                   
    heights_50=[[np.sort(heights1.value)],
                 [np.sort(heights2.value)],
                 [np.sort(heights3.value)],
                 [np.sort(heights4.value)],
                 [np.sort(heights5.value)],
                 [np.sort(heights6.value)],
                 [np.sort(heights7.value)]]
    
    heights_50=heights_50[:N]
    
    for i in range(N):

        timeofd = plt.subplot(gs[2,i])
        strat = plt.subplot(gs[:2,i])
        
        density = scipy.stats.gaussian_kde(mcmc.trace('velocity'+str(i+1))[:]*1000.0)
        xs = np.linspace(0,mcmc.trace('velocity'+str(i+1))[:].max()*1000.0,200)
        density.covariance_factor = lambda : .3
        density._compute_covariance()
        timeofd.plot(xs,density(xs)/np.trapz(density(xs)),color=plt.rcParams['axes.color_cycle'][3])
        timeofd.set_xlabel('Vertical Velocity (mm/yr)')
        timeofd.set_ylabel('Probability')
        #timeofd.set_xlim([0,.5])
        
        
          
        strat.plot(np.array(fits_50[i][0]),np.array(heights_50[i][0]),color=plt.rcParams['axes.color_cycle'][2],lw=2)
        strat.plot(np.array(fits_2_5[i][0]),np.array(heights_50[i][0]),color=plt.rcParams['axes.color_cycle'][2],lw=2,alpha=.3) 
        strat.plot(np.array(fits_97_5[i][0]),np.array(heights_50[i][0]),color=plt.rcParams['axes.color_cycle'][2],lw=2,alpha=.3) 
        strat.plot(np.array(allData[i][0]),np.array(allHeights[i][0]),linestyle="None",marker='o',markeredgecolor='none',alpha=.8)          
        strat.set_xlabel('$\delta^{13}$C')
        strat.set_ylabel('Meters')
        strat.set_title(names[i])

ator = plt.subplot(gs[1,-1])
RRate = plt.subplot(gs[2,-1]) 
errPlot = plt.subplot(gs[0,-1]) 

density = scipy.stats.gaussian_kde(np.sqrt(1/mcmc.trace('err')[:]))
xs = np.linspace(0,mcmc.trace('err')[:].max(),200)
density.covariance_factor = lambda : .3
density._compute_covariance()
errPlot.plot(xs,density(xs)/np.trapz(density(xs)),color=plt.rcParams['axes.color_cycle'][3],lw=3)
errPlot.plot([0.2,0.2],[0,(density(xs)/np.trapz(density(xs))).max()*1.1],color=plt.rcParams['axes.color_cycle'][1],lw=2)
errPlot.set_xlabel('$\sigma$ error')
errPlot.set_ylabel('Probability')
#errPlot.set_xlim([0,1])


density = scipy.stats.gaussian_kde(mcmc.trace('RR')[:]*1e6)
xs = np.linspace0,mcmc.trace('RR')[:].max()*1e6,200)
density.covariance_factor = lambda : .3
density._compute_covariance()
ator.plot(xs,density(xs)/np.trapz(density(xs)),color=plt.rcParams['axes.color_cycle'][3],lw=3)
ator.set_xlabel('% Rock Reacted per Ma')
ator.set_ylabel('Probability')
ator.set_xlim([1e-2,2e1])


density = scipy.stats.gaussian_kde(mcmc.trace('Age')[:]/1e6)
xs = np.linspace(0,mcmc.trace('Age')[:].max()/1e6,200)
density.covariance_factor = lambda : .3
density._compute_covariance()
RRate.plot(xs,density(xs)/np.trapz(density(xs)),color=plt.rcParams['axes.color_cycle'][3],lw=3)
RRate.set_xlabel('Duration of Diagenesis (Ma)')
RRate.set_ylabel('Probability')
RRate.set_xlim([0,4.0])   
#    
fig.savefig(('realData_7_1e6.pdf'), format='pdf', dpi=300)  
#
#
#
#
#
#mcmc.db.close()

#db = pymc.database.pickle.load('Disaster.pickle')