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
    
    
interp1d=scipy.interpolate.interp1d
def bilinear_interpolation(x, y, points):
    
    points = sorted(points)               # order points by x, then by y
    (x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = points

    if x1 != _x1 or x2 != _x2 or y1 != _y1 or y2 != _y2:
        raise ValueError('points do not form a rectangle')
    if not x1 <= x <= x2 or not y1 <= y <= y2:
        raise ValueError('(x, y) not within the rectangle')

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
LVmeters=np.array(Leadville.SAMP_HEIGHT[Leadville.d13c<0]+9.0)
LVd13c=np.array(Leadville.d13c[Leadville.d13c<0])

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

modelSolutions=pickle.load(open('MeshSolutions_4ma.pkl','rb'))
#generate perfect synthetic data
AgeSynth=np.random.uniform(1e6,3e6)
RRSynth=np.random.uniform(1e-7,1e-5)
velocitiesSynth=np.random.uniform(.001,.01,5)
testmeters=sorted(np.floor(np.random.uniform(0,103,25)))
sAR=np.log10(velocitiesSynth[0]/RRSynth)
sInject=(AgeSynth*velocitiesSynth[0])/.9
testd13c=synthData(sAR,sInject,np.array(testmeters).astype(int))


meters=testmeters
data=testd13c
sectionName='Synthetic1'

#initialize pymc stochastic variables
#velocity = pm.Uniform('velocity',1e-4,1.0)
err = pm.Uniform("err", 0, 500) #uncertainty on d13c values, flat prior
A_to_R=pm.Uniform('A_to_R',2.845,6.845)  #Cont uniform for indexes of solution set for A_to_R, or K in the other code (20)
#model_iterations=pm.Uniform('model_iterations',0,311111) #Cont uniform for indexes of solution set for A_to_R, or K in the other code (100)
Age=pm.Uniform('Age',.1e6,6e6)
RR=pm.Uniform('RR',1e-9,1e-5)   
#set pymc observations from data
metersModel=pm.Normal("meters", 0, 1, value=meters, observed=True)

#velocity=pm.Uniform('velocity',.00001,(310100*.9/Age))

@pm.deterministic
def velocity(A_to_R=A_to_R, RR=RR):              
    return RR*(10**A_to_R)
    
@pm.deterministic
def model_iterations(velocity=velocity, Age=Age):              
    return (Age*velocity)/.9  #.07 comes from the scheme i solved with for the model solutions
#
#           
#@pm.deterministic
#def velocity(Age=Age, Time_of_diagenesis=Time_of_diagenesis):              
#    return Time_of_diagenesis*(.9/.07)*(.9/Age)  #.07 comes from the scheme i solved with for the model solutions
#


#model prediction function from model results (should be able to interpolate in 3d to make this continuous)    
@pm.deterministic
def predd13c(A_to_R=A_to_R, model_iterations=model_iterations, metersModel=metersModel):
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
    if idxT_2>=100:
        idxT_2=99
        idxT=98
        model_iterations=t[-1]
        
    highResModel1=interp1d(np.linspace(103,0,104),modelSolutions[idxA,idxT,:])
    highResModel2=interp1d(np.linspace(103,0,104),modelSolutions[idxA_2,idxT,:])
    highResModel3=interp1d(np.linspace(103,0,104),modelSolutions[idxA,idxT_2,:])
    highResModel4=interp1d(np.linspace(103,0,104),modelSolutions[idxA_2,idxT_2,:])    
    data=np.array([highResModel1(metersModel),highResModel2(metersModel),highResModel3(metersModel),highResModel4(metersModel)]) 
    points= [(a2r[idxA], t[idxT], 0),
             (a2r[idxA_2], t[idxT], 0),
             (a2r[idxA], t[idxT_2], 0),
             (a2r[idxA_2], t[idxT_2], 0)]
    a,b,c,d=bilinear_interpolation(A_to_R,model_iterations,points)
    return data.T.dot([a,b,c,d]).flatten()
    
    
obs = pm.Normal("obs", predd13c, err, value=data, observed=True)
   
#set up pymc model with all variables, observations, and deterministics
model = pm.Model([predd13c, err, obs, metersModel, A_to_R, velocity, RR, model_iterations])

mcmc=pm.MCMC(model,db='pickle', dbname=(sectionName+'diagenesis.pickle'))

#%%
mcmc.sample(100000, 50000) #N, burn, thin


#%%
    
with plt.style.context('ggplot'):
    rcParams.update({'figure.autolayout': True})
    fig=plt.figure(figsize=(4, 12))
    gs = gridspec.GridSpec(5, 1) 
    timeofd = plt.subplot(gs[3,0])
    ator = plt.subplot(gs[2,0])
    RRate = plt.subplot(gs[4,0])
    strat = plt.subplot(gs[:2,0])
    
    hist, bins = np.histogram(mcmc.trace('velocity')[:],100,normed=True, density=True)
    unity_density = hist / hist.sum()
    width = 1.0 * (bins[1] - bins[0])    
    center = (bins[:-1] + bins[1:]) / 2
    timeofd.bar(center, unity_density, align='center', width=width,edgecolor='none',alpha=.7)
    timeofd.set_xlabel('Vertical Velocity (m/yr)')
    timeofd.set_ylabel('Probability')
    
    hist, bins = np.histogram(mcmc.trace('RR')[:]*1e6,100,normed=True, density=True)
    unity_density = hist / hist.sum()
    width = 1.0 * (bins[1] - bins[0])    
    center = (bins[:-1] + bins[1:]) / 2
    ator.bar(center, unity_density, align='center', width=width, edgecolor='none',alpha=.7)
    ator.set_xlabel('% Rock Reacted per Ma')
    ator.set_ylabel('Probability')
    
    hist, bins = np.histogram(mcmc.trace('Age')[:]/1e6,100,normed=True, density=True)
    unity_density = hist / hist.sum()
    width = 1.0 * (bins[1] - bins[0])    
    center = (bins[:-1] + bins[1:]) / 2
    RRate.bar(center, unity_density, align='center', width=width, edgecolor='none',alpha=.7)
    RRate.set_xlabel('Duration of Diagenesis (Ma)')
    RRate.set_ylabel('Probability')
    
    
    strat.plot(data,meters,linestyle="None",marker='.',markersize=6,alpha=.8)    
    strat.plot(sorted(predd13c.stats()['quantiles'][50],reverse=True),np.sort(metersModel.value),color=[.2,.3,.8],lw=2)    
    strat.plot(sorted(predd13c.stats()['quantiles'][2.5],reverse=True),np.sort(metersModel.value),color=[.2,.3,.8],lw=2,alpha=.3)    
    strat.plot(sorted(predd13c.stats()['quantiles'][97.5],reverse=True),np.sort(metersModel.value),color=[.2,.3,.8],lw=2,alpha=.3)    
    strat.set_xlabel('$\delta^{13}$C')
    strat.set_ylabel('Meters')
    strat.set_title(sectionName)
    
    

    
fig.savefig((sectionName+'Vel_MCMC100k.pdf'), format='pdf', dpi=300)  





mcmc.db.close()

#db = pymc.database.pickle.load('Disaster.pickle')