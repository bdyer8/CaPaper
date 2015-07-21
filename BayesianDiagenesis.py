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


a2r=np.linspace(4,8.0,20) 
t=np.linspace(0,4e6,100) 
h=np.linspace(103,0,104)


#load datasets and clean
ArrowCanyon=pd.read_csv('samples_ArrowCanyon.csv')
SanAndres=pd.read_csv('samples_SanAndres.csv')
Leadville=pd.read_csv('samples_B416.csv')
BattleshipWash=pd.read_csv('samples_BattleshipWash.csv')
meter=np.array(Leadville.SAMP_HEIGHT[Leadville.d13c<0]+9.0)
data=np.array(Leadville.d13c[Leadville.d13c<0])

#arrow canyon, etc
ACd13c=(ArrowCanyon.d13c[ArrowCanyon.SAMP_HEIGHT<103])
BWd13c=(BattleshipWash.d13c[(((BattleshipWash.SAMP_HEIGHT-240)<103) & ((BattleshipWash.SAMP_HEIGHT-240)>0))])
ACd13c=list(ACd13c)+list(BWd13c)
ACmeters=(ArrowCanyon.SAMP_HEIGHT[ArrowCanyon.SAMP_HEIGHT<103])
BWmeters=(BattleshipWash.SAMP_HEIGHT[(((BattleshipWash.SAMP_HEIGHT-240)<103) & ((BattleshipWash.SAMP_HEIGHT-240)>0))]-240.0)
ACmeters=list(ACmeters)+list(BWmeters)




#initialize pymc stochastic variables
err = pm.Uniform("err", 0, 500) #uncertainty on d13c values, flat prior
A_to_R_LV=pm.Uniform('A_to_R_LV',4.0,8.0,size=1)  #Cont uniform for indexes of solution set for A_to_R, or K in the other code (20)
A_to_R_AC=pm.Uniform('A_to_R_AC',4.0,8.0,size=1)  #Cont uniform for indexes of solution set for A_to_R, or K in the other code (20)
Time_of_diagenesis=pm.Uniform('Time_of_diagenesis',0,4e6) #Cont uniform for indexes of solution set for A_to_R, or K in the other code (100)
Time_of_diagenesis_AC=pm.Uniform('Time_of_diagenesis_AC',0,4e6)
modelSolutions=pickle.load(open('MeshSolutions_4ma.pkl','rb'))



    
#set pymc observations from data
metersModelLV=pm.Normal("metersLV", 0, 1, value=meter, observed=True)
metersModelAC=pm.Normal("metersAC", 0, 1, value=ACmeters, observed=True)

#model prediction function from model results (should be able to interpolate in 3d to make this continuous)
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
           
           
           
@pm.deterministic
def predd13cLV(A_to_R_LV=A_to_R_LV, Time_of_diagenesis=Time_of_diagenesis, metersModelLV=metersModelLV):
    idxA=np.abs((a2r-A_to_R_LV)).argmin()
    if a2r[idxA]>A_to_R_LV:
        idxA_2=idxA.copy()
        idxA=idxA-1
    else:
        idxA_2=idxA+1
    idxT=np.abs((t-Time_of_diagenesis)).argmin()
    if t[idxT]>Time_of_diagenesis:
        idxT_2=idxT.copy()
        idxT=idxT-1
    else:
        idxT_2=idxT+1
    highResModel1=interp1d(np.linspace(103,0,104),modelSolutions[idxA,idxT,:])
    highResModel2=interp1d(np.linspace(103,0,104),modelSolutions[idxA_2,idxT,:])
    highResModel3=interp1d(np.linspace(103,0,104),modelSolutions[idxA,idxT_2,:])
    highResModel4=interp1d(np.linspace(103,0,104),modelSolutions[idxA_2,idxT_2,:])    
    data=np.array([highResModel1(metersModelLV),highResModel2(metersModelLV),highResModel3(metersModelLV),highResModel4(metersModelLV)]) 
    points= [(a2r[idxA], t[idxT], 0),
             (a2r[idxA_2], t[idxT], 0),
             (a2r[idxA], t[idxT_2], 0),
             (a2r[idxA_2], t[idxT_2], 0)]
    a,b,c,d=bilinear_interpolation(A_to_R_LV,Time_of_diagenesis,points)
    return data.T.dot([a,b,c,d]).flatten()

@pm.deterministic
def predd13cAC(A_to_R_AC=A_to_R_AC, Time_of_diagenesis_AC=Time_of_diagenesis_AC, metersModelAC=metersModelAC):
    idxA=np.abs((a2r-A_to_R_AC)).argmin()
    if a2r[idxA]>A_to_R_AC:
        idxA_2=idxA.copy()
        idxA=idxA-1
    else:
        idxA_2=idxA+1
    idxT=np.abs((t-Time_of_diagenesis_AC)).argmin()
    if t[idxT]>Time_of_diagenesis_AC:
        idxT_2=idxT.copy()
        idxT=idxT-1
    else:
        idxT_2=idxT+1
    highResModel1=interp1d(np.linspace(103,0,104),modelSolutions[idxA,idxT,:])
    highResModel2=interp1d(np.linspace(103,0,104),modelSolutions[idxA_2,idxT,:])
    highResModel3=interp1d(np.linspace(103,0,104),modelSolutions[idxA,idxT_2,:])
    highResModel4=interp1d(np.linspace(103,0,104),modelSolutions[idxA_2,idxT_2,:])    
    data=np.array([highResModel1(metersModelAC),highResModel2(metersModelAC),highResModel3(metersModelAC),highResModel4(metersModelAC)]) 
    points= [(a2r[idxA], t[idxT], 0),
             (a2r[idxA_2], t[idxT], 0),
             (a2r[idxA], t[idxT_2], 0),
             (a2r[idxA_2], t[idxT_2], 0)]
    a,b,c,d=bilinear_interpolation(A_to_R_AC,Time_of_diagenesis_AC,points)
    return data.T.dot([a,b,c,d]).flatten()


    
    
obsLV = pm.Normal("obsLV", predd13cLV, err, value=data, observed=True)
obsAC = pm.Normal("obsAC", predd13cAC, err, value=ACd13c, observed=True)


    
#set up pymc model with all variables, observations, and deterministics
model = pm.Model([predd13cLV,predd13cAC, err, obsLV, obsAC, metersModelLV, metersModelAC, A_to_R_LV, A_to_R_AC, Time_of_diagenesis,Time_of_diagenesis_AC])

mcmc=pm.MCMC(model,db='pickle', dbname='diagenesisACLV.pickle')

#%%
mcmc.sample(200000, 50000, 2) #N, burn, thin


#%%
    
with plt.style.context('ggplot'):
    rcParams.update({'figure.autolayout': True})
    fig=plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(4, 2) 
    timeofdAC = plt.subplot(gs[3,0])
    timeofdLV = plt.subplot(gs[3,1])
    atorAC = plt.subplot(gs[2,0])
    atorLV = plt.subplot(gs[2,1])
    lv = plt.subplot(gs[:2,1])
    ac = plt.subplot(gs[:2,0])
    
    hist, bins = np.histogram(mcmc.trace('Time_of_diagenesis_AC')[:]/1e6,100,normed=True, density=True)
    unity_density = hist / hist.sum()
    width = 1.0 * (bins[1] - bins[0])    
    center = (bins[:-1] + bins[1:]) / 2
    timeofdAC.bar(center, unity_density, align='center', width=width,edgecolor='none',alpha=.7)
    timeofdAC.set_xlabel('Duration of diagenesis (Ma)')
    timeofdAC.set_ylabel('Probability')
    
    hist, bins = np.histogram(mcmc.trace('Time_of_diagenesis')[:]/1e6,100,normed=True, density=True)
    unity_density = hist / hist.sum()
    width = 1.0 * (bins[1] - bins[0])    
    center = (bins[:-1] + bins[1:]) / 2
    timeofdLV.bar(center, unity_density, align='center', width=width,edgecolor='none',alpha=.7)
    timeofdLV.set_xlabel('Duration of diagenesis (Ma)')
    timeofdLV.set_ylabel('Probability')
    
    hist, bins = np.histogram(mcmc.trace('A_to_R_AC')[:],100,normed=True, density=True)
    unity_density = hist / hist.sum()
    width = 1.0 * (bins[1] - bins[0])    
    center = (bins[:-1] + bins[1:]) / 2
    atorAC.bar(center, unity_density, align='center', width=width, edgecolor='none',alpha=.7)
    atorAC.set_xlabel('Ratio of Advection to Reaction Rate 10^N')
    atorAC.set_ylabel('Probability')
    
    hist, bins = np.histogram(mcmc.trace('A_to_R_LV')[:],100,normed=True, density=True)
    unity_density = hist / hist.sum()
    width = 1.0 * (bins[1] - bins[0])    
    center = (bins[:-1] + bins[1:]) / 2
    atorLV.bar(center, unity_density, align='center', width=width, edgecolor='none',alpha=.7)
    atorLV.set_xlabel('Ratio of Advection to Reaction Rate 10^N')
    atorLV.set_ylabel('Probability')
    
    ac.plot(ACd13c,ACmeters,linestyle="None",marker='.',markersize=6,alpha=.8)    
    ac.plot(sorted(predd13cAC.stats()['quantiles'][50],reverse=True),np.sort(metersModelAC.value),color=[.2,.3,.8],lw=2)    
    ac.plot(sorted(predd13cAC.stats()['quantiles'][2.5],reverse=True),np.sort(metersModelAC.value),color=[.2,.3,.8],lw=2,alpha=.3)    
    ac.plot(sorted(predd13cAC.stats()['quantiles'][97.5],reverse=True),np.sort(metersModelAC.value),color=[.2,.3,.8],lw=2,alpha=.3)    
    ac.set_xlabel('$\delta^{13}$C')
    ac.set_ylabel('Meters')
    ac.set_title('Arrow Canyon')
    
    lv.plot(Leadville.d13c[Leadville.d13c<0],Leadville.SAMP_HEIGHT[Leadville.d13c<0]+9,linestyle="None",marker='.',markersize=6,alpha=.8)    
    lv.plot(sorted(predd13cLV.stats()['quantiles'][50],reverse=True),np.sort(metersModelLV.value),color=[.2,.3,.8],lw=2)    
    lv.plot(sorted(predd13cLV.stats()['quantiles'][2.5],reverse=True),np.sort(metersModelLV.value),color=[.2,.3,.8],lw=2,alpha=.3)    
    lv.plot(sorted(predd13cLV.stats()['quantiles'][97.5],reverse=True),np.sort(metersModelLV.value),color=[.2,.3,.8],lw=2,alpha=.3)    
    lv.set_xlabel('$\delta^{13}$C')
    lv.set_ylabel('Meters')
    lv.set_title('Leadville')

    
fig.savefig('combined_LVAC_independentT.pdf', format='pdf', dpi=300)  


#%%


mcmc.db.close()

#db = pymc.database.pickle.load('Disaster.pickle')
