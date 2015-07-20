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

meter=ACmeters
data=ACd13c


#initialize pymc stochastic variables
err = pm.Uniform("err", 0, 500) #uncertainty on d13c values, flat prior
A_to_R=pm.Uniform('A_to_R',4.0,8.0,size=1)  #Discrete uniform for indexes of solution set for A_to_R, or K in the other code (20)
Time_of_diagenesis=pm.Uniform('Time_of_diagenesis',0,4e6) #Discrete uniform for indexes of solution set for A_to_R, or K in the other code (100)

modelSolutions=pickle.load(open('MeshSolutions_4ma.pkl','rb'))



    
#set pymc observations from data
metersModel=pm.Normal("meters", 0, 1, value=meter, observed=True)

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
def predd13c(A_to_R=A_to_R, Time_of_diagenesis=Time_of_diagenesis, metersModel=metersModel):
    idxA=np.abs((a2r-A_to_R)).argmin()
    if a2r[idxA]>A_to_R:
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
    data=np.array([highResModel1(metersModel),highResModel2(metersModel),highResModel3(metersModel),highResModel4(metersModel)]) 
    points= [(a2r[idxA], t[idxT], 0),
             (a2r[idxA_2], t[idxT], 0),
             (a2r[idxA], t[idxT_2], 0),
             (a2r[idxA_2], t[idxT_2], 0)]
    a,b,c,d=bilinear_interpolation(A_to_R,Time_of_diagenesis,points)
    return data.T.dot([a,b,c,d]).flatten()


    
    
obs = pm.Normal("obs", predd13c, err, value=data, observed=True)


    
#set up pymc model with all variables, observations, and deterministics
model = pm.Model([predd13c, err, obs, metersModel, A_to_R, Time_of_diagenesis])

mcmc=pm.MCMC(model)

#%%
mcmc.sample(100000, 25000)


#%%
def predd13cForPlot(A_to_R, Time_of_diagenesis, metersModel):
    idxA=np.abs((a2r-A_to_R)).argmin()
    if a2r[idxA]>A_to_R:
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
    data=np.array([highResModel1(metersModel),highResModel2(metersModel),highResModel3(metersModel),highResModel4(metersModel)]) 
    points= [(a2r[idxA], t[idxT], 0),
             (a2r[idxA_2], t[idxT], 0),
             (a2r[idxA], t[idxT_2], 0),
             (a2r[idxA_2], t[idxT_2], 0)]
    a,b,c,d=bilinear_interpolation(A_to_R,Time_of_diagenesis,points)
    return data.T.dot([a,b,c,d]).flatten()
    
with plt.style.context('ggplot'):
    rcParams.update({'figure.autolayout': True})
    fig=plt.figure(figsize=(16, 5))
    gs = gridspec.GridSpec(1, 3) 
    timeofd = plt.subplot(gs[0,0])
    ator = plt.subplot(gs[0,1])
    lv = plt.subplot(gs[0,2])
    hist, bins = np.histogram(mcmc.trace('Time_of_diagenesis')[:]/1e6,100,normed=True, density=True)
    unity_density = hist / hist.sum()
    width = 1.0 * (bins[1] - bins[0])    
    center = (bins[:-1] + bins[1:]) / 2
    timeofd.bar(center, unity_density, align='center', width=width,edgecolor='none',alpha=.7)
    timeofd.set_xlabel('Duration of diagenesis (Ma)')
    timeofd.set_ylabel('Probability')
    
    hist, bins = np.histogram(mcmc.trace('A_to_R')[:],100,normed=True, density=True)
    unity_density = hist / hist.sum()
    width = 1.0 * (bins[1] - bins[0])    
    center = (bins[:-1] + bins[1:]) / 2
    ator.bar(center, unity_density, align='center', width=width, edgecolor='none',alpha=.7)
    ator.set_xlabel('Ratio of Advection to Reaction Rate 10^N')
    ator.set_ylabel('Probability')
    
    #lv.plot(Leadville.d13c[Leadville.d13c<0],Leadville.SAMP_HEIGHT[Leadville.d13c<0]+9,linestyle="None",marker='.',markersize=10)
    lv.plot(data,meter,linestyle="None",marker='.',markersize=6,alpha=.8)    
    lv.plot(sorted(predd13c.stats()['quantiles'][50],reverse=True),np.sort(metersModel.value),color=[.2,.3,.8],lw=2)    
    lv.plot(sorted(predd13c.stats()['quantiles'][2.5],reverse=True),np.sort(metersModel.value),color=[.2,.3,.8],lw=2,alpha=.3)    
    lv.plot(sorted(predd13c.stats()['quantiles'][97.5],reverse=True),np.sort(metersModel.value),color=[.2,.3,.8],lw=2,alpha=.3)    

    lv.set_xlabel('$\delta^{13}$C')
    lv.set_ylabel('Meters')
    
fig.savefig('AC_cont.pdf', format='pdf', dpi=300)  


#%%


import numpy
import pdb

a2r=linspace(4,8.0,20) 
t=linspace(0,4e6,100) 
h=linspace(103,0,104)


y = modelSolutions
A,B,C=np.meshgrid(a2r,t,h)
coord = np.hstack([A.ravel().reshape(A.size,1), B.ravel().reshape(B.size,1), C.ravel().reshape(C.size,1)])

f = scipy.interpolate.LinearNDInterpolator(coord,y.ravel())

