# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 14:58:28 2015

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
            
            

caMass=np.linspace(1.0,20.0,20)
alpha=np.linspace(.999,1.0,20)

caMass=[   0.46666667,   52.26666667,  104.06666667,  155.86666667,
        207.66666667,  259.46666667,  311.26666667,  363.06666667,
        414.86666667,  466.66666667]
time=np.linspace(.9985,1.1,5)

#load datasets and clean
ArrowCanyon=pd.read_csv('samples_ArrowCanyon.csv')
Leadville=pd.read_csv('samples_B416.csv')
BattleshipWash=pd.read_csv('samples_BattleshipWash.csv')

#leadville 1
LV1meters=np.array(Leadville.SAMP_HEIGHT[Leadville.d13c<0]+9.0)
LV1d13c=np.array(Leadville.d13c[Leadville.d13c<0])

#arrow canyon
ACd13c=(ArrowCanyon.d13c[ArrowCanyon.SAMP_HEIGHT<103])
BWd13c=(BattleshipWash.d13c[(((BattleshipWash.SAMP_HEIGHT-240)<103) & ((BattleshipWash.SAMP_HEIGHT-240)>0))])
ACd13c=list(ACd13c)+list(BWd13c)
ACmeters=(ArrowCanyon.SAMP_HEIGHT[ArrowCanyon.SAMP_HEIGHT<103])
BWmeters=(BattleshipWash.SAMP_HEIGHT[(((BattleshipWash.SAMP_HEIGHT-240)<103) & ((BattleshipWash.SAMP_HEIGHT-240)>0))]-240.0)
ACmetersd13c=list(ACmeters)+list(BWmeters)



ACd44ca=ArrowCanyon.d44ca[(ArrowCanyon.SAMP_HEIGHT<103)].dropna()
ACd13c=ArrowCanyon.d13c[ACd44ca.index]
ACmeters=ArrowCanyon.SAMP_HEIGHT[ACd44ca.index]
BWd44ca=(BattleshipWash.d44ca[(((BattleshipWash.SAMP_HEIGHT-240)<103) & ((BattleshipWash.SAMP_HEIGHT-240)>0))]).dropna()
BWd13c=BattleshipWash.d13c[BWd44ca.index]
BWmeters=(BattleshipWash.SAMP_HEIGHT[BWd44ca.index]-240.0)
ACd44ca=list(ACd44ca)+list(BWd44ca)
ACmetersd13c=list(ACmeters)+list(BWmeters)
ACd13c=list(ACd13c)+list(BWd13c)
#ACmeters=(ArrowCanyon.SAMP_HEIGHT[ArrowCanyon.SAMP_HEIGHT<103])
#BWmeters=(BattleshipWash.SAMP_HEIGHT[(((BattleshipWash.SAMP_HEIGHT-240)<103) & ((BattleshipWash.SAMP_HEIGHT-240)>0))]-240.0)
ACmeters=list(ACmeters)+list(BWmeters)


LV1meters=np.array(Leadville.SAMP_HEIGHT[Leadville.d13c<0]+9.0)
LV1d13c=Leadville.d13c[Leadville.d13c<0]
LV1d44ca=Leadville.d44ca[Leadville.d13c<0]
LV1d44cameters=Leadville.SAMP_HEIGHT[Leadville.d44ca[Leadville.d13c<0].index]+9

carbonSolutions=pickle.load(open('carbonSolutionsAC.pkl','rb'))
calciumSolutions=pickle.load(open('calciumSolutionsAC.pkl','rb'))

carbonSolutions=pickle.load(open('d13cAC.pkl','rb'))
calciumSolutions=pickle.load(open('d44caAC.pkl','rb'))

carbonSolutions=carbonSolutions[8,:,:,:]
calciumSolutions=calciumSolutions[8,:,:,:]

def predFunct(A, M, metersModel,solSet):
    idxA=np.abs((alpha-A)).argmin()
    if alpha[idxA]>A:
        idxA_2=idxA.copy()
        idxA=idxA-1
    else:
        idxA_2=idxA+1
    idxT=np.abs((caMass-M)).argmin()
    if caMass[idxT]>M:
        idxT_2=idxT.copy()
        idxT=idxT-1
    else:
        idxT_2=idxT+1


    
    highResModel1=interp1d(np.linspace(103,0,104),solSet[idxT,idxA,:])
    highResModel2=interp1d(np.linspace(103,0,104),solSet[idxT,idxA_2,:])
    highResModel3=interp1d(np.linspace(103,0,104),solSet[idxT_2,idxA,:])
    highResModel4=interp1d(np.linspace(103,0,104),solSet[idxT_2,idxA_2,:])    
    data=np.array([highResModel1(metersModel),highResModel2(metersModel),highResModel3(metersModel),highResModel4(metersModel)]) 
    points= [(alpha[idxA], caMass[idxT], 0),
             (alpha[idxA_2], caMass[idxT], 0),
             (alpha[idxA], caMass[idxT_2], 0),
             (alpha[idxA_2], caMass[idxT_2], 0)]
#    if model_iterations>500000.0:
#        data=data/data*-7.0
    a,b,c,d=bilinear_interpolation(A,M,points)
    results=data.T.dot([a,b,c,d]).flatten()
    return results.T
    
    

errCa = pm.Uniform("errCa", 0, 500) #uncertainty on d13c values, flat prior
errC = pm.Uniform("errC", 0, 500) #uncertainty on d13c values, flat prior

#alphaDist=pm.Uniform('alpha',.999,1.0)
#caMassDist=pm.Uniform('caMass',1.0,20.0)

alphaDist=pm.Uniform('alpha',.9985,1.1)
caMassDist=pm.Uniform('caMass',1.0,70.0)


#set pymc observations from data
heights1=pm.Normal("ACmeters", 0, 100, value=ACmeters, observed=True)
heights2=pm.Normal("ACmetersC", 0, 100, value=ACmetersd13c, observed=True)
base=pm.Uniform('baseCa',-0.2,0.2)
d13cbase=pm.Uniform('baseC',-1.0,1.0)
@pm.deterministic
def predACd44Ca(base=base,alpha=alphaDist, caMass=caMassDist, heights=heights1,calciumSolutions=calciumSolutions):
    return predFunct(alpha,caMass,heights,calciumSolutions)+base

obs1 = pm.Normal("ACd44ca", predACd44Ca, errCa, value=np.array(ACd44ca), observed=True)
@pm.deterministic
def predACd13c(d13cbase=d13cbase, alpha=alphaDist, caMass=caMassDist, heights=heights1,carbonSolutions=carbonSolutions):
    return predFunct(alpha,caMass,heights,carbonSolutions)+d13cbase
    
obs2 = pm.Normal("ACd13c", predACd13c, errC, value=ACd13c, observed=True)

model = pm.Model([errCa,base,errC,alphaDist,caMassDist,heights1,heights2,predACd13c,predACd44Ca,obs1,obs2])

#map_start=pm.MAP(model)
#map_start.fit()
mcmc=pm.MCMC(model)

#%%

mcmc.sample(50000,25000,1)

#%%

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

    
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(3, 3) 
    AC = plt.subplot(gs[:,0])
    AC.set_ylabel('height(m)')
    AC.set_xlabel('$\delta$44Ca')
    AC.set_ylim([0,120])
    #AC.set_xlim([-10,6])
    offset=mcmc.trace('baseCa').stats()['quantiles'][50]
    AC2 = plt.subplot(gs[:,2])
    AC2.set_xlabel('$\delta$13C')
    AC.plot(ACd44ca,ACmeters,linestyle="None",marker='o',markeredgecolor='none',alpha=.6,markersize=7,color=plt.rcParams['axes.color_cycle'][0])
    ca2_5=[np.array(sorted(np.percentile(mcmc.trace('predACd44Ca')[extraburn::thin,:],2.5,axis=0),reverse=True))]
    ca50=[np.array(sorted(np.percentile(mcmc.trace('predACd44Ca')[extraburn::thin,:],50,axis=0),reverse=True))]
    ca97_5=[np.array(sorted(np.percentile(mcmc.trace('predACd44Ca')[extraburn::thin,:],97.5,axis=0),reverse=True))]
    c2_5=[np.array(sorted(np.percentile(mcmc.trace('predACd13c')[extraburn::thin,:],2.5,axis=0),reverse=True))]
    c50=[np.array(sorted(np.percentile(mcmc.trace('predACd13c')[extraburn::thin,:],50,axis=0),reverse=True))]
    c97_5=[np.array(sorted(np.percentile(mcmc.trace('predACd13c')[extraburn::thin,:],97.5,axis=0),reverse=True))] 
    AC2.plot(ACd13c,ACmetersd13c,linestyle="None",marker='o',markeredgecolor='none',alpha=.6,markersize=7,color=plt.rcParams['axes.color_cycle'][0])
    
    caHeights=[np.sort(heights1.value)]
    cHeights=[np.sort(heights2.value)]
    
    CaInterp2_5=interp1d(np.linspace(0,103,ca2_5[0].size),ca2_5)   
    CaInterp50=interp1d(np.linspace(0,103,ca2_5[0].size),ca50)   
    CaInterp97_5=interp1d(np.linspace(0,103,ca2_5[0].size),ca97_5)   
    
    AC.plot(np.array(ca2_5).ravel(),np.array(caHeights).ravel(),color='#6c71c4',lw=2,alpha=.5)
    AC.plot(np.array(ca50).ravel(),np.array(caHeights).ravel(),color='#6c71c4',lw=2,alpha=1)
    AC.plot(np.array(ca97_5).ravel(),np.array(caHeights).ravel(),color='#6c71c4',lw=2,alpha=.5)
    AC2.plot(np.array(c2_5).ravel(),np.array(cHeights).ravel(),color='#6c71c4',lw=2,alpha=.5)
    AC2.plot(np.array(c50).ravel(),np.array(cHeights).ravel(),color='#6c71c4',lw=2,alpha=1)
    AC2.plot(np.array(c97_5).ravel(),np.array(cHeights).ravel(),color='#6c71c4',lw=2,alpha=.5)
    
    crossPlt = plt.subplot(gs[2,1])
    alphaPlt = plt.subplot(gs[0,1])
    caMassPlt = plt.subplot(gs[1,1])
    crossPlt.set_xlabel('$\delta$13C')
    crossPlt.set_ylabel('$\delta$44Ca')
    alphaPlt.set_xlabel('fractionation factor')
    caMassPlt.set_xlabel('Ca mass flux yr$^{-1}$')
    alphaPlt.hist(np.abs(1.0-mcmc.trace('alpha')[extraburn::thin])*1000,100)
    caMassPlt.hist(mcmc.trace('caMass')[extraburn::thin],100)
    
#    crossPlt.plot(np.array(c2_5).ravel(),np.array(CaInterp2_5(cHeights)).ravel(),color='#6c71c4',lw=2,alpha=.5)
#    crossPlt.plot(np.array(c50).ravel(),np.array(CaInterp50(cHeights)).ravel(),color='#6c71c4',lw=2,alpha=1)
#    crossPlt.plot(np.array(c97_5).ravel(),np.array(CaInterp97_5(cHeights)).ravel(),color='#6c71c4',lw=2,alpha=.5)
    crossPlt.plot(mcmc.trace('predACd13c')[:,:],mcmc.trace('predACd44Ca')[:,:],linestyle='none',color=[.8,.8,.9],alpha=.1,marker='o',markersize=7,markeredgecolor='none')
    crossPlt.plot(ACd13c,ACd44ca,linestyle='none',marker='o',alpha=1,markeredgecolor=[1,1,1])   
#    crossPlt.plot(np.sort(mcmc.trace('predACd13c').stats()['quantiles'][97.5]),np.sort(mcmc.trace('predACd44Ca').stats()['quantiles'][2.5],),color='r',alpha=1,lw=1)
#    crossPlt.plot(np.sort(mcmc.trace('predACd13c').stats()['quantiles'][2.5]),np.sort(mcmc.trace('predACd44Ca').stats()['quantiles'][97.5],),color='r',alpha=1,lw=1)    
#    crossPlt.plot(np.sort(mcmc.trace('predACd13c').stats()['quantiles'][97.5]),np.sort(mcmc.trace('predACd44Ca').stats()['quantiles'][97.5],),color='r',alpha=1,lw=1)    
#    crossPlt.plot(np.sort(mcmc.trace('predACd13c').stats()['quantiles'][2.5]),np.sort(mcmc.trace('predACd44Ca').stats()['quantiles'][2.5],),color='r',alpha=1,lw=1)    
    crossPlt.plot(np.sort(mcmc.trace('predACd13c').stats()['quantiles'][50]),np.sort(mcmc.trace('predACd44Ca').stats()['quantiles'][50]),alpha=1,lw=3)
    
    fig.savefig(('alphaFittingAClowsolset.pdf'), format='pdf', dpi=60)  
        
    


