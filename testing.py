# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 15:33:21 2015

@author: bdyer
"""

#%%

meshY=3
meshX=10
v=np.ones([meshY,meshX])*0.0
u=np.ones([meshY,meshX])*(1.75703864479316496e-05*5.0)
aveSpeed=np.sqrt(v.mean()**2+u.mean()**2)
injectionSites=np.zeros([meshY,meshX])
injectionSites=injectionSites>0
injectionSites[:,0]=True  


u=np.ones([meshY,meshX])*0.0025535400741844192
N=2
results=np.zeros((N,10))
for k in range(1):
    for i in range(1,N,1):
        carbonMass=.2
        mesh=DiagenesisMesh.meshRock(10,3,u*1,v,2,2,-1.04,3.5703864479316496e-06,injectionSites,[-8.0,-10.0,-1.0,0.0],[1.0,1.0,1.0],[[10*240.0,10*carbonMass],[10*960.0,10*889.0],[10*800.0,10*0.4117647058823529*(carbonMass)/12.0*40.0]]) #reaction rate from same sim    
        if ((mesh.massRatio[0][1]*.1)<=(mesh.r*mesh.massRatio[0][0]*mesh.dt)):
            mesh.dt=.9*(mesh.massRatio[0][1]*.1)/(mesh.r*mesh.massRatio[0][0])
        mesh.hardReset()    
        mesh.inject(int(2.5e5/mesh.dt))
        results[i-1,:]=mesh.printBox('rock','d13c')[1,:]
        print(a_r(carbonMass))
        plt.plot(results[i-1,1:],range(9,0,-1),lw=3,color='r')
        plt.scatter(ArrowCanyon.d13c[ArrowCanyon.SAMP_HEIGHT<105],(ArrowCanyon.SAMP_HEIGHT[ArrowCanyon.SAMP_HEIGHT<105]-5)/10.0)
