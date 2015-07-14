# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 16:14:34 2015

@author: bdyer
"""
#%%
import pstats, cProfile

#reload(DiagenesisMesh)
#meshX=500/2
#meshY=80
#u=load('uLensSeaWater.npy')[:,:250]
#v=load('vLensSeaWater.npy')[:,:250]
#u=u[:,:]*5.0
#v=v[:,:]*-5.0
#mesh=DiagenesisMesh.meshRock(meshX,meshY,np.array(list(reversed(u))),np.array(list(reversed(v))),2.0,-1,-1,1.0e-03)  #.01 = 1% per timestep~Ma
#

cProfile.runctx("mesh.inject(10)",globals(), locals(), "Profile.prof")

s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()