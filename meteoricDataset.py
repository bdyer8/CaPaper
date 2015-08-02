# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 12:24:34 2015

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


#load datasets and clean
Liang=pd.read_csv('Liang2015.csv')
Patterson=pd.read_csv('Patterson1994.csv')
