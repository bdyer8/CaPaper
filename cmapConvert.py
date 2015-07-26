# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 17:13:34 2015

@author: bdyer
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

#-----------------------------------------
def array2cmap(X):
    N = X.shape[0]

    r = np.linspace(0., 1., N+1)
    r = np.sort(np.concatenate((r, r)))[1:-1]

    rd = np.concatenate([[X[i, 0], X[i, 0]] for i in xrange(N)])
    gr = np.concatenate([[X[i, 1], X[i, 1]] for i in xrange(N)])
    bl = np.concatenate([[X[i, 2], X[i, 2]] for i in xrange(N)])

    rd = tuple([(r[i], rd[i], rd[i]) for i in xrange(2 * N)])
    gr = tuple([(r[i], gr[i], gr[i]) for i in xrange(2 * N)])
    bl = tuple([(r[i], bl[i], bl[i]) for i in xrange(2 * N)])


    cdict = {'red': rd, 'green': gr, 'blue': bl}
    return colors.LinearSegmentedColormap('my_colormap', cdict, N)
    
mycmap=array2cmap(cmapArray)
values = np.random.rand(10, 10)
plt.gca().pcolormesh(values, cmap=mycmap)

cb = plt.cm.ScalarMappable(norm=None, cmap=mycmap)
cb.set_array(values)
cb.set_clim((0., 1.))

import pickle
fp = open('cmap.pkl', 'wb')
pickle.dump(mycmap, fp)
fp.close()