# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 14:50:35 2015

@author: bdyer
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from sklearn import linear_model, datasets

AC=pd.read_csv('swartClino')
clino=pd.read_csv('swartClino.csv', header=None, names=['age','d13c'])

X=clino.values[:,1:2]
y=clino.values[:,0:1]
plt.scatter(X,y)

model = linear_model.LinearRegression()
model.fit(X, y)

#%%


# Robustly fit linear model with RANSAC algorithm
model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression())
model_ransac.fit(X, y)
inlier_mask = model_ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

# Predict data of estimated models
line_X = np.arange(-5, 3)
line_y = model.predict(line_X[:, np.newaxis])
line_y_ransac = model_ransac.predict(line_X[:, np.newaxis])

# Compare estimated coefficients


plt.plot(X[inlier_mask], y[inlier_mask], '.g', label='Inliers')
plt.plot(X[outlier_mask], y[outlier_mask], '.r', label='Outliers')
plt.plot(line_X, line_y, '-k', label='Linear regressor')
plt.plot(line_X, line_y_ransac, '-b', label='RANSAC regressor')
plt.legend(loc='lower right')
plt.show()