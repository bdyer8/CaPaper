# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 13:32:44 2015

@author: bdyer
"""
from distutils.core import setup
from Cython.Build import cythonize
import numpy
from copy import copy
#python setup.py build_ext --inplace
setup(
    ext_modules = cythonize("DiagenesisMesh.pyx"),include_dirs=[numpy.get_include()]
)