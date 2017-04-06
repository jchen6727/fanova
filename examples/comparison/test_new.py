# -*- coding: utf-8 -*-
"""
Test for the newest fANOVA version
"""
import numpy as np 
import sys

import fanova

from collections import OrderedDict

import ConfigSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter

import os
path = os.path.dirname(os.path.realpath(__file__))

my_dict = OrderedDict()

'''
Online LDA example
'''
def online_lda():
    marginals =[]
    the_keys = []
    X = np.loadtxt('../example_data/online_lda/online_lda_features.csv', delimiter=",")
    Y = np.loadtxt('../example_data/online_lda/online_lda_responses.csv', delimiter=",")
    f = fanova.fANOVA(X,Y, n_trees=32,bootstrapping=True)

    res = f.quantify_importance((0, 1, 2))

    for key in res.keys():
        if key != (0, 1, 2):
            marginals.append(res[key]['individual importance'])
            the_keys.append(key)
    return the_keys, marginals

'''
Diabetes example
'''
def csv_example():
    marginals =[]
    the_keys = []
    data = np.loadtxt("../example_data/csv-example/test_data.csv", delimiter=",")
    X = np.array(data[:, :2], dtype = np.float)
    Y = np.array(data[:,-1:], dtype = np.float).flatten()
    # config space
    pcs = list(zip(np.min(X,axis=0), np.max(X, axis=0)))
    cs = ConfigSpace.ConfigurationSpace()
    for i in range(len(pcs)):
        cs.add_hyperparameter(UniformFloatHyperparameter("%i" %i, pcs[i][0], pcs[i][1]))

    f2 = fanova.fANOVA(X, Y, cs)
    res = f2.quantify_importance((0, 1))
    
    for key in res.keys():
        marginals.append(res[key]['individual importance'])
        the_keys.append(key)
    
    return the_keys, marginals

if __name__ == "__main__":
    example = sys.argv[1]
    if example == 'online_lda':
        res = online_lda()
    elif example == 'csv_example':
        res = csv_example()
    print(res)
    