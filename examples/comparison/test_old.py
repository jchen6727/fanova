# -*- coding: utf-8 -*-
"""
Test for the old fANOVA version
"""
import sys
import numpy as np

from pyfanova.fanova import Fanova
from pyfanova.fanova_from_csv import FanovaFromCSV

import os
path = os.path.dirname(os.path.realpath(__file__))

'''
Online LDA example
'''
def online_lda():
    marginals = []
    f = Fanova(path + '/old_online_lda')
    marginals.append(f.get_pairwise_marginal(0,1))
    marginals.append(f.get_pairwise_marginal(1,2))
    marginals.append(f.get_marginal(0))
    marginals.append(f.get_marginal(1))
    marginals.append(f.get_marginal(2))
    marginals.append(f.get_pairwise_marginal(0,2))
    
    return marginals
    
def csv_example():
    marginals = []
    f = FanovaFromCSV("../example_data/csv-example/test_data.csv")
    marginals.append(f.get_pairwise_marginal(0,1))
    marginals.append(f.get_marginal(0))
    marginals.append(f.get_marginal(1))
    
    return marginals

if __name__ == "__main__":
    example = sys.argv[1]
    if example == 'online_lda':
        res = online_lda()
    elif example == 'csv_example':
        res = csv_example()
    print(res)