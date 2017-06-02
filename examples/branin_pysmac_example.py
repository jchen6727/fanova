# -*- coding: utf-8 -*-
"""
branin example for testing pysmac-fanova
"""

import math
import pysmac
import pysmac.utils.pysmac_fanova as pysmac_fanova

import os
path = os.path.dirname(os.path.realpath(__file__))

'''
Branin as example
'''
def modified_branin(x1, x2, x3):
    # enforce that x3 has an integer value
    if (int(x3) != x3):
        raise ValueError("parameter x3 has to be an integer!")
    a = 1
    b = 5.1 / (4*math.pi**2)
    c = 5 / math.pi
    r = 6
    s = 10
    t = 1 / (8*math.pi)
    ret  = a*(x2-b*x1**2+c*x1-r)**2+s*(1-t)*math.cos(x1)+s + x3
    return ret

# parameter definition    
parameter_definition=dict(\
                x1=('real',    [-5, 5],  1),     # this line means x1 is a float between -5 and 5, with a default of 1
                x2=('real',    [-5, 5], -1),     # same as x1, but the default is -1
                x3=('integer', [0, 10],  1),     # integer that ranges between 0 and 10, default is 1
                )
# optimizer object
opt = pysmac.SMAC_optimizer(working_directory = path + '/pysmac_output', persistent_files=True)
# call its minimize method
value, parameters = opt.minimize(modified_branin,      # the function to be minimized
                                 100,                 # 1000 the maximum number of function evaluations
                                 parameter_definition,
                                 num_runs = 3) # the parameter dictionary

# fanova object
fanova = pysmac_fanova.smac_to_fanova(path + '/pysmac_output/out/scenario', path + "/merged_states")
res = fanova.quantify_importance((2, ))
print(res)
best_margs = fanova.get_most_important_pairwise_marginals(n=3)
print('Most important pairwise marginals: %s' %best_margs)
