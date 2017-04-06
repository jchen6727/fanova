# -*- coding: utf-8 -*-
import subprocess
import numpy as np
import ast



def start_comparison(example):
    print('Starting the %s example comparison: \n' %example)
    p = subprocess.Popen("python3 test_new.py %s" %example, stdout=subprocess.PIPE, shell=True)
    new_res = p.communicate()[0]
    new_res = ast.literal_eval(new_res)
    marginals = np.array(new_res[1])*100
    params = np.array(new_res[0])
    
    p2 = subprocess.Popen("python test_old.py %s" %example, stdout=subprocess.PIPE, shell=True)
    old_res = p2.communicate()[0]
    old_res = old_res.split('\n')
    old_marginals = np.asarray(old_res)
    old_vals = old_marginals[-2]
    marg_vals = [float(e) for e in old_vals.strip("[] \n").split(",")]
    for i, param in enumerate(params):
        diff_val = abs(float(marg_vals[i]) - float(marginals[i]))
        print('New fANOVA importance for parameter/s %s is: %f' %(param, marginals[i]))
        print('Old fANOVA importance for parameter/s %s is: %f' %(param, marg_vals[i]))
        print('The difference in parameter/s %s is:  %f ' %(param, diff_val))
        print('')

start_comparison('online_lda')
start_comparison('csv_example')