import sys

sys.path.append("/home/sfalkner/repositories/github/random_forest_run/build")
sys.path.append("/home/sfalkner/repositories/github/fanova")
sys.path.append("/home/sfalkner/repositories/github/ConfigSpace")

import os
import pickle
import tempfile
import unittest


import numpy as np
import ConfigSpace as cfs
import fanova


class TestfANOVAtoyData(unittest.TestCase):

	def setUp(self):
		
		self.X = np.loadtxt('toy_data_set_features.csv', delimiter=',')
		self.y = np.loadtxt('toy_data_set_responses.csv', delimiter=',')
		
		self.cfs = cfs.ConfigurationSpace()
		
		
		f1 = cfs.UniformFloatHyperparameter('x1', 0,100 )
		f2 = cfs.CategoricalHyperparameter('x2', [0,1,2])
		
		self.cfs.add_hyperparameter(f1)
		self.cfs.add_hyperparameter(f2)
		
	
	def tearDown(self):
		self.X = None
		self.y = None
		
	def test_with_toy_data(self):
		
		f = fanova.fANOVA(self.X,self.y,self.cfs, bootstrapping=False, n_trees=1, seed=5, max_features=1)

		f.the_forest.save_latex_representation('/tmp/fanova_')
		print("="*80)
		print(f.the_forest.all_split_values())
		print("total variances", f.the_forest.get_trees_total_variances())
		print(f.quantify_importance([0,1]))
		print(f.trees_total_variance)
		
		print(f.V_U)
		

if __name__ == '__main__':
	unittest.main()
