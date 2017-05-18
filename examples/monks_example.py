import numpy as np
import urllib
from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter

from fanova import fANOVA
import fanova.visualizer 
import os
path = os.path.dirname(os.path.realpath(__file__))

# directory in which you can find all plots
plot_dir = path + '/example_data/test_plots' 
  
'''
example using categoricals
'''
# url with dataset
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-1.train"
# download the file
raw_data = urllib.urlopen(url)
# load the CSV file as a numpy matrix
dataset = np.loadtxt(raw_data, dtype= 'string', delimiter=" ")

X = np.array(dataset[:,2:8], dtype = np.float)
y = np.array(dataset[:,1], dtype = np.float)


cs = ConfigurationSpace()
cs.add_hyperparameter(CategoricalHyperparameter('a1', ['1','2','3'])) 
cs.add_hyperparameter(CategoricalHyperparameter('a2', ['1','2','3']))
cs.add_hyperparameter(CategoricalHyperparameter('a3', ['1','2']))
cs.add_hyperparameter(CategoricalHyperparameter('a4', ['1','2','3']))
cs.add_hyperparameter(CategoricalHyperparameter('a5', ['1','2','3', '4']))
cs.add_hyperparameter(CategoricalHyperparameter('a6', ['1','2']))
   
# create an instance of fanova with trained forest and ConfigSpace
f = fanova.fANOVA(X = X, Y= y, cs=cs)
# marginal of particular parameter:
dims = list([0])
res = f.quantify_importance(dims)
print(res)
#plots
vis = visualizer.Visualizer(f, cs)
vis.create_all_plots(plot_dir)
