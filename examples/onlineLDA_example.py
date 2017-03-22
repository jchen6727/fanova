import numpy as np
from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
import csv
import fanova
#import visualizer

import os
path = os.path.dirname(os.path.realpath(__file__))

# get sample data from online lda
data = np.loadtxt(path + '/example_data/online_lda/uniq_configurations-it2.csv', delimiter=",")
X = data[:,1:]

# get responses
y = []
with open(path + '/example_data/online_lda/runs_and_results-it2.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    # skip first row
    next(csvreader)
    for row in csvreader:
        y.append(row[10])
# Note: the random forest needs type float
Y = np.array([float(i) for i in y[:len(X)]])

# setting up config space:
param_file = path + '/example_data/online_lda/param-file.txt'
f = open(param_file, 'rb')
cs = ConfigurationSpace()
for row in f:
    cs.add_hyperparameter(UniformFloatHyperparameter("%s" %row[0:4], np.float(row[6:9]), np.float(row[10:13]),np.float(row[18:21])))
param = cs.get_hyperparameters()

# create an instance of fanova with data for the random forest and the configSpace
f = fanova.fANOVA(X = X, Y = Y)

# marginal for first parameter
p_list = [0]
res = f.quantify_importance(p_list)
print(res)

# getting the most important pairwise marginals sorted by importance
best_margs = f.get_most_important_pairwise_marginals(n=3)
print(best_margs)

# visualizations:

# directory in which you can find all plots
plot_dir = path + '/example_data/test_plots'
# first create an instance of the visualizer with fanova object and configspace
vis = visualizer.Visualizer(f, cs)
# creating all plots in the directory
vis.create_all_plots(plot_dir)
