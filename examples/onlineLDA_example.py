import numpy as np
from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
from fanova import fANOVA
import fanova.visualizer

import os
path = os.path.dirname(os.path.realpath(__file__))

# get sample data from online lda
X = np.loadtxt(path + '/example_data/online_lda/online_lda_features.csv', delimiter=",")
Y = np.loadtxt(path + '/example_data/online_lda/online_lda_responses.csv', delimiter=",")

# setting up config space:
param_file = path + '/example_data/online_lda/param-file.txt'
f = open(param_file, 'rb')

cs = ConfigurationSpace()
for row in f:
    cs.add_hyperparameter(UniformFloatHyperparameter("%s" %row[0:4].decode('utf-8'), np.float(row[6:9]), np.float(row[10:13]),np.float(row[18:21])))
param = cs.get_hyperparameters()


# create an instance of fanova with data for the random forest and the configSpace
f = fANOVA(X = X, Y = Y, config_space = cs)

# marginal for first parameter
p_list = (0, )
res = f.quantify_importance(p_list)
print(res)

p2_list = ('Col1', 'Col2')
res2 = f.quantify_importance(p2_list)
print(res2)
p2_list = ('Col0', 'Col2')
res2 = f.quantify_importance(p2_list)
print(res2)
p2_list = ('Col1', 'Col0')
res2 = f.quantify_importance(p2_list)
print(res2)

# getting the most important pairwise marginals sorted by importance
best_p_margs = f.get_most_important_pairwise_marginals(n=3)
print(best_p_margs)

triplets = f.get_triple_marginals(['Col0','Col1', 'Col2'])
print(triplets)
# visualizations:

# directory in which you can find all plots
plot_dir = path + '/example_data/online_lda'
# first create an instance of the visualizer with fanova object and configspace
vis = fanova.visualizer.Visualizer(f, cs, plot_dir)
# generating plot data for col0
mean, std, grid = vis.generate_marginal(0)

# creating all plots in the directory
vis.create_all_plots()
#vis.create_most_important_pairwise_marginal_plots(plot_dir, 3)
