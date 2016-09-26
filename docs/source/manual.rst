Manual
======

.. role:: bash(code)
    :language: bash

Quick Start
-----------
To run the LDA example, just download the `data <fanova/examples/example_data/online_lda.tar.gz>`_ 

First, you need to import all the necessary libraries and load the data from certain csv files:

.. code-block:: python

	import numpy as np
	from smac.configspace import ConfigurationSpace
	from ConfigSpace.hyperparameters import UniformFloatHyperparameter
	import csv
	import fanova
	import visualizer

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
	

Afterwards you can specify yor configuration space. Otherwise it will be generated automatically by counting the parameters and taking the max and min example values for each parameter as default range.

.. code-block:: python

	param_file = path + '/example_data/online_lda/param-file.txt'
	f = open(param_file, 'rb')
	cs = ConfigurationSpace()
	for row in f:
    		cs.add_hyperparameter(UniformFloatHyperparameter("%s" %row[0:4], np.float(row[6:9]), np.float(row[10:13]),np.float(row[18:21])))

Create a new Fanova object and fit the Random Forest on the specified data set.

.. code-block:: python

	f = fanova.fANOVA(X = X, Y = Y, cs=cs)

**Note**: Here you use the default setting for the random forest:

- num_trees=16
- seed=None
- bootstrapping=True
- points_per_tree = None
- max_features=None
- min_samples_split=0
- min_samples_leaf=0
- max_depth=64

You can also specify the number of trees in the random forest as well as the minimum number of points to make a new split in a tree by:

.. code-block:: python

	f = fanova.fANOVA(X = X, Y = Y, cs=cs, forest=None, 
                num_trees=16, seed=None, bootstrapping=True,
                points_per_tree = None, max_features=None,
                min_samples_split=0, min_samples_leaf=0,
                max_depth=64)


For getting the marginal for any parameter/parameter combination you need to pass them as a list.
To compute the marginal of the first parameter :

.. code-block:: python

	p_list = [0]
	res = f.get_marginal(p_list)

More functions
--------------

	* **Fanova.get_marginal_for_values(dimlist, valuesToPredict)**

    	Computes the mean and standard deviation of the parameter list **dimlist** for certain values **valuesToPredict**

	* **Fanova.get_most_important_pairwise_marginals(n)**

    	Returns the **n** most important pairwise marginals

Visualization
-------------

To visualize the single and pairwise marginals, you have to create a visualizer object which needs the fanova object and comfiguration space as arguments first.

.. code-block:: python

	vis = visualizer.Visualizer(f, cs)

You can then plot single marginals (Note that there are two different functions depending on if you have a categorical or continuous parameter):

.. code-block:: python

	vis.plot_marginal(self, param, resolution=100, log_scale=False)
 
	vis.plot_categorical_marginal(param)

For the next two functions you need to specify the path to the directory in which all plots should be stored.
Then you can plot the most important pairwise marginals or simply plot everything in one command.

.. code-block:: python

	plot_dir = path + '/example_data/test_plots'
	vis.create_most_important_pairwise_marginal_plots(plot_dir, n=3)

	# creating all plots in the directory
	vis.create_all_plots(plot_dir)


Example Code
-------------
.. literalinclude:: ../../examples/onlineLDA_example.py

