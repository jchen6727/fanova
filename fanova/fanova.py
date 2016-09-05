import numpy as np
import math
from collections import OrderedDict
import itertools as it
import pyrfr.regression
import ConfigSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformFloatHyperparameter


class fANOVA(object):
    def __init__(self, X=None, Y=None, cs=None, forest=None, 
                num_trees=16, seed=None, bootstrapping=True,
                points_per_tree = None, max_features=None,
                min_samples_split=0, min_samples_leaf=0,
                max_depth=64):

        """
        Calculate and provide midpoints and sizes from the forest's 
        split values in order to get the marginals
        
        Parameters
        ------------
        X: matrix with the features
        
        Y: vector with the response values
        
        cs : ConfigSpace instantiation
        
        forest: trained random forest

        num_trees: number of trees in the forest to be fit
        
        seed: seed for the forests randomness
        
        bootstrapping: whether or not to bootstrap the data for each tree
        
        points_per_tree: number of points used for each tree (only subsampling if bootstrapping is false)
        
        max_features: number of features to be used at each split, default is 70%
        
        min_samples_split: minimum number of samples required to attempt to split 
        
        min_samples_leaf: minimum number of samples required in a leaf
        
        max_depth: maximal depth of each tree in the forest
        """

        
        # if no ConfigSpace is specified, let's build one with all continuous variables
        if (cs is None):
            if (X is None) or (Y is None):
                raise RuntimeError("If no ConfigSpace argument is given, you have to "
                                    "provide data for X and Y.")


            # if no info is given, use min and max values of each variable as bounds
            pcs = list(zip( np.min(X,axis=0), np.max(X, axis=0) ))
            cs = ConfigSpace.ConfigurationSpace()
            for i in range(len(pcs)):
                cs.add_hyperparameter(UniformFloatHyperparameter("%i" %i, pcs[i][0], pcs[i][1]))

        self.cs = cs        
        self.cs_params =self.cs.get_hyperparameters()
        # at this point we have a valid ConfigSpace object
        # check if param number is correct etc:
        if X.shape[1] != len(self.cs_params):
            raise RuntimeError('Number of parameters in config space do not match input X')
        for i in range(len(self.cs_params)):
            if not isinstance(self.cs_params[i], (CategoricalHyperparameter)):
                if (np.max(X[:,i]) > self.cs_params[i].upper) or (np.min(X[:,i]) > self.cs_params[i].lower):
                    raise RuntimeError('Some sample values from X are not in the given interval')
            else:
                unique_vals = set(X[:,i])
                if len(unique_vals) > self.cs_params[i]._num_choices:
                    raise RuntimeError('There are some categoricals missing in the config space specification')
                if len(unique_vals) < self.cs_params[i]._num_choices:
                    raise RuntimeError('There are too many categoricals specified in the config space')

        
        # if no forest has been trained yes, than 
        if (forest is None):
            if (X is None) or (Y is None):
                raise RuntimeError("If no pyrfr forest is present, you have to "
                                    "provide data for X and Y.")

            forest = pyrfr.regression.binary_rss()
            forest.num_trees=num_trees
            forest.seed= np.random.randint(2**31-1) if seed is None else seed
            forest.do_bootstrapping=bootstrapping
            forest.num_data_points_per_tree=X.shape[0] if points_per_tree is None else points_per_tree
            forest.max_features = (X.shape[1]*7)//10 if max_features is None else max_features

            forest.min_samples_to_split = min_samples_split
            forest.min_samples_in_leaf = min_samples_leaf
            forest.max_depth=max_depth
            forest.epsilon_purity = 1e-8

            # TODO: Get types from the ConfigSpace
            types = np.zeros(X.shape[1],dtype=np.uint)
            data = pyrfr.regression.numpy_data_container(X, Y, types)
            forest.fit(data)


        # initialize a dictionary with parameter dims
        self.param_dic = OrderedDict([('parameters', OrderedDict([]))])       
        self.the_forest = forest
        types = np.zeros(len(self.cs_params), dtype=np.uint)
        # getting split values
        forest_split_values = self.the_forest.all_split_values(types)
        
        
        self.all_midpoints = []
        self.all_sizes = []
        
        # set the max and min of values and store them
        val_mins = []
        val_maxs = []
        for param in self.cs_params:
            if not isinstance(param, (CategoricalHyperparameter)):
                val_mins.append(param.lower)
                val_maxs.append(param.upper)
            else:
                val_mins.append(np.nan)
                val_maxs.append(np.nan)
            
        for tree_split_values in forest_split_values:
            # considering the hyperparam settings
            updated_array = []
            # categoricals are treated differently        
            for i in range(len(tree_split_values)):
                if not math.isnan(val_mins[i]):
                    plus_setting = [val_mins[i]] + tree_split_values[i]
                    plus_setting.append(val_maxs[i])
                    updated_array.append(plus_setting)
                else:
                    updated_array.append(tree_split_values[i])
            var_splits = updated_array
            sizes =[]
            midpoints =  []
            i = 0
            for var_splits in tree_split_values:
                if isinstance(self.cs_params[i], (CategoricalHyperparameter)):
                    midpoint_p = var_splits
                    size = np.ones(len(midpoint_p))
                    midpoints.append(midpoint_p)
                    sizes.append(size)
                else:
                    # compute the midpoints
                    midpoint_p = (1/2)* (np.array(var_splits[1:]) + np.array(var_splits[:-1]))
                    size = np.array(var_splits[1:]) - np.array(var_splits[:-1])
                    midpoints.append(midpoint_p)
                    sizes.append(size)
                i += 1
            # all midpoints treewise for the whole forest
            self.all_midpoints.append(midpoints)
            self.all_sizes.append(sizes)
        

    def get_marginal(self, dim_list):
        """
        Returns the marginal of selected parameters
                
        Parameters
        ----------
        dim_list: list
                Contains the indices of ConfigSpace for the selected parameters 
                (starts with 0) 
             
        Returns
        -------
        double
            marginal value
        """        
        
        K = len(dim_list)
        for k in range(1,K+1):
            for dimension in tuple(it.combinations(dim_list, k)):
                if self.param_dic['parameters'].has_key(dimension):
                    thisMarginalVarianceContribution = self.param_dic['parameters'][dimension]['MarginalVarianceContribution']                    
                
                else:
                    for tree in range(len(self.all_midpoints)):
                        sample = np.ones(len(self.all_midpoints[tree]))
                        combi_midpoints = []
                        sizes = []
                        dim_helper = []
                        for dim in dimension:
                            combi_midpoints.append(self.all_midpoints[tree][dim])
                            sizes.append(self.all_sizes[tree][dim])
                            dim_helper.append(dim)
                        midpoints = list(it.product(*combi_midpoints))
                        interval_sizes = list(it.product(*sizes))
                        sample[:] = np.nan
                        weightedSum = 0
                        weightedSumOfSquares = 0
                        for points in midpoints:
                            interval_size = []
                            singleVarianceContributions = []
                            for i in range(len(points)):
                                sample[dim_helper[i]] = points[i]
                                if not isinstance(self.cs_params[i], (CategoricalHyperparameter)):
                                    interval_size.append(interval_sizes[dim_helper[i]])
                                else:
                                    interval_size.append(1)
                            pred = self.the_forest.marginalized_prediction(sample)
                            marg = pred[tree]
                            weightedSum += marg*np.prod(np.array(interval_size))
                            weightedSumOfSquares += np.power(marg,2)*np.prod(np.array(interval_size))
                            thisMarginalVarianceContribution = weightedSumOfSquares - np.power(weightedSum,2)
                            if len(dimension)== 1:
                                # store into dictionary as one param
                                self.param_dic['parameters'][dimension] = {}
                                self.param_dic['parameters'][dimension]['Name'] = self.cs_params[dim].name
                                self.param_dic['parameters'][dimension]['MarginalVarianceContribution'] = thisMarginalVarianceContribution 
                            else:
                                for i in range(len(points)):
                                    singleVarianceContributions.append(self.param_dic['parameters'][(dim_helper[i], )]['MarginalVarianceContribution'])
                                for singleVarianceContribution in singleVarianceContributions:
                                    thisMarginalVarianceContribution -= singleVarianceContribution
                                    params = tuple(dim_helper)
                                    # store it into dictionary as tuple
                                    self.param_dic['parameters'][params] = {}
                                    self.param_dic['parameters'][params]['MarginalVarianceContribution'] = thisMarginalVarianceContribution
        
        return thisMarginalVarianceContribution

        
    def get_marginal_for_values(self, dimlist, valuesToPredict):
        """
        Returns the marginal of selected parameters for specific values
                
        Parameters
        ----------
        dimlist: list
                Contains the indices of ConfigSpace for the selected parameters 
                (starts with 0) 
        
        valuesToPredict: list
                Contains the values to be predicted
              
        Returns
        -------
        double
            marginal value
        """
        num_dims = len(self.all_midpoints[0])
        sample = np.empty(num_dims, dtype=np.float)
        sample.fill(np.NAN)
        for i in range(len(dimlist)):
            sample[dimlist[i]] = valuesToPredict[i]
        preds = self.the_forest.marginalized_prediction(sample)
    
        return np.mean(preds), np.std(preds)

    def get_most_important_pairwise_marginals(self, n=10):
        """
        Returns the n most important pairwise marginals from the whole ConfigSpace
            
        Parameters
        ----------
        n: int
             The number of most relevant pairwise marginals that will be returned
          
        Returns
        -------
        list: 
             Contains the n most important pairwise marginals
        """
        pairwise_marginals = []
        dimensions = np.arange(len(self.cs_params))
        param_combis = list(it.combinations(dimensions,2))
        for combi in param_combis:
            pairwise_marginal_performance = self.get_marginal(combi)
            pairwise_marginals.append((pairwise_marginal_performance, combi[0], combi[1]))
        
        pairwise_marginal_performance = sorted(pairwise_marginals, reverse=True)
        important_pairwise_marginals = [(p1, p2) for marginal, p1, p2  in pairwise_marginal_performance[:n]]

        return important_pairwise_marginals
