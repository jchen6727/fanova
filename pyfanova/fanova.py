import numpy as np
from collections import OrderedDict
import itertools as it
from ConfigSpace.hyperparameters import CategoricalHyperparameter

class fANOVA(object):
    def __init__(self, cs, forest):
        """
        Calculate and provide midpoints and sizes from the forest's 
        split values in order to get the marginals
        
        Parameters
        ------------
        cs : ConfigSpace instantiation
        
        forest: trained random forest
        
        """
        # initialize a dictionary with parameter dims
        self.param_dic = OrderedDict([('parameters', OrderedDict([]))])       
        self.the_forest = forest
        self.cs = cs
        
        self.cs_params =self.cs.get_hyperparameters()
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
            
        for tree_split_values in forest_split_values:
            # considering the hyperparam settings
            updated_array = []
            # categoricals are treated differently 
            j = 0          
            for i in range(len(tree_split_values)):
                if isinstance(tree_split_values[i], (CategoricalHyperparameter)):
                    i +=1
                else:
                    plus_setting = [val_mins[j]] + tree_split_values[i]
                    plus_setting.append(val_maxs[j])
                    j +=1
                    updated_array.append(plus_setting)
            var_splits = updated_array
            sizes =[]
            midpoints =  []
            for var_splits in tree_split_values:
                if isinstance(var_splits, (CategoricalHyperparameter)):
                    midpoint_p = var_splits
                    size = 1
                    midpoints.append(midpoint_p)
                    sizes.append(size)
                else:
                    # compute the midpoints
                    midpoint_p = (1/2)* (np.array(var_splits[1:]) + np.array(var_splits[:-1]))
                    size = np.array(var_splits[1:]) - np.array(var_splits[:-1])
                    midpoints.append(midpoint_p)
                    sizes.append(size)
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
                            interval_size.append(interval_sizes[dim_helper[i]])
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
        tree_midpoints = self.all_midpoints[0]
        sample = np.ones(len(tree_midpoints))
        sample[:] = np.nan
        for i in range(len(dimlist)):
            sample[dimlist[i]] = valuesToPredict[i]
        preds = self.the_forest.marginalized_prediction(sample)
    
        return np.mean(preds), np.var(preds)

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
