import numpy as np
from collections import OrderedDict
import itertools as it
import pyrfr.regression as reg
import pyrfr.util
import ConfigSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformFloatHyperparameter


class fANOVA(object):
    def __init__(self, X, Y, cs=None, 
                num_trees=16, seed=None, bootstrapping=True,
                points_per_tree = None, max_features=None,
                min_samples_split=0, min_samples_leaf=0,
                max_depth=64, cutoffs= (-np.inf, np.inf))):

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
        
        points_per_tree: number of points used for each tree 
                        (only subsampling if bootstrapping is false)
        
        max_features: number of features to be used at each split, default is 70%
        
        min_samples_split: minimum number of samples required to attempt to split 
        
        min_samples_leaf: minimum number of samples required in a leaf
        
        max_depth: maximal depth of each tree in the forest
        """

		pcs = [(np.nan, np.nan)]*len(X.shape[1])

        # if no ConfigSpace is specified, let's build one with all continuous variables
        if (cs is None):
            if (X is None) or (Y is None):
                raise RuntimeError("If no ConfigSpace argument is given, you have to "
                                    "provide data for X and Y.")

            # if no info is given, use min and max values of each variable as bounds
            cs = ConfigSpace.ConfigurationSpace()
            for i (mn, mx) in enumerate(zip(np.min(X,axis=0), np.max(X, axis=0) )):
                cs.add_hyperparameter(UniformFloatHyperparameter("%i" %i, mn, mx))
        
        self.cs = cs        
        self.cs_params =self.cs.get_hyperparameters()
        self.n_dims = len(self.cs_params)

        # at this point we have a valid ConfigSpace object
        # check if param number is correct etc:
        if X.shape[1] != len(self.cs_params):
            raise RuntimeError('Number of parameters in ConfigSpace object does not match input X')
        for i in range(len(self.cs_params)):
            if not isinstance(self.cs_params[i], (CategoricalHyperparameter)):
                if (np.max(X[:,i]) > self.cs_params[i].upper) or (np.min(X[:,i]) > self.cs_params[i].lower):
                    raise RuntimeError('Some sample values from X are not in the given interval')
            else:
                unique_vals = set(X[:,i])
                if len(unique_vals) > self.cs_params[i]._num_choices:
                    raise RuntimeError('There are some categoricals missing in the ConfigSpace specification')
                if len(unique_vals) < self.cs_params[i]._num_choices:
                    raise RuntimeError('There are too many categoricals specified in the ConfigSpace')



        # initialize all types as 0
        types = np.zeros(len(self.cs_params), dtype=np.uint)
        # retrieve the types and the bounds from the ConfigSpace 
        # TODO: Test if that actually works
        for i, hp in enumerate(self.cs_params):
            if isinstance( hp , CategoricalHyperparameter):
                types[i] = len(hp.choices)
                pcs[i] = (len(hp.choices), np.nan)
            else:
				pcs[i] = (hp.lower, hp.upper)

        
        # set forest options
		forest = reg.fanova_forest()
		forest.options.num_trees = num_trees
		forest.options.seed = np.random.randint(2**31-1) if seed is None else seed
		forest.options.do_bootstrapping = bootstrapping
		forest.options.num_data_points_per_tree = X.shape[0] if points_per_tree is None else points_per_tree
		forest.options.max_features = (X.shape[1]*7)//10 if max_features is None else max_features

		forest.options.tree_opts.min_samples_to_split = min_samples_split
		forest.options.tree_opts.min_samples_in_leaf = min_samples_leaf
		forest.options.tree_opts.max_depth=max_depth
		forest.options.tree_opts.epsilon_purity = 1e-8

		# create data conatainer and provide all the necessary information
		rng = reg.default_random_engine()
		data = reg.data_container()

		for i, (mn,mx) in enumerate(pcs):
			if(np.isnan(mx)):
				data.set_type_of_feature(i, mn)
			else:
				data.set_bounds(i, mn, mx)

		for i in range(len(Y)):
			data.add_data_point(X[i],Y[i])
		
		forest.fit(data, rng)

        self.the_forest = forest

        # initialize a dictionary with parameter dims
        self.variance_dict = dict()    


        # getting split values
        forest_split_values = self.the_forest.all_split_values(types)
        
        
        # all midpoints and interval sizes treewise for the whole forest
        self.all_midpoints = []
        self.all_sizes = []
        
        
        #compute midpoints and interval sizes for variables in each tree
        for tree_split_values in forest_split_values:
            sizes =[]
            midpoints =  []
            for i, split_vals in enumerate(tree_split_values):
                if np.isnan(pcs[i][1]): # categorical parameter
                    midpoints.append(split_vals)
                    sizes.append( np.ones(len(split_vals)))
                else:
                    # add bounds to split values
                    sv = np.array([pcs[i][0]] + split_vals + [pcs[i][1]])
                    # compute midpoints and sizes
                    midpoints.append((1/2)* (sv[1:] + sv[:-1]))
                    sizes.append(sv[1:] - sv[:-1])

            self.all_midpoints.append(midpoints)
            self.all_sizes.append(sizes)
            

		# capital V in the paper
        self.trees_total_variances = []
        # dict of lists where the keys are tuples of the dimensions
        # and the value list contains \hat{f}_U for the individual trees
        self.f_hat_U = {}

        self.cutoffs = cutoffs
        self.set_cutoffs(cufoffs)
        

        
    def set_cutoffs(self, cutoffs = (-np.inf, np.inf)):
		"""
			Setting the cutoffs to constrain the input space
			
			To properly do things like 'improvement over default' the
			fANOVA now supports cutoffs on the y values. These will exclude
			parts of the parameters space where the prediction is not within
			the provided cutoffs. This is is specialization of 
			"Generalized Functional ANOVA Diagnostics for High Dimensional
			Functions of Dependent Variables" by Hooker.
		"""
		self.the_forest.set_cutoffs(cutoffs[0], cutoffs[1])
		
		# reset all the variance fractions computed
		self.trees_variance_fractions = {}
		
		# update the mean predictions
		sample = np.full(self.num_dims, np.nan,dtype=np.float).tolist()
		
		for (t_idx in range(self.num_trees)):
			self.f_hat_U[tuple()] = [self.the_forest.marginal_mean_prediction_of_tree(tree_idx, sample)]
		
		# recompute the trees' total variance
		self.trees_total_variances = self.the_forest.get_trees_total_variances();


    def compute_marginals(self, dimensions):
        """
        Returns the marginal of selected parameters
                
        Parameters
        ----------
        dim_list: list
                Contains the indices of ConfigSpace for the selected parameters 
                (starts with 0) 
        """        

		# check if values has been previously computed
		if self.f_hat_U.has_key(dimensions):
			return
        
        # otherwise make sure all lower order marginals have been
        # computed, if not compute them
        for k in range(1,len(dimensions)):
			for sub_dims in it.combinations(dimensions, k):
				if not self.f_hat_U.has_key(sub_dims):
					self.get_marginal(sub_dims)
		
		# now all lower order terms have been computed
		# and this combination's \hat{f}_U
		
		self.f_hat_U[dimensions] = []
		for tree_idx in range(len(self.all_midpoints)):
			# collect all the midpoints and corresponding sizes for that tree
			midpoints = [self.all_midpoints[tree_idx][dim] for dim in dimensions]
			sizes     = [self.all_sizes[tree_idx][dim]     for dim in dimensions]
			
			stat = rfr.util.weighted_running_statistics()

			prod_midpoints = it.product(*midpoints)
			prod_sizes     = it.product(*sizes)
			
			sample = np.full(self.num_dims, np.nan, dtype=np.float);

			# make prediction for all midpoints and weigh them by the corresponding size
			for i, (m, s) in enumerate(zip(prod_midpoints, prod_sizes)):
				sample[dimenions] = m
				marg = self.the_forest.marginal_mean_prediction_of_tree(tree_idx, sample.tolist())
				stat.push( marg, np.prod(np.array(s)) )

			
			# equation (4)
			f_hat_U = stat.mean()
			for k in range(1,len(dimensions)):
				for sub_dims in it.combinations(dimensions, k):
					f_hat_U -= self.f_hat_U[sub_dims][tree_idx]
			# must not forget the overall mean
			f_hat_U -= self.f_hat_U[tuple()][tree_idx]


			self.f_hat_U[dimensions].append(f_hat_U)

        
    def marginal_mean_variance_for_values(self, dimlist, valuesToPredict):
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
        sample = np.full(self.num_dims, np.nan, dtype=np.float)
        for i in range(len(dimlist)):
            sample[dimlist[i]] = valuesToPredict[i]

        return self.the_forest.marginal_mean_variance_prediction(sample)

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
        num_dims = len(self.all_midpoints[0])
        dimensions = range(num_dims)
        for combi in it.combinations(dimensions,2):
            pairwise_marginal_performance = self.get_marginal(combi)
            pairwise_marginals.append((pairwise_marginal_performance, combi[0], combi[1]))
        
        pairwise_marginal_performance = sorted(pairwise_marginals, reverse=True)
        important_pairwise_marginals = [(p1, p2) for marginal, p1, p2  in pairwise_marginal_performance[:n]]

        return important_pairwise_marginals
