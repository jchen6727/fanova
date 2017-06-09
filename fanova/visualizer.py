from ConfigSpace.hyperparameters import CategoricalHyperparameter
import os
import numpy as np
import matplotlib.pyplot as plt
import itertools as it
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

class Visualizer(object):

    def __init__(self, fanova, cs):
        """        
        Parameters
        ------------
        cs : ConfigSpace instantiation
        
        forest: trained random forest        
        """
        self.fanova = fanova   
        self.cs_params = cs.get_hyperparameters()

    def create_all_plots(self, directory, **kwargs):
        """
        Creates plots for all main effects and stores them into a directory
        
        Parameters
        ------------
        directory: str
                Path to the directory in which all plots will be stored
        """
        assert os.path.exists(directory), "directory %s doesn't exist" % directory

        for i in range(len(self.cs_params)):
            param = i
            param_name = self.cs_params[param].name
            plt.close()
            outfile_name = os.path.join(directory, param_name.replace(os.sep, "_") + ".png")
            print("creating %s" % outfile_name)
            if isinstance(self.cs_params[param], (CategoricalHyperparameter)):
                self.plot_categorical_marginal(param, show=False)
            else:
                self.plot_marginal(param, show=False, **kwargs)
            plt.savefig(outfile_name)
        # additional pairwise plots:
        dimensions = []
        for i in range(len(self.cs_params)):
            if not isinstance(self.cs_params[i], (CategoricalHyperparameter)):
                dimensions.append(i)
        combis = list(it.combinations(dimensions,2))
        for combi in combis:
            param_names = []
            for p in combi:
                param_names.append(self.cs_params[p].name)
            plt.close()
            outfile_name = os.path.join(directory, str(param_names).replace(os.sep, "_") + ".png")
            print("creating %s" % outfile_name)
            self.plot_pairwise_marginal(combi, **kwargs)
            plt.savefig(outfile_name)

    def generate_pairwise_marginal(self, param_list, resolution=20):
        """
        Creates a plot of pairwise marginal of a selected parameters
        
        Parameters
        ------------
        param_list: list of ints or strings
            Contains the selected parameters  
        
        resolution: int
            Number of samples to generate from the parameter range as
            values to predict

        """
        assert len(param_list) == 2, "You have to specify 2 (different) parameters"
        
        grid_list = []
        param_names = []
        for p in range(len(param_list)):
            if type(p) == str:
                p = self.cs.get_idx_by_hyperparameter_name(p)
            lower_bound = self.cs_params[p].lower
            upper_bound = self.cs_params[p].upper
            param_names.append(self.cs_params[p].name)
            grid = np.linspace(lower_bound, upper_bound, resolution)
            grid_list.append(grid)

        zz = np.zeros([resolution * resolution])
        for i, y_value in enumerate(grid_list[1]):
            for j, x_value in enumerate(grid_list[0]):
                zz[i * resolution + j] = self.fanova.marginal_mean_variance_for_values(param_list, [x_value, y_value])[0]

        zz = np.reshape(zz, [resolution, resolution])

        return grid_list, zz

    def plot_pairwise_marginal(self, param_list, resolution=20, show=True):
        """
        Creates a plot of pairwise marginal of a selected parameters
        
        Parameters
        ------------
        param_list: list of ints or strings
            Contains the selected parameters
        
        resolution: int
            Number of samples to generate from the parameter range as
            values to predict

        """
        param_names = []
        for p in range(len(param_list)):
            if type(p) == str:
                p = self.cs.get_idx_by_hyperparameter_name(p)
            lower_bound = self.cs_params[p].lower
            upper_bound = self.cs_params[p].upper
            param_names.append(self.cs_params[p].name)
        grid_list, zz = self.generate_pairwise_marginal(param_list, resolution)

        display_xx, display_yy = np.meshgrid(grid_list[0], grid_list[1])

        fig = plt.figure()
        ax = Axes3D(fig)

        surface = ax.plot_surface(display_xx, display_yy, zz, rstride=1, cstride=1, cmap=cm.jet, linewidth=0, antialiased=False)
        ax.set_xlabel(param_names[0])
        ax.set_ylabel(param_names[1])
        ax.set_zlabel("Performance")
        fig.colorbar(surface, shrink=0.5, aspect=5)
        if show:
            plt.show()
        else:
            return plt

    def generate_marginal(self, param, resolution=100):
        """
        Creates marginals of a selected parameter for own plots
        
        Parameters
        ------------
        param: int or str
            Index of chosen parameter in the ConfigSpace (starts with 0)
        
        resolution: int
            Number of samples to generate from the parameter range as
            values to predict
        
        """
        if type(param) == str:
            param = self.cs.get_idx_by_hyperparameter_name(param)
        lower_bound = self.cs_params[param].lower
        upper_bound = self.cs_params[param].upper
        param_name = self.cs_params[param].name
        grid = np.linspace(lower_bound, upper_bound, resolution)
      
        mean = np.zeros(resolution)
        std = np.zeros(resolution)

        dim = [param]
        for i in range(0, resolution):
            (m, v) = self.fanova.marginal_mean_variance_for_values(dim, [grid[i]])
            mean[i] = m
            std[i] = np.sqrt(v)

        return mean, std, grid

    def plot_marginal(self, param, resolution=100, log_scale=False, show=True):
        """
        Creates a plot of marginal of a selected parameter
        
        Parameters
        ------------
        param: int or str
            Index of chosen parameter in the ConfigSpace (starts with 0)
        
        resolution: int
            Number of samples to generate from the parameter range as
            values to predict
        
        log_scale: boolean
            If log scale is required or not
        """
        if type(param) == str:
            param = self.cs.get_idx_by_hyperparameter_name(param)
        lower_bound = self.cs_params[param].lower
        upper_bound = self.cs_params[param].upper
        param_name = self.cs_params[param].name

        mean, std, grid = self.generate_marginal(param, resolution)
        mean = np.asarray(mean)
        std = np.asarray(std)
        
        lower_curve = mean - std
        upper_curve = mean + std

        if log_scale or (np.diff(grid).std() > 0.000001):
            plt.semilogx(grid, mean, 'b')
        else:
            plt.plot(grid, mean, 'b')
        plt.fill_between(grid, upper_curve, lower_curve, facecolor='red', alpha=0.6)
        plt.xlabel(param_name)
        
        plt.ylabel("Performance")
        if show:
            plt.show()
        else:
            return plt

    def get_categorical_marginal(self, param):
        """
        Creates marginals of a selected categorical parameter
        
        Parameters
        ------------
        param: int or str
            Index of chosen categorical parameter in the ConfigSpace (starts with 0)
        
        """
        if type(param) == str:
            param = self.cs.get_idx_by_hyperparameter_name(param)
        param_name = self.cs_params[param].name
        labels= self.cs_params[param].choices
        categorical_size  = self.cs_params[param]._num_choices
        marginals = [self.fanova.marginal_mean_variance_for_values([param], [i]) for i in range(categorical_size)]
        mean, std = list(zip(*marginals))

        return mean, std
        
    def plot_categorical_marginal(self, param, show=True):
        """
        Creates a plot of marginal of a selected categorical parameter
        
        Parameters
        ------------
        param: int or str
            Index of chosen categorical parameter in the ConfigSpace (starts with 0)
        
        """
        if type(param) == str:
            param = self.cs.get_idx_by_hyperparameter_name(param)
        param_name = self.cs_params[param].name
        labels= self.cs_params[param].choices
        categorical_size  = self.cs_params[param]._num_choices

        mean, std = self.get_categorical_marginal(param)
        indices = np.arange(1,categorical_size+1, 1)
        b = plt.boxplot([[x] for x in mean])
        plt.xticks(indices, labels)
        min_y = mean[0]
        max_y = mean[0]
        # blow up boxes 
        for box, std_ in zip(b["boxes"], std):
            y = box.get_ydata()
            y[2:4] = y[2:4] + std_
            y[0:2] = y[0:2] - std_
            y[4] = y[4] - std_
            box.set_ydata(y)
            min_y = min(min_y, y[0] - std_)
            max_y = max(max_y, y[2] + std_)
        
        plt.ylim([min_y, max_y])
        
        plt.ylabel("Performance")
        plt.xlabel(param_name)
        if show:
            plt.show()
        else:
            return plt
            
    def plot_pairwise_categorical_marginal(self, param_list, show=True):
        """
        Creates a plot of marginal of a selected categorical parameter
        
        Parameters
        ------------
        param_list: list of ints or strings
            Contains the selected categorical parameters
        
        """
        if type(param) == str:
            param = self.cs.get_idx_by_hyperparameter_name(param)
        param_name = self.cs_params[param].name
        labels= self.cs_params[param].choices
        categorical_size  = self.cs_params[param]._num_choices

        mean, std = self.get_categorical_marginal(param)
        indices = np.arange(1,categorical_size+1, 1)
        b = plt.boxplot([[x] for x in mean])
        plt.xticks(indices, labels)
        min_y = mean[0]
        max_y = mean[0]
        # blow up boxes 
        for box, std_ in zip(b["boxes"], std):
            y = box.get_ydata()
            y[2:4] = y[2:4] + std_
            y[0:2] = y[0:2] - std_
            y[4] = y[4] - std_
            box.set_ydata(y)
            min_y = min(min_y, y[0] - std_)
            max_y = max(max_y, y[2] + std_)
        
        plt.ylim([min_y, max_y])
        
        plt.ylabel("Performance")
        plt.xlabel(param_name)
        if show:
            plt.show()
        else:
            return plt
        
    def create_most_important_pairwise_marginal_plots(self, directory, n=20):
        """
        Creates plots of the n most important pairwise marginals of the whole ConfigSpace
        
        Parameters
        ------------
        directory: str
            Path to the directory in which all plots will be stored
        
        n: int
             The number of most relevant pairwise marginals that will be returned
            
        """
        most_important_pairwise_marginals = self.fanova.get_most_important_pairwise_marginals(n)
        print(most_important_pairwise_marginals)
        for param1, param2 in most_important_pairwise_marginals:
            param_names = [self.cs_params[param1].name, self.cs_params[param2].name]
            outfile_name = os.path.join(directory, str(param_names).replace(os.sep, "_") + ".png")
            plt.clf()
            print("creating %s" % outfile_name)
            self.plot_pairwise_marginal([param1, param2], show=False)
            plt.savefig(outfile_name)
