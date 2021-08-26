from ipywidgets.widgets import Label, FloatProgress, FloatSlider, Button, Checkbox,FloatRangeSlider, Button, Text,FloatText,\
Dropdown,SelectMultiple, Layout, HBox, VBox, interactive, interact, Output,jslink
from IPython.display import display, clear_output
from ipywidgets import GridspecLayout
from ipywidgets.widgets.interaction import show_inline_matplotlib_plots

import numpy as np
import pandas as pd
from copy import deepcopy as dc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

from tqdm import tqdm_notebook as tqdm

import XPyS
from XPyS import bkgrds as backsub
from .helper_functions import index_of, guess_from_data

import XPyS.gui_element_dicts
import XPyS.config as cfg
import XPyS.VAMAS
import os
import h5py

from sklearn.decomposition import PCA, NMF
from scipy.stats import pearsonr
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

import matplotlib.patches as mpatches
import decimal

from IPython import embed as shell

class HellaSpectra:
    """
    Class for holding lots of spectra objects as well as their spectra in matrix and dataframe format. Used
    for statistical analysis and data exploration.
    """
    def __init__(self,spectra_objects, spectra=None):
        self.info = []
        self.spectra_objects = spectra_objects
        """
        
        Parameters
        ----------
        spectra_objects: dictionary
            dictionary of spectra objects. Keys are sample names or any identifier and values are XPyS.spectra.spectra object insances

        Notes
        -----


        Examples
        --------


        """
    

    def bgsuball(self,subpars):
        """
        Calls bg_sub on all the spectra objects held in self.spectra_objects

        Parameters
        ----------
        subpars: list
            list of the background subtraction parameters. See spectra.bg_sub

        """
        for sample in self.spectra_objects.items():
            try:
                sample[1].bg_sub(subpars=subpars)
            except:
                print('Could Not bg_sub ',sample[0])

    def _get_xnew(self,emin=None,emax=None,step = 0.1):
        """Get the interpolation positions of the energy

        Parameters
        ----------
        emin: int,float
            minimum binding energy value

        emax: int,float
            maximum binding energy value

        step: int, float
            step between binding energies 
        """
        # print('2')
        ynew = None
        if self.spectra_type == 'raw':
            if (emin == None) and (emax != None):
                # print('2.1')
                emin, _emax = self._auto_bounds()
            elif (emin != None) and (emax == None):
                # print('2.2')
                _emin, emax = self._auto_bounds()
            elif (emin == None) and (emax == None):
                # print('2.3')
                emin, emax = self._auto_bounds()
            self._check_bounds(emin,emax)
            xnew = np.arange(emin, emax, step)

        elif self.spectra_type == 'sub':
            # print('2.4')
            emin,emax = self._auto_bounds()

            self._check_bounds(emin,emax)
            xnew = np.arange(emin, emax, step)
        
        # round the interpolation Binding Energies off to the thousandth
        xnew = np.round(xnew,3)
        return xnew
            # probably want to make the spacing even given by ev step 0.1 but fuck it for now.
        # print(emin,emax)
        # xnew = np.arange(emin, emax+step, step)
        # d = decimal.Decimal(str(step)).as_tuple().exponent
        # if d < 0:
        #     xnew = np.round(xnew,-1*d)
        # print(np.min(xnew),np.max(xnew))

        # try:
            # first = True

    def _interp_and_build(self,emin=None,emax=None,step = 0.1):
        """Interpolate the spectra to all have the same binding energies using _get_xnew then build the
        spectra and the params dataframes.

        Parameters
        ----------
        emin: int,float
            minimum binding energy value

        emax: int,float
            maximum binding energy value

        step: int, float
            step between binding energies 
        """

        ynew = None
        self.energy = self._get_xnew(emin=emin,emax=emax,step = 0.1)
        
        df_list = []
        df_params_list = []
        start_idx = 0
        for name,spectra_obj in self.spectra_objects.items():

            n_data = len(spectra_obj.__dict__[self.dset[1]])

            idx_id = np.arange(start_idx,start_idx+n_data)  # Need an index id to perform joins on different dataframes
            start_idx = start_idx+n_data
            first = True
            # Spectra DataFrame
            for i in range(n_data):
                f = interp1d(spectra_obj.__dict__[self.dset[0]], spectra_obj.__dict__[self.dset[1]][i],kind = 'cubic')
                if not first:
                    ynew = np.vstack((ynew,f(self.energy)))
                else:
                    ynew = f(self.energy)
                    first = False
            if self.target != None:
                df_list.append(pd.DataFrame(ynew,index = [[self.target[name]]*n_data,[name]*n_data,idx_id]))
                
            else:
                df_list.append(pd.DataFrame(ynew,index = [[0]*n_data,[name]*n_data,idx_id]))

            # Parameters DataFrame
            if hasattr(spectra_obj,'fit_results'):
                _dd = {key: [spectra_obj.fit_results[i].params.valuesdict()[key] \
                            for i in range(len(spectra_obj.fit_results))] \
                    for key in  spectra_obj.fit_results[0].params.valuesdict().keys()}

                if self.target != None:
                    df_params_list.append(pd.DataFrame(_dd,index = [[self.target[name]]*n_data,[name]*n_data,idx_id]))
                else:
                    df_params_list.append(pd.DataFrame(_dd,index = [[0]*len(spectra_obj.fit_results),[name]*len(spectra_obj.fit_results),idx_id]))

            
        df_spectra = pd.concat(df_list)
        df_spectra.columns = self.energy
        

        df_params = pd.concat(df_params_list) 
        
        # Add names to the indices
        # if self.target != None:
        df_spectra.index.set_names(['target', 'name','id'], inplace=True)
        df_params.index.set_names(['target', 'name','id'], inplace=True)
        # else:
            # df_spectra.index.set_names(['id', 'name'], inplace=True)
            # df_params.index.set_names(['id', 'name'], inplace=True)

        return df_spectra, df_params



    def _auto_bounds(self):
        """Search through the spectra to get bounds for the interpolation since some spectra have 
        different binding energy ranges
        """
        largest_min_value = np.max([np.min(so[1].__dict__[self.dset[0]]) for so in self.spectra_objects.items()])
        smallest_max_value = np.min([np.max(so[1].__dict__[self.dset[0]]) for so in self.spectra_objects.items()])

        emin = (np.round(100*largest_min_value,2)+1)/100
        emax = (np.round(100*smallest_max_value,2)-1)/100

        return emin,emax

    def _check_bounds(self,min_bnd,max_bnd):
        """check to make sure the bounds are not outside the binding energies of any of the spectra"""

        check_min = [so[0] for so in self.spectra_objects.items() if not np.min(np.round(so[1].__dict__[self.dset[0]],2)) <= min_bnd <=np.max(np.round(so[1].__dict__[self.dset[0]],2))]
        check_max = [so[0] for so in self.spectra_objects.items() if not np.min(np.round(so[1].__dict__[self.dset[0]],2)) <= max_bnd <=np.max(np.round(so[1].__dict__[self.dset[0]],2))]
        
        if check_min !=[]:
            raise ValueError('The specified bounds are outside the minimum values for', check_min )
        if check_max != []:
            raise ValueError('The specified bounds are outside the maximum values for', check_max )

    def build_dataframes(self,emin=None,emax=None,spectra_type = 'raw',subpars = None,step = 0.1,target = None):
        """
        Build a 2d array of all of the spectra from the spectra objects

        Parameters
        ----------
        emin: int,float
            minimum binding energy value

        emax: int,float
            maximum binding energy value

        spectra_type: str
            Option to interpolate the raw data or backgroudn subtracted data. ('raw' or 'sub')

        subpars: list
            list of the background subtraction parameters. See spectra.bg_sub

        step: int, float
            step between binding energies     

        target: dict, None
            Dictionary of the target    
        """
        self.spectra_type = spectra_type
        self.target = target

        if target != None:
            if sorted(list(self.spectra_objects.keys())) != sorted(list(self.target.keys())):
                raise KeyError('There is a spectra object that is not in the target dictionary')


        if self.spectra_type == 'sub':
            self.dset = ['esub','isub']

            if subpars == None:
                raise ValueError('You must specify subpars')
            self.bgsuball(subpars = subpars)
        elif self.spectra_type == 'raw':
            self.dset = ['E','I']
            
        self.spectra, self.params = self._interp_and_build(emin = emin, emax = emax,step = step)
        self._spectra = self.spectra



    def reset(self):
        """Reset spectra array and dataframe to initial loaded states"""
        self.spectra = dc(self._spectra)
        self.info = []

    def normalize(self):
        """Normalize all of the spectra"""
        _yN = np.empty(self.spectra.values.shape)

        for i in range(len(_yN)):
            _yN[i,:] = self.spectra.values[i,:]/np.trapz(self.spectra.values[i,:])

        self.spectra = pd.DataFrame(_yN,columns = self.spectra.columns,index = self.spectra.index)
        self._update_info('Normalized')

    def plot_spectra(self,offset = 0,avg = False):
        """
        Plot all the spectra

        Parameters
        ----------
        offset: int, float
            offset each spectra by set amount or stacked display
        
        avg: bool
            Option to plot mean spectra. Default False
        """
        fig, ax = plt.subplots()
        if not avg:
            for i in range(len(self.spectra.values)):
                ax.plot(self.energy,self.spectra.values[i,:]+i*offset)
        if avg:
            ax.plot(self.energy,self.spectra.values.mean(axis=0))

        ax.set_xlabel('Binding Energy',fontsize = 15)
        if 'Normalized' in self.info:
            ax.set_ylabel('Counts/sec (N.U.)',fontsize = 15)
        else:
            ax.set_ylabel('Counts/sec',fontsize = 15)

    def _update_info(self,message):
        """Update order of operations on the spectra array to keep track processing history"""
        if self.info != []:
            if self.info[-1] == message:
                return
            else:
                self.info.append(message)
        else:
            self.info.append(message)


    def peak_tracker(self,peak_pos,energy= None,spectra_matrix=None,plotflag = True,**kws):
        """
        Get the peak positions for all the spectra using guess_from_data() in helper_functions module.
        Then plot the differences in the peak positions from the peak_pos parameter

        Parameters
        ----------
        peak_pos: int, float
            Guess position of the peak. guess_from_data() will search in vicinity for the maximum
        
        """
        if energy ==None:
            energy = self.energy
        if spectra_matrix == None:
            spectra_matrix = self.spectra.values
        
        cen = np.empty(len(spectra_matrix))
        amp = np.empty(len(spectra_matrix))

        for i in range(len(spectra_matrix)):
            amp[i],cen[i] = guess_from_data(energy,spectra_matrix[i],peakpos = peak_pos,**kws)

        self.peaktrack = cen-peak_pos
        if plotflag:
            plt.plot(self.peaktrack,'o-')


    def align_peaks(self,peak_pos,plotflag = True,**kws):
        """
        Align all the peaks.  Will search for maximum of peak aroudn peak_pos and then adjust spectra to peak_pos

        Parameters
        ----------
        peak_pos: int, float
            Guess position of the peak. guess_from_data() will search in vicinity for the maximum
        
        """

        spectra_matrix = self.spectra.values

        cen = np.empty(len(spectra_matrix))
        amp = np.empty(len(spectra_matrix))
        mv_spec = np.empty([len(spectra_matrix),len(self.energy)])


        for i in range(len(spectra_matrix)):

            amp[i],cen[i] = guess_from_data(self.energy,spectra_matrix[i],peakpos = peak_pos,**kws)

            mv_ev = np.round(cen[i] - peak_pos,2)

            mv_pts = np.int(mv_ev*(len(self.energy)/(self.energy[-1] - self.energy[0])))

            if mv_pts ==0:
                mv_spec[i] = spectra_matrix[i]

            elif mv_pts > 0:
                mv_spec[i] = np.asarray(list(spectra_matrix[i][mv_pts:]) + [0]*mv_pts)
            
            elif mv_pts < 0:
                mv_spec[i] = np.asarray([0]*np.abs(mv_pts)+list(spectra_matrix[i][:mv_pts]))

        if plotflag:
            fig,ax = plt.subplots()
            for i in range(len(spectra_matrix)):
                ax.plot(self.energy,mv_spec[i])

            ax.set_xlabel('Binding Energy',fontsize = 15)
            
            if 'Normalized' in self.info:
                ax.set_ylabel('Counts/sec (N.U.)',fontsize = 15)
            else:
                ax.set_ylabel('Counts/sec',fontsize = 15)

            plt.axvline(self.energy[index_of(self.energy,peak_pos)])

        self.aligned_pos = peak_pos
        self.spectra = pd.DataFrame(mv_spec,columns = self.spectra.columns,index = self.spectra.index)

        self._update_info('adjusted to '+str(peak_pos))


    def par_histogram(self,pars):
        """
        Create a histogram of the desired input parameters

        Parameters
        ----------
        pars: list
            list of strings of the parameters to get histograms of distribution
        """

        def gaussian(x, mean, amplitude, standard_deviation):
            return amplitude * np.exp( - (x - mean)**2 / (2*standard_deviation ** 2))
        
        fig,ax = plt.subplots(figsize = (12,8))

        prefix = ['_'.join(p.split('_')[:-1])+'_' for p in pars]

        # If the prefix is stored in gui_element_dicts use that color, otherwise create colors
        if all(elem in list(XPyS.gui_element_dicts.element_color.keys())  for elem in prefix):
            colors = [XPyS.gui_element_dicts.element_color[prefix[i]] for i in range(len(pars))]
        else:
            colormap = plt.cm.nipy_spectral
            colors = [colormap(i) for i in np.linspace(0, 1,len(pars))]
            ax.set_prop_cycle('color', colors)

        center_stats = {}
        for par in enumerate(pars):
            bin_heights, bin_borders, _ = ax.hist(self.params[par[1]].values, bins='auto', label='histogram',color=colors[par[0]])
            
            try:
                bin_centers = bin_borders[:-1] + np.diff(bin_borders) / 2
                center_stats[par[1]], _ = curve_fit(gaussian, bin_centers, bin_heights, p0=[self.params[par[1]].values.mean(), 40, 0.5])

                x_interval_for_fit = np.linspace(bin_borders[0]-1, bin_borders[-1]+1, 10000)
                ax.plot(x_interval_for_fit, gaussian(x_interval_for_fit, *center_stats[par[1]]), label='fit',color=colors[par[0]])
            except:
                print('Gaussian not good for ',par)
            
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                    ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(24)
            
        ax.legend(pars,fontsize = '16')

        leg_patches = []
        for p in enumerate(pars):
            leg_patches.append(mpatches.Patch(color=colors[p[0]], label=p[1]))

        ax.legend(handles=leg_patches,bbox_to_anchor=(1.05, 0.5), loc='upper left',fontsize = 18)
        return fig, ax

    ## Using dictionary

    def make_spectra(self,spectra_model,number):

        """
        Make a bunch of simulated spectra using the parameters in the dataframe as bounds. Useful for training
        Machine Learning Models

        Parameters
        ----------
        params: lmfit Parameters object
            Parameters object to be used to evaluate the model function

        mod: lmfit Model object
            Model object to be used to evaluate the parameters

        pairlist: list of tuples
            pairlist of prefixes in model object

        number: int
            number of spectra to create
        """
        params = spectra_model.pars
        mod = spectra_model.model
        pairlist = spectra_model.pairlist
        spec_train = []
        par_train = []

        par_train_row = []
        pars = [pars for pars in [par for component_pars in [model_component._param_names for model_component in mod.components] \
            for par in component_pars if any([p[0] in par for p in pairlist])] if not 'skew' in pars and not 'fraction' in pars] 
        
        randpar = {par:[] for par in pars}
        for i in tqdm(range(number)):
            for par in pars:

                if ('amplitude' in par) or ('sigma' in par):
                    _randpar = np.min(self.df_params[par].values) + (np.max(self.df_params[par].values)-np.min(self.df_params[par].values))*np.random.rand()
                    randpar[par].append(_randpar)
                    params[par].set(_randpar)

                if 'center' in par:
    #                 if par == 'Nb_52_center':
    #                     variation = 0.005
    #                 else:
    #                     variation = 0.01
    #                 _randpar = center_stats[par][0] + 0.1*randn()
                    # _randpar = center_stats[par][0] + center_stats[par][2]*randn()
                    _randpar = np.min(self.df_params[par].values) + (np.max(self.df_params[par].values)-np.min(self.df_params[par].values))*np.random.rand()
                    randpar[par].append(_randpar)
                    params[par].set(_randpar)
                

            spec_train.append(mod.eval(params = params,x= self.energy))

        spec_train = np.asarray(spec_train)
    
        return spec_train, randpar



    # Dimensionality Reduction

    def pca(self,n_pca = 3):
        """
        Perform Principal Component Analysis on the spectra and plot the principal components

        Parameters
        ----------
        n_pca: int
            number of principal components to evaluate
        """
        self.n_pca = n_pca
        pca = PCA(n_components=self.n_pca)

        # PCA of raw signal
        X_r = pca.fit(self.spectra.values)
        X_tr = pca.fit_transform(self.spectra.values)

        if self.info != []:
            print(' '.join(self.info))

        print('explained variance ratio: %s'
            % str(pca.explained_variance_ratio_ *100 ) )
        print('cumulative variance: %s'
            % str(np.cumsum(pca.explained_variance_ratio_ *100) ) ) 

        self.pc_names = ['pc{}'.format(i) for i in range(1,self.n_pca+1)]
        # Build Dictionary of principal components
        pc = {self.pc_names[i] : X_tr[:,i] for i in range(len(self.pc_names))}
        self.pc = pd.DataFrame(pc,index = self.spectra.index)
        self.pc_vec = X_r.components_
        self._plotpcavecs()

    def _plotpcavecs(self):
        fig,ax = plt.subplots()

        for i in range(self.n_pca):
            ax.plot(self.pc_vec[i,:])
            
        ax.legend(self.pc_names,bbox_to_anchor=(1.05, 1), loc='upper left',fontsize = 18)

        ax.set_title('Raw Data',fontsize = 18)
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                    ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(18)

    def plotpca(self,x = 'pc1',y = 'pc2',z=None):
        if z is None:
            self._plot_dimred_2D(x,y,self.pc)
        else:
            self._plot_dimred_3D(x,y,z,self.pc)

    def _plot_dimred_2D(self,x,y,df):
        """
        Plot 2d scatter plot of principal components

        Parameters
        ----------
        x: str 'pc1','pc2','nmf1','nmf2',...etc
            dimensionality reduction component to plot on x-axis

        y: str 'pc1','pc2','nmf1','nmf2',...etc
            dimensionality reduction component to plot on y-axis

        """

        fig,(ax1,ax2) = plt.subplots(1,2,figsize = (18,6))

        sample_names = list(df.index.levels[1])
        n_samples=len(sample_names)
        n_targets = len(df.index.levels[0])

        colormap = plt.cm.nipy_spectral
        colors = [colormap(i) for i in np.linspace(0, 1,n_samples)]
        colors_targets = [colormap(i) for i in np.linspace(0, 1,n_targets)]
        ax1.set_prop_cycle('color', colors)
        ax2.set_prop_cycle('color', colors_targets)

        # First plot the principal components color coded by sample
        for sample in sample_names:
            ax1.plot(df.xs(sample,level = 'name')[x].values,df.xs(sample,level = 'name')[y].values,'o',markersize = 10)
        
        # Next plot the principal components color coded by target
        for target in df.index.levels[0]:  # Level 0 is the target level
            ax2.plot(df.xs(target,level = 'target')[x].values,df.xs(target,level = 'target')[y].values,'o',markersize = 10)

        if 'pc' in x:
            title = 'principal Components'
        elif 'nmf' in x:
            title = 'Non-Negative Matrix Factorization'
        for ax in [ax1, ax2]:
            ax.set_title(title)
            ax.set_xlabel(x)
            ax.set_ylabel(y)
            ax.tick_params('x',labelrotation=80)

            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +\
                ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(20)

        ax1.legend(list(self.spectra_objects.keys()),bbox_to_anchor=(1.05, 1.2), loc='upper left',fontsize = 18)

        leg_patches = []
        for tar in enumerate(df.index.levels[0]):
            leg_patches.append(mpatches.Patch(color=colors_targets[tar[0]], label=tar[1]))

        ax2.legend(handles=leg_patches,bbox_to_anchor=(1.05, 0.5), loc='upper left',fontsize = 18)
        fig.tight_layout()


    def _plot_dimred_3D(self,X, Y, Z, df):
        """
        Plot 3d scatter plot of principal components

        Parameters
        ----------
        x: str 'P1','P2',...etc
            principal component to plot on x-axis, Default 'P1'

        y: str 'P1','P2',...etc
            principal component to plot on y-axis, Default 'P2'

        z: str 'P1','P2',...etc
            principal component to plot on z-axis, Default 'P3'
        """

        fig = plt.figure(figsize = (12,5))
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')

        sample_names = list(df.index.levels[1])
        n_samples=len(sample_names)
        n_targets = len(df.index.levels[0])

        colormap = plt.cm.nipy_spectral
        colors = [colormap(i) for i in np.linspace(0, 1,n_samples)]
        colors_targets = [colormap(i) for i in np.linspace(0, 1,n_targets)]
        ax1.set_prop_cycle('color', colors)
        ax2.set_prop_cycle('color', colors_targets)

        colormap = plt.cm.nipy_spectral
        colors = [colormap(i) for i in np.linspace(0, 1,n_samples)]
        colors_targets = [colormap(i) for i in np.linspace(0, 1,n_targets)]
        ax1.set_prop_cycle('color', colors)
        ax2.set_prop_cycle('color', colors_targets)
        

        # First plot the principal components color coded by sample
        for sample in sample_names:
            ax1.plot(df.xs(sample,level = 'name')[X].values,df.xs(sample,level = 'name')[Y].values,df.xs(sample,level = 'name')[Z].values,'o',markersize = 10)

        # Next plot the principal components color coded by target
        for target in df.index.levels[0]:  # Level 0 is the target level
            ax2.plot(df.xs(target,level = 'target')[X].values,df.xs(target,level = 'target')[Y].values,df.xs(target,level = 'target')[Z].values,'o',markersize = 10)
        
        for ax in [ax1,ax2]:
            ax.set_xlabel(X,fontsize = 16)
            ax.set_ylabel(Y,fontsize = 16)
            ax.set_zlabel(Z,fontsize = 16)

        # ax1.legend(list(self.spectra_objects.keys()),bbox_to_anchor=(1.05, 1.2), loc='upper left',fontsize = 18)
        # leg_patches = []
        # for tar in enumerate(self.pc.index.levels[0]):
        #     leg_patches.append(mpatches.Patch(color=colors_targets[tar[0]], label=tar[1]))
        # ax2.legend(handles=leg_patches,bbox_to_anchor=(1.05, 0.5), loc='upper left',fontsize = 18)

        fig.tight_layout()

    def nmf(self,n_nmf = 3,**kws):
        """
        Find Non-Negative Matrix Factorization of spectra

        Parameters
        ----------
        n_nmf: int
            number of non-negative matrix components
        
        nmf_kws: dict
            keywords to pass to sklearn.decomposition.NMF
        """
        self.n_nmf = n_nmf
        if np.any(self.spectra.values < 0):
            self.spectra.values[np.where(self.spectra.values < 0)] = 0


        # Calculate NMF
        model = NMF(n_components=self.n_nmf, random_state=0, **kws)

        W = model.fit_transform(self.spectra)
        self.H = model.components_

        if self.info != []:
            print(' '.join(self.info))

        self.nmf_names = ['nmf{}'.format(i) for i in range(1,self.n_nmf+1)]

        nmf = {self.nmf_names[i] : W[:,i] for i in range(len(self.nmf_names))}
        self.W = pd.DataFrame(nmf,index = self.spectra.index)

        self._plotnmfvecs()

    def _plotnmfvecs(self):
        fig,ax = plt.subplots()

        for i in range(self.n_nmf):
            ax.plot(self.H[i,:])

        ax.legend(self.nmf_names,bbox_to_anchor=(1.05, 1), loc='upper left',fontsize = 18)

        ax.set_title('Raw Data',fontsize = 18)
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                    ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(18)

    def plotnmf(self,x = 'nmf1',y = 'nmf2',z=None):
        if z is None:
            self._plot_dimred_2D(x,y,self.W)
        else:
            self._plot_dimred_3D(x,y,z,self.W)



    def be_corr(self,pars,plotflag = True):
        """
        Find and plot correlation of a parameter against the Binding Energies for all the spectra

        Parameters
        ----------
        par: str 
            Parameter to correlate against Binding Energies

        plotflag: bool
            Option to plot. Default = True
        """
        if not type(pars) is list:
            raise TypeError('pars argument must be a list')


        print(' '.join(self.info))
        spec_and_params = self.spectra.join(self.params,how = 'inner')
        self.BEcorrs = {}
        for par in pars:
            self.BEcorrs[par] = {}
            self.BEcorrs[par]['pmax'] = spec_and_params.corr()[par].iloc[0:self.spectra.values.shape[1]].max()
            self.BEcorrs[par]['emax'] = spec_and_params.corr()[par].iloc[0:self.spectra.values.shape[1]].idxmax()
            # print('maximum correlation of',np.round(pmax,3),'at',np.round(emax,2))
        if plotflag:
            fig,axs = plt.subplots(len(self.BEcorrs.keys()),2,figsize = (12,4*len(self.BEcorrs.keys())))
            axs = axs.ravel()

            axi = 0
            for par in self.BEcorrs.keys():

                axs[axi].plot(self.energy,spec_and_params.corr()[par].iloc[0:self.spectra.values.shape[1]].values)
                axs[axi].plot(self.BEcorrs[par]['emax'],self.BEcorrs[par]['pmax'],'x',marker = 'x',markersize = 15,mew = 5,color = 'black')
                axs[axi].set_ylabel('p-value',fontsize = 14)
                axs[axi].set_xlabel('B.E. (eV)',fontsize = 14)
                axs[axi].set_title(par,fontsize = 14)
                axi += 1

                t = spec_and_params.corr()[par].iloc[0:self.spectra.values.shape[1]].values
                sc = axs[axi].scatter(self.energy,self.spectra.values.mean(axis=0),c = t, cmap=cm.bwr, vmin=-1, vmax=1, s=100)
                specmax_idx = index_of(self.energy,self.BEcorrs[par]['emax'])
                axs[axi].plot(self.BEcorrs[par]['emax'],self.spectra.values.mean(axis=0)[specmax_idx],marker = 'x',markersize = 15,mew = 5,color = 'black')
                axs[axi].set_ylabel('Counts/sec',fontsize = 14)
                axs[axi].set_xlabel('B.E. (eV)',fontsize = 14)
                fig.colorbar(sc,ax=axs[axi])
                axi += 1
            fig.tight_layout()

            return fig, axs

    def scattercorr(self,Xs,Ys):
        """
        Plots the correllation scatter plots and finds the pearson p-number between the given parameters.

        Parameters
        ----------
        Xs: list 
            list of parameter names on x-axes

        Ys: list 
            list of parameter names on y-axes
        """
        fulldf = self.spectra.join(self.params,how = 'inner')
        if hasattr(self,'pc'):
            fulldf = fulldf.join(self.pc,how = 'inner')
        if hasattr(self,'W'):
            fulldf = fulldf.join(self.W,how = 'inner')

        fig, axs = plt.subplots(len(Ys),len(Xs),figsize = (4*len(Xs),4*len(Ys)))
        axs = axs.ravel()

        ylabels = []
        corrmat = np.empty([len(Ys),len(Xs)])
        a = 0
        for i in range(len(Ys)):
            j =0

            if type(Ys[i]) == float:
                axs[a].set_ylabel(Ys[i],fontsize = 26)
            else:
                axs[a].set_ylabel(Ys[i],fontsize = 26)

            for j in range(len(Xs)):

                try:
                    axs[a].plot(fulldf[Xs[j]].values,fulldf[Ys[i]].values,'o')
                    corrmat[i][j], _ = pearsonr(fulldf[Xs[j]].values, fulldf[Ys[i]].values) 

                    axs[a].set_title('Pearsons correlation: %.3f' % corrmat[i][j],fontsize = 16)
                    axs[a].set_xlabel(Xs[j],fontsize = 26)
                    axs[a].set_xticklabels('')
                    axs[a].set_yticklabels('')
                    a+=1


                except:
                    pass

        fig.tight_layout()

        return fig, axs

    def linfit(self,X,Y,plotflag = True):
        """
        Linear regression of specified entries in self.df

        Parameters
        ----------
        Xs: list 
            list of x-axes df entries

        Ys: list 
            list of y-axes df entries
        """
        fulldf = self.spectra.join(self.params,how = 'inner')
        if hasattr(self,'pc'):
            fulldf = fulldf.join(self.pc,how = 'inner')
        if hasattr(self,'W'):
            fulldf = fulldf.join(self.W,how = 'inner')

        # We can rewrite the line equation as y = Ap, where A = [[x 1]] and p = [[m], [b]]
        autofitparlist = []
        fig, axs = plt.subplots()

        lfpars = {}

        x = np.array(fulldf[X])
        y = np.array(fulldf[Y])
        A = np.hstack((x.reshape(-1,1),  np.ones(len(x)).reshape(-1,1)))
        lfpars['m'],lfpars['b'] = np.linalg.lstsq(A, y,rcond = None)[0]

        # m = np.linalg.lstsq(x.reshape(-1,1), y,rcond = None)[0][0]

        if plotflag == True:
            axs.plot(x, y,'o')
            axs.plot(x, lfpars['m']*x+lfpars['b'])
            axs.set_xlabel(str(Y) + ' vs ' + str(X),fontsize = 14)
            fig.tight_layout()

        return lfpars, fig, axs