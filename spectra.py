from ipywidgets.widgets import Label, FloatProgress, FloatSlider, Button, Checkbox,FloatRangeSlider, Button, Text,FloatText,\
Dropdown,SelectMultiple, Layout, HBox, VBox, interactive, interact, Output,jslink
from IPython.display import display, clear_output
from ipywidgets import GridspecLayout
from ipywidgets.widgets.interaction import show_inline_matplotlib_plots

import time
import threading
import logging
import math
import numpy as np
from copy import deepcopy as dc
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
import pickle
import lmfit as lm
# from lmfit.model import load_model

import sys
sys.path.append("/Volumes/GoogleDrive/My Drive/XPS/XPS_Library")
import xps
# from xps.io import loadmodel
from xps import bkgrds as backsub
# from xps import bkgrds as background_sub
from xps.helper_functions import *
from xps.gui_element_dicts import *
from xps.auto_fitting import *

import xps.VAMAS

import os
import glob

"""
Load pre-defined models fitting
"""

def loadmodel(element):

    f = open('/Volumes/GoogleDrive/My Drive/XPS/XPS_Library/xps/models/load_model_info.pkl', 'rb')   # 'r' for reading; can be omitted
    load_dict = pickle.load(f)         # load file content as mydict
    f.close() 

    mod = lm.model.load_model(load_dict[element]['model_path'])
    pars = pickle.load(open(load_dict[element]['params_path'],"rb"))
    pairlist = load_dict[element]['pairlist']
    element_ctrl = load_dict[element]['element_ctrl']

    return mod, pars, pairlist, element_ctrl



class spectra:

    def __init__(self,sample_object=None,orbital=None,parameters=None,model=None,bg_info = None, pairlist=None,element_ctrl=None,\
        spectra_name = None, BE_adjust = 0,load_spectra_object = False,load_model = False, autofit = False):
            # def __init__(self, func, independent_vars=None, param_names=None,
            #      nan_policy='raise', prefix='', name=None, **kws):
        """Class for holding spectra of a particular elemental scan

        The model function will normally take an independent variable
        (generally, the first argument) and a series of arguments that are
        meant to be parameters for the model. It will return an array of
        data to model some data as for a curve-fitting problem.

        Parameters
        ----------
        func : callable
            Function to be wrapped.
        independent_vars : list of str, optional
            Arguments to func that are independent variables (default is None).
        param_names : list of str, optional
            Names of arguments to func that are to be made into parameters
            (default is None).
        nan_policy : str, optional
            How to handle NaN and missing values in data. Must be one of
            'raise' (default), 'propagate', or 'omit'. See Note below.
        prefix : str, optional
            Prefix used for the model.
        name : str, optional
            Name for the model. When None (default) the name is the same as
            the model function (`func`).
        **kws : dict, optional
            Additional keyword arguments to pass to model function.

        Notes
        -----
        1. Parameter names are inferred from the function arguments,
        and a residual function is automatically constructed.

        2. The model function must return an array that will be the same
        size as the data being modeled.

        3. nan_policy sets what to do when a NaN or missing value is
        seen in the data. Should be one of:

           - 'raise' : Raise a ValueError (default)

           - 'propagate' : do nothing

           -  'omit' : drop missing data

        Examples
        --------
        The model function will normally take an independent variable (generally,
        the first argument) and a series of arguments that are meant to be
        parameters for the model.  Thus, a simple peak using a Gaussian
        defined as:

        >>> import numpy as np
        >>> def gaussian(x, amp, cen, wid):
        ...     return amp * np.exp(-(x-cen)**2 / wid)

        can be turned into a Model with:

        >>> gmodel = Model(gaussian)

        this will automatically discover the names of the independent variables
        and parameters:

        >>> print(gmodel.param_names, gmodel.independent_vars)
        ['amp', 'cen', 'wid'], ['x']

        """

        self.bg_info = bg_info
        self.BE_adjust = BE_adjust
        # self.prefixlist = [self.mod.components[i].prefix for i in range(len(self.mod.components))]   #Need this

        if load_spectra_object == False:
            return
            # self.parent_sample = sample_object.sample_name
            # self.orbital = orbital
            # if spectra_name == None:
            #     self.spectra_name = dc(self.parent_sample + '_' + orbital)
            # else:
            #     self.spectra_name = spectra_name + '_' + orbital

            # self.carbon_adjust = carbon_adjust
            
            # if self.carbon_adjust:
            #     print('Adjusted energy to carbon reference')
            #     self.E = dc(sample_object.data[orbital]['energy']) - self.carbon_adjust
            # else:
            #     self.E = dc(sample_object.data[orbital]['energy'])

            # self.I = dc(sample_object.data[orbital]['intensity'])

            # self.isub = dc([[] for k in range(len(self.I))])
            # self.bg = dc([[] for k in range(len(self.I))])
            # self.bgpars = dc([[] for k in range(len(self.I))])
            # self.area = dc([[] for k in range(len(self.I))])  
            
            # self.data = dc(sample_object.data[orbital])

            # if load_model == False:
            #     self.params_full = parameters
            #     self.mod = model
            #     self.pairlist = pairlist
            #     self.element_ctrl = element_ctrl
            # else:
            #     self.mod, self.params_full, self.pairlist, self.element_ctrl = loadmodel(load_model)

            # if type(self.params_full) == list:
            #     self.params = dc(self.params_full[0])
            # else:
            #     self.params = dc(self.params_full)

            # self.autofit = autofit
            # self.leglist = ['Data','Fit Guess'] + [self.pairlist[i][0] for i in range(len(self.pairlist))]
            # self.prefixlist = [self.mod.components[i].prefix for i in range(len(self.mod.components))]


        elif load_spectra_object == True:

            spectra_path = os.path.join('/Volumes/GoogleDrive/My Drive/XPS/XPS_Library/xps/samples',\
                            sample_object,orbital)
            f = open(os.path.join(spectra_path,'spectra_attributes.pkl'), 'rb')   
            savedict_load = pickle.load(f)         
            f.close() 

            for key in savedict_load:
                self.__dict__[key] = savedict_load[key]

            self.mod = lm.model.load_model(os.path.join(spectra_path,self.spectra_name+'_model.sav'))

            fit_result_path_list = [f for f in glob.glob(spectra_path+"/*.sav") \
                if 'fit_result' in f]
            self.fit_results = [lm.model.load_modelresult(fit_result_path_list[i]) for i in range(len(fit_result_path_list))]

    def load_experiment_spectra_from_vamas(self, vamas_obj,orbital = None):
        print(orbital)
        self.orbital = orbital
        
        # if not hasattr(self, 'spectra_name') == None:
        #     self.spectra_name = self.parent_sample + '_' + self.orbital
        # else:
        #     self.spectra_name = spectra_name + '_' + self.orbital
            
        self.E = np.asarray([-1*block.abscissa() for block in vamas_obj.blocks if ''.join(block.species_label.split()) in [self.orbital]][0])
        self.I = np.asarray([block.ordinate(0)/(block.number_of_scans_to_compile_this_block * block.signal_collection_time) \
                for block in vamas_obj.blocks if ''.join(block.species_label.split()) in [self.orbital]])

    ### Analysis functions
        
    def bg_sub(self,crop_details=None,idx = None,UT2_params = None):


        if not crop_details == None:
            self.bg_info = crop_details

        if idx == None:
            self.isub = [[] for k in range(len(self.I))]
            self.bg = [[] for k in range(len(self.I))]
            self.area = [[] for k in range(len(self.I))]
            self.bgpars = [[] for k in range(len(self.I))]
            bg_range = range(len(self.I))
        else:
            bg_range = [idx]
        for i in bg_range:

            if self.bg_info[1] == 'shirley':
                Ei = [index_of(self.E,self.bg_info[0][0]), index_of(self.E,self.bg_info[0][1])]
                self.esub = self.E[min(Ei):max(Ei)]
                intensity_crop = self.I[i][min(Ei):max(Ei)]
                self.bg[i] = backsub.shirley(self.esub, intensity_crop)
                self.isub[i]= intensity_crop - self.bg[i]  
                self.bgpars[i] = None
                
            elif self.bg_info[1] =='linear':
                Ei = [index_of(self.E,self.bg_info[0][0]), index_of(self.E,self.bg_info[0][1])]
                self.esub= self.E[min(Ei):max(Ei)]
                intensity_crop = self.I[i][min(Ei):max(Ei)]
                self.bg[i] = backsub.linear(self.esub, intensity_crop)
                self.isub[i] = intensity_crop - self.bg[i]  
                self.bgpars[i] = None

            elif self.bg_info[1] =='UT2':
                self.esub = self.E

                if UT2_params is None:
                    toupars = lm.Parameters()
                    toupars.add('B', value =self.bg_info[2][0], min = 0,vary = self.bg_info[2][1])
                    toupars.add('C', value =self.bg_info[2][2], min = 0,vary = self.bg_info[2][3])
                    toupars.add('D', value =0, min = 0,vary = 0)
                
                    if self.bg_info[0] == None:
                        fit_ind = (0,5)
                    else:
                        fit_ind = ( index_of(self.E,np.max(self.bg_info[0])),index_of(self.E,np.min(self.bg_info[0])) )

                    # Put in an error here incase the croplims are outside of the bounds
                    fitter = lm.Minimizer(backsub.Tougaard_fit, toupars,fcn_args=(self.I[i],self.E), fcn_kws={'fit_inds': fit_ind})
                    result_tou = fitter.minimize()
                    
                    self.bg[i],self.isub[i] = backsub.Tougaard(result_tou.params, self.I[i],self.E)
                    self.bgpars[i] = result_tou.params
                elif UT2_params != None:
                    
                    self.bg[i],self.isub[i] = backsub.Tougaard(UT2_params, self.I[i],self.E)
                    self.bgpars[i] = UT2_params
                    
            self.area[i] = np.trapz( np.flip(self.isub[i]) , np.flip(self.esub) )

        # return self.esub, self.isub[i], self.bg[i], self.bgpars[i], self.area[i]


    def fit(self,fit_method = 'powell',specific_points = None, plotflag = True, track = True):


        if not hasattr(self,"fit_results"):
            self.fit_results = [[] for i in range(len(self.I))]            
            
        if specific_points is None:
            specific_points = np.arange(len(self.I))

        if track:
            pbar = tqdm(total=len(specific_points))

        for i in specific_points:

            self.fit_results[i]  = self.mod.fit(self.isub[i], self.params, x=self.esub, method = fit_method)     

            if track:
                pbar.update()
        
        if track:
            pbar.close()

        self.fit_results_idx = [j for j,x in enumerate(self.fit_results) if x]
        print('%%%% Fitting done! %%%%')
        
        if plotflag:
            self.plot_fitresults()
        
            
            
            
            
    def plot_fits(self,specific_points = None, plot_with_background_sub = False,ref_lines = False,colors = None,infig = None, inaxs = None, offset = 0):
            

        if colors is None:
            hue = element_color
        else:
            hue = colors

        if specific_points is None:
        
            subplotrows =  math.ceil((len(self.fit_results_idx)/2)) 
            plot_idx = self.fit_results_idx
            print(plot_idx)
        else:
            subplotrows =  math.ceil((len(specific_points)/2))
            plot_idx = specific_points

        # subplotrows =  math.ceil((len(self.plot_idx)/2))  
         
        
        
        if len(plot_idx) == 1:
        # if len(self.fit_results_idx) == 1:

            if inaxs is None:
                fig,axs = plt.subplots(subplotrows,1, figsize = (10,8))
            else:
                fig = infig
                axs = inaxs

            for i in plot_idx:
            # for i in self.fit_results_idx:

                p = [[] for i in range(len(self.pairlist))]
                fit_legend = [element_text[element[0]] for element in self.pairlist]

                    
                print('Plotting fit_results')
                if plot_with_background_sub == True:
                    bkgrd = self.bg[i] - self.bg[-1]
                    axs.plot(self.fit_results[i].userkws['x'], self.isub[i] + bkgrd + offset,'o')
                    axs.plot(self.fit_results[i].userkws['x'], self.fit_results[i].best_fit + bkgrd + offset)
                else:
                    axs.plot(self.fit_results[i].userkws['x'], self.isub[i] + offset,'o')
                    axs.plot(self.fit_results[i].userkws['x'], self.fit_results[i].best_fit + offset)

                for pairs in enumerate(self.pairlist):

                    if plot_with_background_sub == True:

                        bkgrd = self.bg[i] - self.bg[i][-1]
                        p[pairs[0]] = axs.fill_between(self.fit_results[i].userkws['x'],\
                                        sum([self.fit_results[i].eval_components()[comp] for comp in pairs[1]]) + \
                                        bkgrd + offset, bkgrd + offset,\
                                        color = hue[pairs[1][0]], alpha=0.3)
                    else:
                        p[pairs[0]] = axs.fill_between(self.fit_results[i].userkws['x'],\
                                        sum([self.fit_results[i].eval_components()[comp] for comp in pairs[1]]) + offset, offset,\
                                        color = hue[pairs[1][0]], alpha=0.3)



                    if ref_lines == True:
                        axs.axvline(x = element_refpos[pairs[1][0]],color = hue[pairs[1][0]])

                        
                        
                fig.legend(p,fit_legend,bbox_to_anchor=(0.9, 0.4, 0.5, 0.5), loc='lower center',fontsize=30)

                # if 'pos names' in self.data:
                #     axs.set_title(self.spectra_name+'_'+str(self.data['pos names'][i]), fontsize = 30)
                # else:
                #     axs.set_title(self.spectra_name+'_'+str(i), fontsize = 30)

                axs.set_xlabel('Binding Energy (eV)',fontsize=30)
                axs.set_ylabel('Counts/sec',fontsize=30)
                axs.set_xlim(np.max(self.esub),np.min(self.esub))
                axs.tick_params(labelsize=20)

                fig.tight_layout()
            
            
        else:
            if inaxs is None:
                fig,axs = plt.subplots(subplotrows,2, figsize = (18,subplotrows*6))
            else:
                fig = infig
                axs = inaxs
            
            
            axs = axs.ravel()
        
            for i in enumerate(plot_idx):
            # for i in enumerate(self.fit_results_idx):
                axs[i[0]].plot(self.fit_results[i[1]].userkws['x'], self.isub[i[1]] + offset,'o')
                axs[i[0]].plot(self.fit_results[i[1]].userkws['x'], self.fit_results[i[1]].best_fit + offset)

                p = [[] for i in range(len(self.pairlist))]
                fit_legend = [element_text[element[0]] for element in self.pairlist]

                for pairs in enumerate(self.pairlist):

                    if plot_with_background_sub == True:
                        bkgrd = self.bg[i] - self.bg[-1]
                        p[pairs[0]] = axs[i[0]].fill_between(self.fit_results[i[1]].userkws['x'],\
                                        sum([self.fit_results[i[1]].eval_components()[comp] for comp in pairs[1]]) + \
                                        bkgrd + offset, y2 = bkgrd + offset*np.ones(len(self.esub)),\
                                        color = hue[pairs[1][0]], alpha=0.3)
                    else:
                        p[pairs[0]] = axs[i[0]].fill_between(self.fit_results[i[1]].userkws['x'],\
                                                    sum([self.fit_results[i[1]].eval_components()[comp] for comp in pairs[1]]) + offset\
                                                        ,y2 = offset*np.ones(len(self.esub)),\
                                                    color = hue[pairs[1][0]], alpha=0.3)
                    
                    if ref_lines == True:
                        axs[i[0]].axvline(x = element_refpos[pairs[1][0]],color = hue[pairs[1][0]])


                fig.legend(p,fit_legend,bbox_to_anchor=(0.5, 1, 0.5, 0.5), loc='lower center',fontsize=30)

                # if 'pos names' in self.data.keys():
                #     print(i)
                #     axs[i[0]].set_title(self.spectra_name+'_'+str(self.data['pos names'][i[1]]), fontsize = 30)
                # else:
                #     axs[i[0]].set_title(self.spectra_name+'_'+str([i[1]]), fontsize = 30)

                axs[i[0]].set_xlabel('Binding Energy (eV)',fontsize=30)
                axs[i[0]].set_xlim(np.max(self.esub),np.min(self.esub))
                axs[i[0]].tick_params(labelsize=20)

                fig.tight_layout()

        return fig, axs





























###### Attempt at more concise plot


    # def plot(self,specific_fits = None, plot_with_background_sub = False,ref_lines = False,colors = None,infig = None, inaxs = None, offset = 0):
            
    #     # if self.plot_all_chkbx.value is True:
    #     #     print('Plotting all spectra ...')
    #     #     self.plot_idx = [j for j,x in enumerate(self.fit_results) if x]

    #     # elif self.plot_all_chkbx.value is False:
    #     #     print('Plotting' + str(self.spectra_to_fit_widget.value) + ' spectra ...')

    #     #     if self.spectra_to_fit_widget.value[0] == None:
    #     #         print('Error!: You are trying to Not plot all results and Not fit any spectra')
    #     #         return

    #     #     elif 'All' in self.spectra_to_fit_widget.value:
    #     #         self.plot_idx = [j for j,x in enumerate(self.fit_results) if x]

    #     #     else:
    #     #         self.plot_idx = dc(list(self.spectra_to_fit_widget.value)) 
    #     if colors is None:
    #         hue = element_color
    #     else:
    #         hue = colors

    #     if specific_fits is None:
        
    #         subplotrows =  math.ceil((len(self.fit_results_idx)/2)) 
    #         plot_idx = self.fit_results_idx
    #         print(plot_idx)
    #     else:
    #         subplotrows =  math.ceil((len(specific_fits)/2))
    #         plot_idx = specific_fits

    #     # subplotrows =  math.ceil((len(self.plot_idx)/2))  
         
        
        
    #     if len(plot_idx) == 1:
    #     # if len(self.fit_results_idx) == 1:

    #         if inaxs is None:
    #             fig,axs = plt.subplots(subplotrows,1, figsize = (10,8))
    #         else:
    #             fig = infig
    #             axs = inaxs

    #         for i in plot_idx:
    #         # for i in self.fit_results_idx:

    #             p = [[] for i in range(len(self.pairlist))]
    #             fit_legend = [element_text[element[0]] for element in self.pairlist]

                    
    #             print('Plotting fit_results')
    #             if plot_with_background_sub == True:
    #                 bkgrd = self.data['bkgd'][i] - self.data['bkgd'][i][-1]
    #                 axs.plot(self.fit_results[i].userkws['x'], self.data['isub'][i] + bkgrd + offset,'o')
    #                 axs.plot(self.fit_results[i].userkws['x'], self.fit_results[i].best_fit + bkgrd + offset)
    #             else:
    #                 axs.plot(self.fit_results[i].userkws['x'], self.data['isub'][i] + offset,'o')
    #                 axs.plot(self.fit_results[i].userkws['x'], self.fit_results[i].best_fit + offset)

    #             for pairs in enumerate(self.pairlist):

    #                 if plot_with_background_sub == True:

    #                     bkgrd = self.data['bkgd'][i] - self.data['bkgd'][i][-1]
    #                     p[pairs[0]] = axs.fill_between(self.fit_results[i].userkws['x'],\
    #                                     sum([self.fit_results[i].eval_components()[comp] for comp in pairs[1]]) + \
    #                                     bkgrd + offset, bkgrd + offset,\
    #                                     color = hue[pairs[1][0]], alpha=0.3)
    #                 else:
    #                     p[pairs[0]] = axs.fill_between(self.fit_results[i].userkws['x'],\
    #                                     sum([self.fit_results[i].eval_components()[comp] for comp in pairs[1]]) + offset, offset,\
    #                                     color = hue[pairs[1][0]], alpha=0.3)



    #                 if ref_lines == True:
    #                     axs.axvline(x = element_refpos[pairs[1][0]],color = hue[pairs[1][0]])

                        
                        
    #             fig.legend(p,fit_legend,bbox_to_anchor=(0.9, 0.4, 0.5, 0.5), loc='lower center',fontsize=30)

    #             # if 'pos names' in self.data:
    #             #     axs.set_title(self.spectra_name+'_'+str(self.data['pos names'][i]), fontsize = 30)
    #             # else:
    #             #     axs.set_title(self.spectra_name+'_'+str(i), fontsize = 30)

    #             axs.set_xlabel('Binding Energy (eV)',fontsize=30)
    #             axs.set_ylabel('Counts/sec',fontsize=30)
    #             axs.set_xlim(np.max(self.data['esub']),np.min(self.data['esub']))
    #             axs.tick_params(labelsize=20)

    #             fig.tight_layout()
            
            
    #     else:
    #         if inaxs is None:
    #             fig,axs = plt.subplots(subplotrows,2, figsize = (20,subplotrows*8))
    #         else:
    #             fig = infig
    #             axs = inaxs
            
            
    #         axs = axs.ravel()
        
    #         for i in enumerate(plot_idx):
    #         # for i in enumerate(self.fit_results_idx):
    #             axs[i[0]].plot(self.fit_results[i[1]].userkws['x'], self.data['isub'][i[1]] + offset,'o')
    #             axs[i[0]].plot(self.fit_results[i[1]].userkws['x'], self.fit_results[i[1]].best_fit + offset)

    #             p = [[] for i in range(len(self.pairlist))]
    #             fit_legend = [element_text[element[0]] for element in self.pairlist]

    #             for pairs in enumerate(self.pairlist):

    #                 if plot_with_background_sub == True:

    #                     p[pairs[0]] = axs[i[0]].fill_between(self.fit_results[i[1]].userkws['x'],\
    #                                     sum([self.fit_results[i[1]].eval_components()[comp] for comp in pairs[1]]) + \
    #                                     bkgrd + offset, bkgrd + offset,\
    #                                     color = hue[pairs[1][0]], alpha=0.3)
    #                 else:
    #                     p[pairs[0]] = axs[i[0]].fill_between(self.fit_results[i[1]].userkws['x'],\
    #                                                 sum([self.fit_results[i[1]].eval_components()[comp] for comp in pairs[1]]) + offset,offset,'--',\
    #                                                 color = hue[pairs[1][0]], alpha=0.3)
                    
    #                 if ref_lines == True:
    #                     axs[i[0]].axvline(x = element_refpos[pairs[1][0]],color = hue[pairs[1][0]])


    #             fig.legend(p,fit_legend,bbox_to_anchor=(0.5, 1, 0.5, 0.5), loc='lower center',fontsize=30)

    #             # if 'pos names' in self.data.keys():
    #             #     print(i)
    #             #     axs[i[0]].set_title(self.spectra_name+'_'+str(self.data['pos names'][i[1]]), fontsize = 30)
    #             # else:
    #             #     axs[i[0]].set_title(self.spectra_name+'_'+str([i[1]]), fontsize = 30)

    #             axs[i[0]].set_xlabel('Binding Energy (eV)',fontsize=30)
    #             axs[i[0]].set_xlim(np.max(self.data['esub']),np.min(self.data['esub']))
    #             axs[i[0]].tick_params(labelsize=20)

    #             fig.tight_layout()

    #     return fig, axs