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

import sys
import xps_peakfit
import xps_peakfit.config as cfg
import xps_peakfit.models.models as xpsmodels
from xps_peakfit import bkgrds as backsub
from xps_peakfit.helper_functions import *
from xps_peakfit.gui_element_dicts import *

import xps_peakfit.VAMAS
import xps_peakfit.autofit.autofit
import os
import glob
from IPython import embed as shell


class spectra:

    def __init__(self,orbital=None,model=None,bg_info = None, pairlist=None,element_ctrl=None,\
        spectra_name = None, BE_adjust = 0,load_spectra_object = False,load_model = False, autofit = False):
            # def __init__(self, func, independent_vars=None, param_names=None,
            #      nan_policy='raise', prefix='', name=None, **kws):
        """Class for holding spectra of a particular elemental scan


        Parameters
        ----------
        orbital: str
            The name of the orbital 
        parameters=None,
        model=None,
        bg_info : list
            list containing the background subtraction parameters
            bg_info[0] : list
                [lower_bg_bound, upper_bg_bound]
            bg_info[1] : str
                type of background subtraction
            bg_info[2] : tuple
                If this exists it is the parameters for fitting the background using the UT2 method
                bg_info[2][0] : B tougaard parameter
                bg_info[2][1] : bool, whether or not to vary B
                bg_info[2][2] : C tougaard parameter
                bg_info[2][3] : bool, whether or not to vary C               
                
        pairlist : list of tuples
            In case of doublet the j-1/2 peak is linked to the j+1/2 peak. This binds the amplitudes of the 
            different peaks for the fitting

        element_ctrl: list
            Referenced to the pairlist this list specifies the peaks that are to be fit, namely the j+1/2 peaks

        spectra_name 
        BE_adjust = 0,l
        oad_spectra_object = False,
        load_model = False, 
        autofit = False
        **kws : dict, optional
           

        Notes
        -----
        

        Examples
        --------


        """

        self.bg_info = bg_info
        self.BE_adjust = BE_adjust
        self.spectra_name = spectra_name        
        self.orbital = orbital

    def load_experiment_spectra_from_vamas(self, vamas_obj,orbital = None):
        print(orbital)
        self.orbital = orbital
            
        self.E = np.asarray([-1*block.abscissa() for block in vamas_obj.blocks if ''.join(block.species_label.split()) in [self.orbital]][0])
        self.I = np.asarray([block.ordinate(0)/(block.number_of_scans_to_compile_this_block * block.signal_collection_time) \
                for block in vamas_obj.blocks if ''.join(block.species_label.split()) in [self.orbital]])


    def load_model(self,model_name):

        mod, pars, pairlist, el_ctrl = xpsmodels.load_model(model_name)
        
        self.mod = mod
        self.params = pars
        self.pairlist = pairlist
        self.element_ctrl = el_ctrl


    ### Analysis functions
        
    def bg_sub(self,subpars=None,idx = None,UT2_params = None):

        print(self.orbital)
        if not subpars == None:
            self.bg_info = subpars

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
                self.bg[i] = backsub.shirley(self.esub, intensity_crop,self.orbital)
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
                    self.bg_info[2] = (result_tou.params['B'].value, result_tou.params['B'].vary,result_tou.params['C'].value,result_tou.params['C'].vary)

                elif UT2_params != None:
                    
                    self.bg[i],self.isub[i] = backsub.Tougaard(UT2_params, self.I[i],self.E)
                    self.bgpars[i] = UT2_params
                    self.bg_info[2] = (result_tou.params['B'].value, result_tou.params['B'].vary,result_tou.params['C'].value,result_tou.params['C'].vary)


            self.area[i] = np.trapz( np.flip(self.isub[i]) , np.flip(self.esub) )



    def fit(self,fit_method = 'powell',specific_points = None, plotflag = True, track = True,fit_in_reverse = False, update_with_prev_pars = False,\
        autofit = False):

        

        if not hasattr(self,"fit_results"):
            self.fit_results = [[] for i in range(len(self.I))]            
            
        if specific_points is None:
            specific_points = np.arange(len(self.I))
        if fit_in_reverse == True:
            specific_points = specific_points[::-1]

        if track:
            pbar = tqdm(total=len(specific_points))
        # self.par_guess_track = {}
        # self.par_guess_track['O1_amplitude'] = []
        # self.par_guess_track['O2_amplitude'] = []
        if autofit:
            print('autofitting...')
        for i in specific_points:

            if autofit:
                if not hasattr(self,'autofit'):
                    self.autofit = xps_peakfit.autofit.autofit.autofit(self.esub,self.isub[i],self.orbital)
                for par in self.autofit.guess_pars.keys():
                    self.params[par].value = self.autofit.guess_pars[par]
                    # self.par_guess_track[par].append(self.autofit.guess_pars[par])
            # self.fit_results[i]  = self.mod.fit(self.isub[i], self.params, x=self.esub, method = fit_method)     

            # Try using minimuzer class to make the funtion more flexible
            fitter = lm.Minimizer(self.fcn2min,self.params,fcn_args = (self.esub, self.isub[i]))
            self.fit_results[i] = fitter.minimize(method=fit_method)

            if update_with_prev_pars ==True:
                self.params = self.fit_results[i].params.copy()

            if track:
                pbar.update()
        
        if track:
            pbar.close()

        self.fit_results_idx = [j for j,x in enumerate(self.fit_results) if x]
        print('%%%% Fitting done! %%%%')
        
        if plotflag:
            self.plot_fitresults(specific_points = specific_points)
        


    def fcn2min(self, params, x, data):
        
        data_pred = self.mod.eval(params,x = x)

        return data_pred - data            
            
            
            
    def plot_fitresults(self,specific_points = None, plot_with_background_sub = False,ref_lines = False,colors = None,infig = None, inaxs = None, offset = 0):
            

        if colors is None:
            hue = element_color
        else:
            hue = colors

        if specific_points is None:
        
            subplotrows =  math.ceil((len([j for j,x in enumerate(self.fit_results) if x])/2)) 
            plot_idx = [j for j,x in enumerate(self.fit_results) if x]
            print(plot_idx)
        else:
            subplotrows =  math.ceil((len(specific_points)/2))
            plot_idx = specific_points

        # subplotrows =  math.ceil((len(self.plot_idx)/2))  
         
        print(len(plot_idx))
        
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
                    axs.plot(self.esub, self.isub[i] + bkgrd + offset,'o')
                    axs.plot(self.esub, self.mod.eval(params = self.fit_results[i].params,x = self.esub) + bkgrd + offset)
                else:
                    axs.plot(self.esub, self.isub[i] + offset,'o')
                    axs.plot(self.esub, self.mod.eval(params = self.fit_results[i].params,x = self.esub) + offset)

                for pairs in enumerate(self.pairlist):

                    if plot_with_background_sub == True:

                        bkgrd = self.bg[i] - self.bg[i][-1]
                        # p[pairs[0]] = axs.fill_between(self.fit_results[i].userkws['x'],\
                        #                 sum([self.fit_results[i].eval_components()[comp] for comp in pairs[1]]) + \
                        #                 bkgrd + offset, bkgrd + offset,\
                        #                 color = hue[pairs[1][0]], alpha=0.3)

                        p[pairs[0]] = axs.fill_between(self.esub,\
                                        sum([self.mod.eval_components(params = self.fit_results[i].params,x=self.esub)[comp] for comp in pairs[1]]) + \
                                        bkgrd + offset, bkgrd + offset,\
                                        color = hue[pairs[1][0]], alpha=0.3)
                    else:
                        # p[pairs[0]] = axs.fill_between(self.fit_results[i].userkws['x'],\
                        #                 sum([self.fit_results[i].eval_components()[comp] for comp in pairs[1]]) + offset, offset,\
                        #                 color = hue[pairs[1][0]], alpha=0.3)
                        p[pairs[0]] = axs.fill_between(self.esub,\
                                        sum([self.mod.eval_components(params = self.fit_results[i].params,x = self.esub)[comp] for comp in pairs[1]]) + offset, offset,\
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
                axs[i[0]].plot(self.esub, self.isub[i[1]] + offset,'o')
                axs[i[0]].plot(self.esub, self.mod.eval(params = self.fit_results[i[0]].params,x = self.esub) + offset)

                p = [[] for i in range(len(self.pairlist))]
                fit_legend = [element_text[element[0]] for element in self.pairlist]

                for pairs in enumerate(self.pairlist):

                    if plot_with_background_sub == True:
                        bkgrd = self.bg[i] - self.bg[-1]
                        p[pairs[0]] = axs[i[0]].fill_between(self.esub,\
                                        sum([self.mod.eval_components(params = self.fit_results[i[0]].params,x=self.esub)[comp] for comp in pairs[1]]) + \
                                        bkgrd + offset, y2 = bkgrd + offset*np.ones(len(self.esub)),\
                                        color = hue[pairs[1][0]], alpha=0.3)
                    else:
                        p[pairs[0]] = axs[i[0]].fill_between(self.esub,\
                                                    sum([self.mod.eval_components(params = self.fit_results[i[0]].params,x=self.esub)[comp] for comp in pairs[1]]) + offset\
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



    def plot(self,spec_type = 'raw',specific_points = None, plot_with_background_sub = False, offset = 0):
        
        hue = cfg.spectra_colors()[self.orbital]

        if specific_points == None:
            points = range(len(self.I))
        else:
            points = specific_points

        fig, ax = plt.subplots()

        if spec_type == 'raw':
            for i in points:
                ax.plot(self.E,self.I[i]+i*offset,color = hue)

        elif spec_type == 'sub':
            for i in points:
                ax.plot(self.esub,self.isub[i]+i*offset,color = hue)


        ax.set_xlabel('Binding Energy (eV)',fontsize=24)
        ax.set_ylabel('Counts/sec',fontsize=24)

        if spec_type == 'raw':
            ax.set_xlim(np.max(self.E),np.min(self.E))
        elif spec_type == 'sub':
            ax.set_xlim(np.max(self.esub),np.min(self.esub))
        ax.tick_params(labelsize=20)

        fig.tight_layout()
        
        
        return fig, ax
























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