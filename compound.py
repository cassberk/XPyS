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
import XPyS
import XPyS.config as cfg
import XPyS.models.models as xpsmodels
from XPyS import bkgrds as backsub
from XPyS.helper_functions import *
from XPyS.gui_element_dicts import *

import XPyS.VAMAS
import XPyS.autofit.autofit
import os
import glob
from IPython import embed as shell

class CompoundSpectra:
    
    def __init__(self,spectra_list):
        """Class for fitting an atomic compound accross multiple spectra

        This class works a lot like the spectra class. The parameters are a single parameters
        object comprising all the parameters from the individual spectra. The compound object
        holds the spectra objects of interest. Although, they are also held by the sample
        object.

        Parameters
        ----------
        spectra_list: list
            A list of all of the spectra objects to be combined into a compound object

        Notes
        -----


        Examples
        --------


        """
        self.rsf = cfg.avantage_sensitivity_factors()
        self.element_scans = [spec.orbital for spec in spectra_list]
        
        self.params = lm.Parameters()
        for spec in spectra_list:
            
            if not hasattr(spec,'params'):
                raise AttributeError(f"{spec.orbital} does not have params attribute")
                
            self.add_spectra_params(spec)
            
        self.n_scans = len(self.__dict__[self.element_scans[0]].isub)
        
    def add_spectra_params(self,spectra):
        """Function for adding the parameters into a single parameter object as well
        as adding the spectra object to the compound object
        """
        self.params.update(spectra.params)
        self.__dict__[spectra.orbital] = spectra
        
        
        
        
    def link(self,par1,par2,atomic_ratio):
        """Set the peak area of spectra components in different spectra

        Parameters
        ----------
        par1: dict
            par1 is the parent parameter that is allowed to vary in the fit. They dict key
            is the orbital and the value is the spectral component
            ex: par1 = {N1s: N1_amplitude}
            
        par2: dict
            par1 is the child parameter that is set by the expression and not allowed to 
            vary in the fit. They key is the orbital and the value is the spectral component
        ratio: int,float
        
            atomic ratio of par1:par2 
           

        Notes
        -----
        The solution comes from:
        
        (area1/rsf1)/(area2/rsf2) = atomic_ratio
        
        => area2 = area1*(rsf2/rsf1)*(1/atomic_ratio)
        
        if par1 or par2 is a doublet then both peak areas must be taken into account. Only the j+1/2
        orbital is fit while the j-1/2 peak is set to the area determined by the orbital:
        
        2p:
            j = 1/2 and 3/2 with peak area ratio of 1:2
            
            => area = (par+ 0.5*par) = (3/2)*par
        3d:
            j = 3/2 and 5/2 with peak area ratio of 2:3
            
            => area = (par+ (2/3)*par) = (5/3)*par

        Examples
        --------
        This example links the Ti2p and the N1s to form a TiN compound. It also links the 
        O1s and the Ti2p to form TiO2

        compound_example = CompoundSpectra([sample.N1s,sample.Ti2p,sample.O1s])
        compound_example.link({'N1s':'N2_amplitude'},{'Ti2p':'TiN_32_amplitude'},1)
        compound_example.link({'N1s':'N3_amplitude'},{'Ti2p':'TiN_shk32_amplitude'},1)
        compound_example.link({'O1s':'O1_amplitude'},{'Ti2p':'TiO2_32_amplitude'},2) 
        """
         
        
        orb1 = list(par1.keys())[0]
        orb2 = list(par2.keys())[0]
        
        comp1 = list(par1.values())[0]
        comp2 = list(par2.values())[0]
        
        
        if not comp1 in list(self.params.keys()):
            raise NameError(f"{comp1} is not in the parameters")
        if not comp2 in list(self.params.keys()):
            raise NameError(f"{comp2} is not in the parameters")
        
        
        comp1_rsf = comp1+'_rsf'
        comp2_rsf = comp2+'_rsf'
        rsf_ratio = f"({comp2_rsf}/{comp1_rsf})"
        
        if '1s' in orb1:
            orb_ratio1 = '1'
        elif '2p' in orb1:
            orb_ratio1 = '(3/2)'
        elif '3d' in orb1:
            orb_ratio1 = '(5/3)'
        
        if '1s' in orb2:
            orb_ratio2 = '1'
        elif '2p' in orb2:
            orb_ratio2 = '(2/3)'
        elif '3d' in orb2:
            orb_ratio2 = '(3/5)'
            
        
        self.params._asteval.symtable[comp2_rsf] = self.rsf[orb2]
        self.params._asteval.symtable[comp1_rsf] = self.rsf[orb1]
        self.params._asteval.symtable['atomic_ratio'] = atomic_ratio
        
        expression = f"{orb_ratio2}*{orb_ratio1}*{rsf_ratio}*(1/{atomic_ratio})*{comp1}"
       
        self.params[comp2].expr = expression
        
        print(self.params[comp2].expr)
        
        
    def fit(self,fit_method = 'powell',specific_points = None,plotflag = True, track = False, fit_in_reverse = False,update_with_prev_pars = False, autofit = False):
        """Function to fit the spectra to the model held by the spectra object


        Parameters
        ----------
        fit_method: str
            Fitting Method. See the documentation of different fitting methods in lmfit documentation
        specific_points: list
            list of the data to fit
        plotflag: bool
            Wether or not to plot the fits
        track: bool
            Whether or not to track the progress using a progress bar
        fit_in_reverse: bool
            Fit the spectra in reverse. This will also fit the specific_points in reverse if they are speciallized
            This is useful if you are using update_with_prev_pars and want to fit a depth profile where some spectra 
            are prevalent towards the end and others at the beginning.
        update_with_prev_pars: bool
            Update the params object with the fit_results[i].params from the previous fit. This is useful for fitting lots
            of spectra since if spectra are slowly changing eventually the params will not be a good starting point.
        autofit: bool
            Whether or not to use autofit on a specific spectra. Need to set it up in the autofit module.

        Notes
        -----
        

        Examples
        --------


        """     
        
        if not hasattr(self,"fit_results"):
            self.fit_results = [[] for i in range(self.n_scans)]           
            
        if specific_points is None:
            specific_points = np.arange(len(self.I))

        if fit_in_reverse == True:
            specific_points = specific_points[::-1]

        for i in specific_points:
            
            data = np.hstack(tuple([self.__dict__[spec].isub[i] for spec in self.element_scans]))
            
            fitter = lm.Minimizer(self.fcn2min,self.params,fcn_args = (data,))
            self.fit_results[i] = fitter.minimize(method=fit_method)
        
        self._copy_to_spectra()

        if plotflag:
            self.plot_fitresults(specific_points = specific_points)

    def fcn2min(self,params, data):
        """function to be minimized"""
        i = 0
        for spec in self.element_scans:
            if i == 0:
                data_pred = self.__dict__[spec].mod.eval(params,x = self.__dict__[spec].esub)
            else:
                data_pred = np.append(data_pred,self.__dict__[spec].mod.eval(params,x = self.__dict__[spec].esub))
            i+=1
        return data_pred - data
    

    def _copy_to_spectra(self):
        """copy thefit results and the parameters to the spectra objects"""
        for spec in self.element_scans:
            self.__dict__[spec].fit_results = dc(self.fit_results)
            self.__dict__[spec].params = self.params.copy()
     
    
    def plot_fitresults(self,specific_points = None, plot_with_background_sub = False,ref_lines = False,colors = None,infig = None, inaxs = None, offset = 0):


        if colors is None:
            hue = element_color
        else:
            hue = colors

        if specific_points is None:

            subplotrows =  len([j for j,x in enumerate(self.fit_results) if x])
            plot_idx = [j for j,x in enumerate(self.fit_results) if x]
        else:
            subplotrows =  len(specific_points)
            plot_idx = specific_points


        if inaxs is None:
            fig,axs = plt.subplots(subplotrows,len(self.element_scans), figsize = (18,subplotrows*5))
        else:
            fig = infig
            axs = inaxs


        axs = axs.ravel()

        i=0
        for scan_num in plot_idx:
            
            for spec in self.element_scans:
                spectra = self.__dict__[spec]

                axs[i].plot(spectra.esub, spectra.isub[scan_num] + offset,'o')
                axs[i].plot(spectra.esub, spectra.mod.eval(self.fit_results[scan_num].params, x =spectra.esub) + offset)

                p = [[] for i in range(len(spectra.pairlist))]
                fit_legend = [element_text[element[0]] for element in spectra.pairlist]

                for pairs in enumerate(spectra.pairlist):

                    if plot_with_background_sub == True:
                        bkgrd = self.bg[scan_num] - self.bg[-1]
                        p[pairs[0]] = axs[i].fill_between(spectra.esub,\
                                        sum([spectra.mod.eval_components(params = self.fit_results[scan_num].params,x=spectra.esub)[comp] for comp in pairs[1]]) + \
                                        bkgrd + offset, y2 = bkgrd + offset*np.ones(len(spectra.esub)),\
                                        color = hue[pairs[1][0]], alpha=0.3)
                    else:
                        p[pairs[0]] = axs[i].fill_between(spectra.esub,\
                                                    sum([spectra.mod.eval_components(params = self.fit_results[scan_num].params,x=spectra.esub)[comp] for comp in pairs[1]]) + \
                                                    offset,y2 = offset*np.ones(len(spectra.esub)),\
                                                    color = hue[pairs[1][0]], alpha=0.3)

                    if ref_lines == True:
                        axs[i].axvline(x = element_refpos[pairs[1][0]],color = hue[pairs[1][0]])

                axs[i].set_xlabel('Binding Energy (eV)',fontsize=30)
                axs[i].set_xlim(np.max(spectra.esub),np.min(spectra.esub))
                axs[i].tick_params(labelsize=20)

                i+=1

        fig.legend(p,fit_legend,bbox_to_anchor=(0.5, 1, 0.5, 0.5), loc='lower center',fontsize=30)

        fig.tight_layout()

        return fig, axs
