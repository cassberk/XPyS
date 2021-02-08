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
# sys.path.append("/Volumes/GoogleDrive/My Drive/XPS/XPS_Library")
import xps_peakfit
# from xps.io import loadmodel
from xps_peakfit import bkgrds as backsub
# from xps import bkgrds as background_sub
from xps_peakfit.helper_functions import *
from xps_peakfit.gui_element_dicts import *
# from xps_peakfit.auto_fitting import *

import xps_peakfit.VAMAS
import xps_peakfit.autofit.autofit
import os
import glob

from sklearn.decomposition import PCA
import matplotlib.patches as mpatches

class SonnySpectra:

    def __init__(self,spectra=None):
        self.info = []

    def load_spectra_objs(self,data_paths,spectra_name,exclude_list = []):

        datadict = {}
        f = open(data_paths, "r")
        for line in f.readlines():
            # print()
            if line.split(',')[0].split('/')[-3] not in exclude_list:
                datadict[line.split(',')[0].split('/')[-3]] = xps_peakfit.io.load_spectra(filepath = line.split(',')[0],experiment_name = line.split(',')[1].split('\n')[0],spec = spectra_name)
        f.close()
        self.spectra_objects = datadict

    def build_spectra_matrix(self,spectra_dict=None,new_bg_sub = False,target = False):
        
        if spectra_dict is None:
            spectra_dict = self.spectra_objects

            
        sample_list = []
        i = 0
        for sample in self.spectra_objects.items():

            print(sample[0],sample[1])
            sample_list.append(sample[0])
            
            if new_bg_sub:
                sample[1].bg_sub(crop_details=new_bg_sub)

            dd = {key: [sample[1].fit_results[i].params.valuesdict()[key] \
                        for i in range(len( sample[1].fit_results))] \
                for key in  sample[1].fit_results[0].params.valuesdict().keys()}       

            dftemp = pd.DataFrame(dd)
            dftemp['sample'] = [sample[0]]*len(sample[1].fit_results)
            if target:
                dftemp['boe'] = [target[sample[0]]]*len(sample[1].fit_results)
            
            if i ==0:
                y = np.array(sample[1].isub)
                x = np.array(sample[1].esub)
                param_df = dftemp
                i+=1
            else:
                y = np.append(y,np.array(sample[1].isub),axis = 0)
                param_df = param_df.append(dftemp)
                i+=1
            clear_output(wait = True)

            
        self.energy = x
        self.spectra = y
        self._spectra = y
        # self.df_spectra = pd.DataFrame(self.spectra,columns = self.energy)
        self.df_params = param_df.reset_index(drop = True)
        self.df = pd.DataFrame(self.spectra,columns = self.energy).join(self.df_params)
        self._df = pd.DataFrame(self.spectra,columns = self.energy).join(self.df_params)

    def reset(self):
        self.spectra = dc(self._spectra)
        self.df = dc(self._df)
        self.info = []

    def normalize(self):
        _yN = np.empty(self.spectra.shape)

        for i in range(len(_yN)):
            _yN[i,:] = self.spectra[i,:]/np.trapz(self.spectra[i,:])

        self.spectra = _yN
        self.update_info('Normalized')

    def update_info(self,message):

        if self.info != []:
            if self.info[-1] == message:
                return
            else:
                self.info.append(message)

        else:
            self.info.append(message)


    def peak_tracker(self,peak_pos,energy= None,spectra_matrix=None):
        if energy ==None:
            energy = self.energy
        if spectra_matrix == None:
            spectra_matrix = self.spectra
        
        cen = np.empty(len(spectra_matrix))
        amp = np.empty(len(spectra_matrix))

        for i in range(len(spectra_matrix)):
            amp[i],cen[i] = guess_from_data(energy,spectra_matrix[i],negative = None,peakpos = peak_pos)

        self.peaktrack = cen-peak_pos


    def pad_or_truncate(self,some_list, target_len):
        return [0]*(target_len - len(some_list)) + list(some_list)


    def align_peaks(self,peak_pos,energy=None,spec_set = None,plotflag = True):

        fig = plt.figure()
        if energy == None:
            energy = self.energy

        spectra_matrix = self.spectra

        cen = np.empty(len(spectra_matrix))
        amp = np.empty(len(spectra_matrix))
        mv_spec = np.empty([len(spectra_matrix),len(energy)])


        for i in range(len(spectra_matrix)):

            amp[i],cen[i] = guess_from_data(energy,spectra_matrix[i],negative = None,peakpos = peak_pos)

            mv_ev = np.round(cen[i] - peak_pos,2)

            mv_pts = np.int(np.round((cen[i] - peak_pos)*(len(energy)/(energy[0] - energy[-1]))))

            if mv_pts ==0:
                mv_spec[i] = spectra_matrix[i]

            else:
                mv_spec[i] = self.pad_or_truncate(spectra_matrix[i][:-1*mv_pts],len(energy))

        if plotflag:
            for i in range(len(spectra_matrix)):
                plt.plot(mv_spec[i])

            plt.axvline(index_of(energy,peak_pos))

        self.spectra_aligned_pos = peak_pos
        self.spectra = mv_spec

        self.df = pd.DataFrame(self.spectra,columns = self.energy).join(self.df_params)
        self.update_info('adjusted to '+str(peak_pos))

    def pca(self,n_comps = 3):
        pca = PCA(n_components=n_comps)

        """PCA of raw signal"""
        X_r = pca.fit(self.spectra)
        X_tr = pca.fit_transform(self.spectra)

        if self.info != []:
            print(' '.join(self.info))

        print('explained variance ratio: %s'
            % str(pca.explained_variance_ratio_ *100 ) )
        print('cumulative variance: %s'
            % str(np.cumsum(pca.explained_variance_ratio_ *100) ) ) 

        for pc in enumerate(['P1','P2','P3']):
            self.df[pc[1]] = pd.Series(X_tr[:,pc[0]], index = self.df.index)


        fig,ax = plt.subplots()

        for i in range(3):
            ax.plot(X_r.components_[i,:])
            
        ax.legend(['PC1','PC2','PC3'],bbox_to_anchor=(1.05, 1), loc='upper left',fontsize = 18)

        ax.set_title('Raw Data',fontsize = 18)
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                    ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(18)

    # def plot_pca(self,x,y):
    #     fig1, ((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2,3,figsize = (24,12))

    #     number_of_plots=len(self.spectra_objects.keys())
    #     colormap = plt.cm.nipy_spectral
    #     colors = [colormap(i) for i in np.linspace(0, 1,number_of_plots)]
    #     ax1.set_prop_cycle('color', colors)
    #     ax2.set_prop_cycle('color', colors)
    #     ax3.set_prop_cycle('color', colors)

    #     for s in self.spectra_objects.keys():
    #         # print(s)
    #         if self.df_full[self.df_full['sample']==s]['boe'].drop_duplicates().values[0] == 0:
    #             mark = '.'
    #         else:
    #             mark = 'x'
    #         ax1.plot(self.df_full[self.df_full['sample']==s][x+'_raw'].values,self.df_full[self.df_full['sample']==s][y+'_raw'].values,mark,markersize = 10)
    #         ax2.plot(self.df_aligned_full[self.df_aligned_full['sample']==s][x+'_adj'].values,self.df_aligned_full[self.df_aligned_full['sample']==s][y+'_adj'].values,mark,markersize = 10)
    #         ax3.plot(self.df_aligned_full_N[self.df_aligned_full_N['sample']==s][x+'_adj_N'].values,self.df_aligned_full_N[self.df_aligned_full_N['sample']==s][y+'_adj_N'].values,mark,markersize = 10)
            
    #         if self.df_full[self.df_full['sample']==s]['boe'].drop_duplicates().values[0] == 0:
    #             mark = 'bo'
    #         else:
    #             mark = 'rx'
    #         ax4.plot(self.df_full[self.df_full['sample']==s][x+'_raw'].values,self.df_full[self.df_full['sample']==s][y+'_raw'].values,mark,markersize = 10)
    #         ax5.plot(self.df_aligned_full[self.df_aligned_full['sample']==s][x+'_adj'].values,self.df_aligned_full[self.df_aligned_full['sample']==s][y+'_adj'].values,mark,markersize = 10)
    #         ax6.plot(self.df_aligned_full_N[self.df_aligned_full_N['sample']==s][x+'_adj_N'].values,self.df_aligned_full_N[self.df_aligned_full_N['sample']==s][y+'_adj_N'].values,mark,markersize = 10)

    #     ax1.set_title('Raw')
    #     ax2.set_title('Adjusted')
    #     ax3.set_title('Adjusted Normalized')

    #     for ax in [ax1,ax2,ax3,ax4,ax5,ax6]:
    #         ax.set_xlabel(x)
    #         ax.tick_params('x',labelrotation=80)
    #         if (ax ==ax1) or (ax ==ax4):
    #             ax.set_ylabel(y)
    #         for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
    #                     ax.get_xticklabels() + ax.get_yticklabels()):
    #             item.set_fontsize(20)

    #     ax3.legend(list(self.spectra_objects.keys()),bbox_to_anchor=(1.05, 1.2), loc='upper left',fontsize = 18)

    #     red_patch = mpatches.Patch(color='red', label='BOE')
    #     blue_patch = mpatches.Patch(color='blue', label='No BOE')

    #     ax6.legend(handles=[blue_patch,red_patch],bbox_to_anchor=(1.05, 0.5), loc='upper left',fontsize = 18)
    #     fig1.tight_layout()


    def correlate(self,par,plotflag = True):
        fig = plt.figure()
        print(' '.join(self.info))
        pmax = self.df.corr()[par].iloc[0:self.spectra.shape[1]].max()
        emax = self.df.corr()[par].iloc[0:self.spectra.shape[1]].idxmax()
        print('maximum correlation of',pmax,'at',emax)
        
        if plotflag:
            self.df.corr()[par].iloc[0:self.spectra.shape[1]].plot()
            plt.plot(emax,pmax,'x',markersize = 15)
        return pmax,emax