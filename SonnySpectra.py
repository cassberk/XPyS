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
        print('none')

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

        y_norm = np.empty(y.shape)

        for i in range(len(y)):
            y_norm[i,:] = y[i,:]/np.trapz(y[i,:])
            
        self.energy = x
        self.spectra = y
        self.spectra_N = y_norm
        # self.df_spectra = pd.DataFrame(self.spectra,columns = self.energy)
        self.df_params = param_df.reset_index(drop = True)
        self.df_full = pd.DataFrame(self.spectra,columns = self.energy).join(self.df_params)


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
        if energy == None:
            energy = self.energy

        if spec_set == None:
            spectra_matrix = self.spectra
        if spec_set == 'Norm':
            spectra_matrix = self.spectra_N

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
        self.spectra_aligned = mv_spec
        if spec_set =='Norm':
            self.spectra_aligned_N = mv_spec

        if spec_set == None:
            self.df_aligned_full = pd.DataFrame(self.spectra_aligned,columns = self.energy).join(self.df_params)
        if spec_set == 'Norm':
            self.df_aligned_full_N = pd.DataFrame(self.spectra_aligned,columns = self.energy).join(self.df_params)


    def pca(self):
        pca_raw = PCA(n_components=3)
        pca_adj = PCA(n_components=3)
        pca_adj_N = PCA(n_components=3)

        """PCA of raw signal"""
        X_r = pca_raw.fit(self.spectra)
        X_tr = pca_raw.fit_transform(self.spectra)

        """PCA of signal adjusted to nb peak"""
        X_adj = pca_adj.fit(self.spectra_aligned)
        X_tr_adj = pca_adj.fit_transform(self.spectra_aligned)

        """PCA of normalized signal adjusted to nb peak"""
        X_adj_N = pca_adj_N.fit(self.spectra_aligned_N)
        X_tr_adj_N = pca_adj_N.fit_transform(self.spectra_aligned_N)


        print('raw explained variance ratio: %s'
            % str(pca_raw.explained_variance_ratio_ *100 ) )
        print('raw cumulative variance: %s'
            % str(np.cumsum(pca_raw.explained_variance_ratio_ *100) ) ) 
        print('')
        print('adjusted explained variance ratio: %s'
            % str(pca_adj.explained_variance_ratio_ *100 ) )
        print('adjusted cumulative variance: %s'
            % str(np.cumsum(pca_adj.explained_variance_ratio_ *100) ) )
        print('')
        print('adjusted normalized explained variance ratio: %s'
            % str(pca_adj_N.explained_variance_ratio_ *100 ) )
        print('adjusted normalized cumulative variance: %s'
            % str(np.cumsum(pca_adj_N.explained_variance_ratio_ *100) ) ) 

        for pc in enumerate(['P1_raw','P2_raw','P3_raw']):
            self.df_full[pc[1]] = pd.Series(X_tr[:,pc[0]], index = self.df_full.index)
            
        for pc in enumerate(['P1_adj','P2_adj','P3_adj']):
            self.df_aligned_full[pc[1]] = pd.Series(X_tr_adj[:,pc[0]], index = self.df_full.index)
            
        for pc in enumerate(['P1_adj_N','P2_adj_N','P3_adj_N']):
            self.df_aligned_full_N[pc[1]] = pd.Series(X_tr_adj_N[:,pc[0]], index = self.df_full.index)


        fig = plt.figure(figsize = (20,6))
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
        for i in range(3):
            ax1.plot(X_r.components_[i,:])
            ax2.plot(X_adj.components_[i,:])
            ax3.plot(X_adj_N.components_[i,:])
            
        ax3.legend(['PC1','PC2','PC3'],bbox_to_anchor=(1.05, 1), loc='upper left',fontsize = 18)

        ax1.set_title('Raw Data',fontsize = 18),ax2.set_title('Adjusted to peakpos',fontsize = 18), ax3.set_title('Adjusted to Nb and Normalized',fontsize = 18)
        for ax in [ax1, ax2, ax3]:
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                        ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(18)

    def plot_pca(self,x,y):
        fig1, ((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2,3,figsize = (24,12))
        # ax1 = fig1.add_subplot(131)
        # ax2 = fig1.add_subplot(132)
        # ax3 = fig1.add_subplot(133)

        number_of_plots=len(self.spectra_objects.keys())
        colormap = plt.cm.nipy_spectral
        colors = [colormap(i) for i in np.linspace(0, 1,number_of_plots)]
        ax1.set_prop_cycle('color', colors)
        ax2.set_prop_cycle('color', colors)
        ax3.set_prop_cycle('color', colors)

        for s in self.spectra_objects.keys():
            # print(s)
            if self.df_full[self.df_full['sample']==s]['boe'].drop_duplicates().values[0] == 0:
                mark = '.'
            else:
                mark = 'x'
            ax1.plot(self.df_full[self.df_full['sample']==s][x+'_raw'].values,self.df_full[self.df_full['sample']==s][y+'_raw'].values,mark,markersize = 10)
            ax2.plot(self.df_aligned_full[self.df_aligned_full['sample']==s][x+'_adj'].values,self.df_aligned_full[self.df_aligned_full['sample']==s][y+'_adj'].values,mark,markersize = 10)
            ax3.plot(self.df_aligned_full_N[self.df_aligned_full_N['sample']==s][x+'_adj_N'].values,self.df_aligned_full_N[self.df_aligned_full_N['sample']==s][y+'_adj_N'].values,mark,markersize = 10)
            
            if self.df_full[self.df_full['sample']==s]['boe'].drop_duplicates().values[0] == 0:
                mark = 'bo'
            else:
                mark = 'rx'
            ax4.plot(self.df_full[self.df_full['sample']==s][x+'_raw'].values,self.df_full[self.df_full['sample']==s][y+'_raw'].values,mark,markersize = 10)
            ax5.plot(self.df_aligned_full[self.df_aligned_full['sample']==s][x+'_adj'].values,self.df_aligned_full[self.df_aligned_full['sample']==s][y+'_adj'].values,mark,markersize = 10)
            ax6.plot(self.df_aligned_full_N[self.df_aligned_full_N['sample']==s][x+'_adj_N'].values,self.df_aligned_full_N[self.df_aligned_full_N['sample']==s][y+'_adj_N'].values,mark,markersize = 10)

        ax1.set_title('Raw')
        ax2.set_title('Adjusted')
        ax3.set_title('Adjusted Normalized')

        for ax in [ax1,ax2,ax3,ax4,ax5,ax6]:
            ax.set_xlabel(x)
            ax.tick_params('x',labelrotation=80)
            if (ax ==ax1) or (ax ==ax4):
                ax.set_ylabel(y)
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                        ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(20)

        ax3.legend(list(self.spectra_objects.keys()),bbox_to_anchor=(1.05, 1.2), loc='upper left',fontsize = 18)

        red_patch = mpatches.Patch(color='red', label='BOE')
        blue_patch = mpatches.Patch(color='blue', label='No BOE')

        ax6.legend(handles=[blue_patch,red_patch],bbox_to_anchor=(1.05, 0.5), loc='upper left',fontsize = 18)
        fig1.tight_layout()