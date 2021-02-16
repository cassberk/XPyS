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
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

from tqdm import tqdm_notebook as tqdm
import pickle
import lmfit as lm

import sys
import xps_peakfit
from xps_peakfit import bkgrds as backsub
from xps_peakfit.helper_functions import *
from xps_peakfit.gui_element_dicts import *
import xps_peakfit.config as cfg
import xps_peakfit.VAMAS
import xps_peakfit.autofit.autofit
import os
import glob
import h5py

from sklearn.decomposition import PCA
from scipy.stats import pearsonr

import matplotlib.patches as mpatches

class SonnySpectra:

    def __init__(self,spectra=None):
        self.info = []

    def load_spectra_objs(self,data_paths,spectra_name,exclude_list = []):

        datadict = {}
        for file in data_paths:
            if not any([exclude in file for exclude in exclude_list]):
                fpath = os.path.join(cfg.datarepo['stoqd'],file)
                f = h5py.File(fpath,'r')
                exps = [k for k in f.keys()]
                print(file.split('/')[-1],exps[0])
                datadict[file.split('/')[-1].split('.')[0]] = xps_peakfit.io.load_spectra(filepath = fpath,experiment_name = exps[0],spec = spectra_name)
                f.close()

        self.spectra_objects = datadict

    def bgsuball(self,subpars):
        for sample in self.spectra_objects:
            sample.bg_sub(crop_details=subpars)

    def build_spectra_matrix(self,emin,emax,spectra_type = 'raw',subpars = None):

        if spectra_type == 'sub':
            self.bgsuball(subpars = subpars)
            for name,obj in self.spectra_objects.items():
                try:
                    spectra = np.append(spectra,obj.isub,axis = 0)
                except:
                    spectra = obj.isub

        elif spectra_type =='raw':

            for name,obj in self.spectra_objects.items():

                if spectra_type == 'raw':
                    Ei = sorted([index_of(obj.E,emin), index_of(obj.E,emax)])
                    print(name, len(obj.E[Ei[0]:Ei[1]]))
                    energy = obj.E[Ei[0]:Ei[1]]
                    try:
                        spectra = np.append(spectra,obj.I[:,Ei[0]:Ei[1]],axis = 0)
                    except:
                        spectra = obj.I[:,Ei[0]:Ei[1]]

        self.energy = energy
        self.spectra = spectra
        self._spectra = spectra

    def build_dataframe(self,spectra_dict=None,target = None,targetname = 'target',include_fit_params = True, spectra_type = 'sub'):
        
        self.targetname = targetname
        if spectra_dict is None:
            spectra_dict = self.spectra_objects

            
        i = 0
        for sample in self.spectra_objects.items():

            print(sample[0],sample[1])
            
            if include_fit_params:
                dd = {key: [sample[1].fit_results[i].params.valuesdict()[key] \
                            for i in range(len( sample[1].fit_results))] \
                    for key in  sample[1].fit_results[0].params.valuesdict().keys()}       

                dftemp = pd.DataFrame(dd)
                dftemp['sample'] = [sample[0]]*len(sample[1].fit_results)
            if target != None:
                if type(target) == dict:
                    dftemp[self.targetname] = [target[sample[0]]]*len(sample[1].fit_results)
            
            try:
                param_df = param_df.append(dftemp)
                print('try',i)
                i+=1
            except:
                param_df = dftemp
                print('except',i)
                i+=1

            clear_output(wait = True)

            

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

    def plot_spec(self,offset = 0,avg = False):
        if not avg:
            for i in range(len(self.spectra)):
                plt.plot(self.energy,self.spectra[i,:]+i*offset)
        if avg:
            plt.plot(self.energy,self.spectra.mean(axis=0))

    def update_info(self,message):
        if self.info != []:
            if self.info[-1] == message:
                return
            else:
                self.info.append(message)
        else:
            self.info.append(message)


    def peak_tracker(self,peak_pos,energy= None,spectra_matrix=None,plotflag = True):
        if energy ==None:
            energy = self.energy
        if spectra_matrix == None:
            spectra_matrix = self.spectra
        
        cen = np.empty(len(spectra_matrix))
        amp = np.empty(len(spectra_matrix))

        for i in range(len(spectra_matrix)):
            amp[i],cen[i] = guess_from_data(energy,spectra_matrix[i],negative = None,peakpos = peak_pos)

        self.peaktrack = cen-peak_pos
        if plotflag:
            plt.plot(self.peaktrack,'o-')


    def pad_or_truncate(self,some_list, desired_len):
        return [0]*(desired_len - len(some_list)) + list(some_list)


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

    def plotpca2D(self,x,y):
        fig,(ax1,ax2) = plt.subplots(1,2,figsize = (18,6))

        number_of_plots=len(self.spectra_objects.keys())
        colormap = plt.cm.nipy_spectral
        colors = [colormap(i) for i in np.linspace(0, 1,number_of_plots)]
        ax1.set_prop_cycle('color', colors)
        ax2.set_prop_cycle('color', colors)

        for s in self.spectra_objects.keys():
            # print(s)
            if self.df[self.df['sample']==s][self.targetname].drop_duplicates().values[0] == 0:
                mark = '.'
            else:
                mark = 'x'
            ax1.plot(self.df[self.df['sample']==s][x].values,self.df[self.df['sample']==s][y].values,mark,markersize = 10)
            
            if self.df[self.df['sample']==s][self.targetname].drop_duplicates().values[0] == 0:
                mark = 'bo'
            else:
                mark = 'rx'
            ax2.plot(self.df[self.df['sample']==s][x].values,self.df[self.df['sample']==s][y].values,mark,markersize = 10)

        for ax in [ax1, ax2]:
            ax.set_title('Principal Components')
            ax.set_xlabel(x)
            ax.set_ylabel(y)
            ax.tick_params('x',labelrotation=80)

            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +\
                ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(20)

        ax1.legend(list(self.spectra_objects.keys()),bbox_to_anchor=(1.05, 1.2), loc='upper left',fontsize = 18)

        red_patch = mpatches.Patch(color='red', label=self.targetname)
        blue_patch = mpatches.Patch(color='blue', label='No '+self.targetname)

        ax2.legend(handles=[blue_patch,red_patch],bbox_to_anchor=(1.05, 0.5), loc='upper left',fontsize = 18)
        fig.tight_layout()

    def plotpca3D(self,X='P1', Y='P2', Z='P3', label = 'samples'):
        fig = plt.figure(figsize = (20,8))
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        
        number_of_plots=len(self.spectra_objects)
        colormap = plt.cm.nipy_spectral
        colors = [colormap(i) for i in np.linspace(0, 1,number_of_plots)]
        ax.set_prop_cycle('color', colors)
        
        for s in self.spectra_objects.keys():
            if label == 'samples':
                if self.df[self.df['sample']==s][self.targetname].drop_duplicates().values[0] == 0:
                    mark = '.'
                else:
                    mark = 'x'
                ax.plot(self.df[self.df['sample']==s][X].values,self.df[self.df['sample']==s][Y].values,self.df[self.df['sample']==s][Z].values,mark,markersize = 10)
                ax.legend(list(self.spectra_objects.keys()),bbox_to_anchor=(1, 1.1), loc='upper left',fontsize = 18)
                

            elif label =='target':
                if self.df[self.df['sample']==s][self.targetname].drop_duplicates().values[0] == 0:
                    mark = 'bo'
                else:
                    mark = 'rx'
                ax.plot(self.df[self.df['sample']==s][X].values,self.df[self.df['sample']==s][Y].values,self.df[self.df['sample']==s][Z].values,mark,markersize = 10)

                red_patch = mpatches.Patch(color='red', label='BOE')
                blue_patch = mpatches.Patch(color='blue', label='No BOE')

                ax.legend(handles=[blue_patch,red_patch],bbox_to_anchor=(1.05, 0.5), loc='upper left',fontsize = 18)
            
        fig.tight_layout()
        
        
        ax.set_xlabel(X,fontsize = 16)
        ax.set_ylabel(Y,fontsize = 16)
        ax.set_zlabel(Z,fontsize = 16)
        # for ax in [ax1,ax2,ax3,ax4,ax5,ax6]:
        #     ax.set_xlabel('P2')
        #     if (ax ==ax1) or (ax ==ax4):
        #         ax.set_ylabel('P3')
        #     for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
        #                 ax.get_xticklabels() + ax.get_yticklabels()):
        #         item.set_fontsize(16)

    def correlate(self,par,plotflag = True):
        
        print(' '.join(self.info))
        pmax = self.df.corr()[par].iloc[0:self.spectra.shape[1]].max()
        emax = self.df.corr()[par].iloc[0:self.spectra.shape[1]].idxmax()
        print('maximum correlation of',pmax,'at',emax)
        
        if plotflag:
            fig,axs = plt.subplots(1,2,figsize = (12,4))
            axs[0].plot(self.energy,self.df.corr()[par].iloc[0:self.spectra.shape[1]].values)
            axs[0].plot(emax,pmax,'x',markersize = 15)
            axs[0].set_xlabel('B.E. (eV)',fontsize = 14)

            t = self.df.corr()[par].iloc[0:self.spectra.shape[1]].values
            sc = axs[1].scatter(self.energy,self.spectra.mean(axis=0),c=t,cmap=cm.bwr, vmin=-1, vmax=1, s=100)
            axs[1].set_xlabel('B.E. (eV)',fontsize = 14)
            fig.colorbar(sc,ax=axs[1])
            
        return pmax,emax

    def par_corr(self,Xs,Ys):
        fig, axs = plt.subplots(len(Ys),len(Xs),figsize = (4*len(Xs),4*len(Ys)))
        ylabels = []
        corrmat = np.empty([len(Ys),len(Xs)])
        ax = 0
        for i in range(len(Ys)):
            j =0

            if type(Ys[i]) == float:
                axs[i][j].set_ylabel(np.round(Ys[i],2),fontsize = 26)
            else:
                axs[i][j].set_ylabel(Ys[i],fontsize = 26)

            for j in range(len(Xs)):

                try:
                    axs[i][j].plot(self.df[Xs[j]].values,self.df[Ys[i]].values,'o')
                    corrmat[i][j], _ = pearsonr(self.df[Xs[j]].values, self.df[Ys[i]].values) 

                    axs[i][j].set_title('Pearsons correlation: %.3f' % corrmat[i][j],fontsize = 18)
                    axs[i][j].set_xlabel(Xs[j],fontsize = 26)
                    axs[i][j].set_xticklabels('')
                    axs[i][j].set_yticklabels('')


                except:
                    pass

        # for ax in axs:
        #     for item in ([ax.yaxis.label]):
        #         item.set_fontsize(26)

        colmax = np.argmax(corrmat,axis=0)
        for i in enumerate(colmax):
            axs[i[1]][i[0]].set_title('Pearsons correlation: %.3f' % corrmat[i[1]][i[0]],color = 'darkred',fontsize = 18)

        fig.tight_layout()


    def find_linautofit(self,Xs,Ys):
        autofitparlist = []
        fig, axs = plt.subplots(1,len(Xs),figsize = (4*len(Xs),4))

        for i in range(len(Ys)):
            # print(Xs[i])
            # print(Ys[i])
            x = np.array(self.df[Xs[i]])
            y = np.array(self.df[Ys[i]])
            m = np.linalg.lstsq(x.reshape(-1,1), y,rcond = None)[0][0]
            autofitparlist.append(' '.join([Ys[i],'lin',str(np.round(Xs[i],1)),str(m)]))

            axs[i].plot(x, y, 'o')
            axs[i].plot(x, m*x)
            axs[i].set_xlabel(str(np.round(Xs[i],2))+' vs '+str(Ys[i]),fontsize = 14)
        
        return autofitparlist