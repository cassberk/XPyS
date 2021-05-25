# Try to load individual packages at some point
from IPython.display import display, clear_output

import pandas as pd
from copy import deepcopy as dc
import XPyS
from XPyS.helper_functions import *
from XPyS import bkgrds as backsub
from XPyS.avantage_io import load_excel
import lmfit as lm
import numpy as np
import matplotlib.pyplot as plt

import XPyS.spectra as sp
import XPyS.config as cfg
import XPyS.models as mds
import XPyS.VAMAS
import os

class HellaSamples:

    def __init__(self,pathlist,experiment_name,bkgrd_subtraction_dict = None, data_dict_idx = None, overview = True, sputter_time = None, offval=0, plotflag = True, plotspan = True,\
        plot_legend = None,normalize_subtraction = False,name = None,spectra_colors = None,load_derk = False, **kws):

        self.models = {'C1s':mds.load_model('C1s'),\
            'O1s': mds.load_model('O1s_2comp'),
            'Nb3d':mds.load_model('Nb3d'),
            'Si2p':mds.load_model('Si2p'),
            'F1s':mds.load_model('F1s_3comp')
            }
        self.samples= {}

        for samplepath in pathlist:
            samp = XPyS.io.load_sample(filepath = os.path.join(cfg.datarepo['stoqd'],samplepath),\
                                        experiment_name = experiment_name)
            self.samples[samp.sample_name] = samp
            
            clear_output(wait = True)

    
    def xps_overview(self,subpars = None, plotspan = True):
        for sample in self.samples.values():
            sample.xps_overview(subpars = subpars, plotspan = subpars,plotflag = False)
            clear_output(wait = True)


    def plotem(self,offval=0, plotspan=False, saveflag=0,filepath = '',fig = None,ax = None,figdim = None):
        if figdim is None:
            figdim = (15,int(np.ceil((len(self.samples[list(self.samples.keys())[0]].element_scans)+2)/3))*4)

        fig,ax = plt.subplots(int(np.ceil((len(self.samples[list(self.samples.keys())[0]].element_scans)+2)/3)) ,3, figsize = figdim)
        ax = ax.ravel()

        orderlist = [(orbital,np.max(self.samples[list(self.samples.keys())[0]].__dict__[orbital].E)) for orbital in self.samples[list(self.samples.keys())[0]].element_scans]
        orderlist.sort(key=lambda x:x[1])
        orderlist = ['Survey','Valence']+[spec[0] for spec in orderlist][::-1]
        ax = {orb[1]:ax[orb[0]] for orb in enumerate(orderlist)}

        done_it = {spectra:False for spectra in self.samples[list(self.samples.keys())[0]].all_scans}
        for sample in self.samples.values():
            sample.plot_all_spectra(offval=0, plotspan=plotspan, saveflag=0,filepath = '',fig =fig,ax = ax,figdim = None,done_it = done_it)

        return fig, ax

    def fitem(self,specify_spectra = None):

        if specify_spectra is None:
            spectra_list = self.samples[list(self.samples.keys())[0]].element_scans
        else:
            spectra_list = specify_spectra

        for spectra in spectra_list:
            _m, _p, _pl, _ec =  self.models[spectra]
            for sample in self.samples.values():
                if hasattr(sample,spectra):
                    # should put in a logging here incase the sample doesnt have the spectra
                    sample.__dict__[spectra].mod = _m
                    sample.__dict__[spectra].params = _p
                    sample.__dict__[spectra].pairlist = _pl
                    sample.__dict__[spectra].element_ctrl = _ec

                    sample.__dict__[spectra].fit(autofit = True,plotflag = False)
                    clear_output(wait = True)
