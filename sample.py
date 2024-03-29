import pandas as pd
from copy import deepcopy as dc
import XPyS
from .helper_functions import index_of, guess_from_data

import lmfit as lm
import numpy as np
import matplotlib.pyplot as plt

import XPyS.spectra
from XPyS.compound import CompoundSpectra
import XPyS.config as cfg
import XPyS.VAMAS
import os



class sample:

    def __init__(self,dataload_obj = None ,bkgrd_subtraction_dict = None, data_dict_idx = None, overview = True, sputter_time = None, offval=0, plotflag = True, plotspan = True,\
        plot_legend = None,normalize_subtraction = False,name = None,spectra_colors = None, **kws):
        """Class for holding the spectra objects taken on a sample

        All of the elemental spectra will be stored in this sample object. There is 
        the option of loading data from multiple different sources. Additionally, this 
        can serve as a general purpose spectra holder

        Parameters
        ----------
        dataload_obj : object holding the data
            Can be loaded from a vamas file, excel export data, or "derk object"
        bkgrd_subtraction_dict : dict, optional
            dictionary holding parameters for background subtraction.
            Sample object can be instantiated without this, but it helps to have it when doing
            backgroudn subtraction on spectra. Although it can be passed directly to bg_sub_all
            (Default is None)
        data_dict_idx: int, optional
            For Excel load, incase multiple sample sheets are found
            (default is None).
        overview : bool
            Whether or now to automatically generate plots of spectra
        offval: int, optional
            Offset value to stackscans
        plotflag: bool
            Whether or not to plot the data
            (default is True)
        plotspan: bool
            Whether or not to plot the background subtraction limits
            (default is False)
        plot_legend:
        normalize_subtraction
        name: str, optional
            option to name the sample
        spectra_colors: dict, optional
            option to specify spectra colors 



        **kws : dict, optional
            Additional keyword arguments 

        Notes
        -----


        Examples
        --------


        """
        self.sample_name = name
        self.rsf = cfg.avantage_sensitivity_factors()
        self.offval = offval
        self.plotflag = plotflag
        self.plotspan = plotspan
        self.normalize_subtraction = normalize_subtraction
        self.plot_legend = plot_legend
        self.sputter_time = sputter_time
        if bkgrd_subtraction_dict is None:
            self.bg_info = cfg.bkgrd_subtraction
        else:
            self.bg_info = bkgrd_subtraction_dict

        if spectra_colors is None:
            self.spectra_colors = cfg.spectra_colors()
        else:
            self.spectra_colors = spectra_colors


        if type(dataload_obj) == XPyS.VAMAS.VAMASExperiment:
            self.load_experiment_sample_from_vamas(dataload_obj)

        elif (type(dataload_obj) == str) or (type(dataload_obj) == dict):
            self.load_experiment_sample_from_excel(dataload_obj,name)



        if dataload_obj is None:
            overview = False
        if overview:
            self.xps_overview(bkgrd_subtraction_dict = bkgrd_subtraction_dict, plotflag = plotflag, plotspan = plotspan)


    def load_experiment_sample_from_vamas(self,vamas_obj):
        
        self.element_scans = list(dict.fromkeys([spectra for spectra in \
            [''.join(block.species_label.split()) for block in vamas_obj.blocks] if (spectra != 'V.B.') and (spectra != 'XPSSurvey')]))

        all_scans = list(dict.fromkeys([scan for scan in [''.join(block.species_label.split()) for block in vamas_obj.blocks]]))
        name_change = {'V.B.':'vb','XPSSurvey':'survey'}
        self.all_scans = [name_change.get(n, n) for n in all_scans]
        self.data_raw = vamas_obj
        
        for scan in self.all_scans:
            if scan =='vb':
                self.__dict__[scan] = XPyS.spectra.spectra()
                XPyS.spectra.spectra.load_experiment_spectra_from_vamas(self.__dict__[scan],vamas_obj,orbital = 'V.B.')
            elif scan =='survey':
                self.__dict__[scan] = XPyS.spectra.spectra()
                XPyS.spectra.spectra.load_experiment_spectra_from_vamas(self.__dict__[scan],vamas_obj,orbital = 'XPSSurvey')                
            else:
                self.__dict__[scan] = XPyS.spectra.spectra(bg_info = self.bg_info[scan])
                XPyS.spectra.spectra.load_experiment_spectra_from_vamas(self.__dict__[scan],vamas_obj,orbital = scan)

    def load_experiment_sample_from_excel(self,excel_obj,name):
            if type(excel_obj) == str:
                data_dict = load_excel(excel_obj)
                self.data_path = excel_obj

            if type(excel_obj) == dict:
                print('dict')
                data_dict = dc(excel_obj)

            if (len(list(data_dict.keys())) > 1) & (data_dict_idx == None):
                print('There are more than 1 experiment files in the excel file! Choose data_dict_idx')
                print(list(enumerate(data_dict.keys())))
                return
            elif (len(list(data_dict.keys())) == 1) & (data_dict_idx == None):
                data_dict_idx = 0

            if name == None:
                self.sample_name = list(data_dict.keys())[data_dict_idx]
            else:
                self.sample_name = name        

            self.data = data_dict[list(data_dict.keys())[data_dict_idx]]
            self.element_scans = [x for x in list(self.data.keys()) if x not in ('total area', 'Valence','XPS')]




    ####
    def xps_overview(self,subpars = None, plotflag = True, plotspan = True,analyze = True):
        """Returns an overview of the spectra that are held in sample.

       The overview plots all of the raw signals, the background subtracted
       signals and the calculated atomic percent

        Parameters
        ----------
        subpars : dict
            Optional: ability to enter custom background subtraction parameters. If
            None it will use the default in the config file.
        plotflag : bool
            option to plot or not
        plotspan : bool
            If True it will highlight on the plot the range that is backgroudn subtracted
        analyze : bool
            Choose whether or not to calculate the background subtraction and atomic percent
            or to just plot what was loaded

        Returns
        -------
        matplotlib figures : fig
           Multi-plot plots of the data.

        See Also
        --------
        :func: plot_all_spectra(), plot_all_sub(), plot_atomic_percent()

        """
        if subpars != None:
            for orb in subpars.keys():
                self.bg_info[orb] = subpars[orb]

        if analyze == True:
            self.bg_sub_all()
            self.calc_atomic_percent()
        self.plotflag = plotflag

        if self.plotflag == True:

            fig_dict = {}
            ax_dict = {}
            self.fig, self.ax= self.plot_all_spectra(offval = self.offval, plotspan = self.plotspan, saveflag=0,filepath = '',figdim=(15,10))
            self.fig_sub, self.ax_sub = self.plot_all_sub(offval = self.offval)
            self.fig_atp,self.ax_atp = self.plot_atomic_percent()

            # return fig_dict, ax_dict


    #### Analysis Functions

    def bg_sub_all(self):
        """Backgrounds subtraction on all the scans in all the spectra objects.

        Calls spectra.bg_sub() on all spectra 

        Parameters
        ----------

        Returns
        -------

        See Also
        --------
        :func: spectra.bg_sub()
        """        
        for spectra in self.element_scans:
            if not spectra in ['XPS','Valence','vb','Survey']:
                print(self.bg_info[spectra])
                self.__dict__[spectra].bg_sub(subpars = self.bg_info[spectra])
                self.bg_info[spectra] = self.__dict__[spectra].bg_info


    def calc_atomic_percent(self, specify_spectra = None):
        
        len_check = np.min([len(self.__dict__[spectra].I) for spectra in self.element_scans])

        if specify_spectra == None:
            spectra_list = self.element_scans
        else:
            spectra_list = specify_spectra

        self.total_area = [sum([self.__dict__[spectra].area[i]/self.rsf[spectra] for spectra in spectra_list]) for i in range(len_check)]

        for spectra in spectra_list:
            self.__dict__[spectra].atomic_percent = np.asarray([(np.abs(self.__dict__[spectra].area[i]) / self.rsf[spectra] ) / self.total_area[i] \
                for i in range(len_check)])
        self.atomic_percent = {spectra:self.__dict__[spectra].atomic_percent for spectra in spectra_list}




    ### Plotting functions
    def plot_all_spectra(self,offval=0, plotspan=False, saveflag=0,filepath = '',fig = None,ax = None,figdim = None,done_it = False):
        """Plots all of the raw spectra data in the sample object. It will plot the spectra from highest to lowest binding energy
        

        Parameters
        ----------

        Returns
        -------
        matplotlib figure : fig
        dict : matplotlib axes 
           Subplots of all the Raw spectra data. Keys are the orbital names. values are the matplotlib axes object

        """

        if (fig is None) and (ax is None):
            if figdim is None:
                figdim = (15,int(np.ceil((len(self.element_scans)+2)/3))*4)
            
            fig,ax = plt.subplots(int(np.ceil((len(self.element_scans)+2)/3)) ,3, figsize = figdim)
            ax = ax.ravel()

            orderlist = [(orbital,np.max(self.__dict__[orbital].E)) for orbital in self.element_scans]
            orderlist.sort(key=lambda x:x[1])
            orderlist = ['Survey','Valence']+[spec[0] for spec in orderlist][::-1]


            ax = {orb[1]:ax[orb[0]] for orb in enumerate(orderlist)}

        
        for spectra in self.all_scans:
            for i in range(len(self.__dict__[spectra].I)):
                ax[spectra].plot(self.__dict__[spectra].E,self.__dict__[spectra].I[i]+i*offval,label = spectra, color = self.spectra_colors[spectra])

            ax[spectra].set_title(spectra,fontsize=24)
            ax[spectra].set_xlim(max(self.__dict__[spectra].E),min(self.__dict__[spectra].E))
            ax[spectra].set_xlabel('Binding Energy (eV)',fontsize = 20)
            ax[spectra].set_ylabel('Counts/s',fontsize = 20)

            ax[spectra].tick_params(labelsize=16)
            ax[spectra].tick_params(labelsize=16)

            # if (plotspan==True) and (spectra in self.element_scans) and (done_it[spectra] ==False):
            if (plotspan==True) and (spectra in self.element_scans):
                if self.bg_info[spectra][1] == 'shirley':
                    ax[spectra].axvspan( np.min(self.bg_info[spectra][0]), np.max(self.bg_info[spectra][0]) , alpha=0.1, color='orange')
                elif self.bg_info[spectra][1] == 'linear':
                    ax[spectra].axvspan( np.min(self.bg_info[spectra][0]), np.max(self.bg_info[spectra][0]) , alpha=0.1, color='green')
                elif self.bg_info[spectra][1] == 'UT2':
                    ax[spectra].axvspan( np.min(self.bg_info[spectra][0]), np.max(self.bg_info[spectra][0]) , alpha=0.1, color='blue')
                # if not done_it is None:
                #     done_it[spectra] = True

        fig.tight_layout(pad=2)

        if saveflag ==True:
            plt.savefig(filepath,bbox_inches = "tight")
        
        return fig, ax
            
            
            
    def plot_all_sub(self,offval=0):
        """Plots all of the background subtracted spectra

        Parameters
        ----------
        offval : int,float
            Will stack each spectra by offval value

        Returns
        -------
        matplotlib figures : fig
           Multi-plot plots of the data.

        See Also
        --------
        :func: plot_all_spectra(), plot_all_sub(), plot_atomic_percent()
        """

        fig,ax = plt.subplots(int(np.ceil(len(self.element_scans)/2)) ,2, figsize = (15,15))
        ax = ax.ravel()

        orderlist = [(orbital,np.max(self.__dict__[orbital].E)) for orbital in self.element_scans]
        orderlist.sort(key=lambda x:x[1])
        orderlist = [spec[0] for spec in orderlist][::-1]


        ax = {orb[1]:ax[orb[0]] for orb in enumerate(orderlist)}

        for spectra in self.element_scans:

            for i in range(len(self.__dict__[spectra].I)):
                
                ax[spectra].plot(self.__dict__[spectra].esub,self.__dict__[spectra].isub[i] + offval*i,label = spectra, color = self.spectra_colors[spectra])


            ax[spectra].set_title(spectra,fontsize=24)
            ax[spectra].set_xlim(max(self.__dict__[spectra].esub),min(self.__dict__[spectra].esub))

            # if self.normalize_subtraction ==True:
            #     ax[j].set_ylim(-0.05,(np.ceil(np.max([np.max(self.__dict__[spectra].isub[i]) for i in range(len(self.__dict__[spectra].isub))\
            #                    if np.max(self.__dict__[spectra].isub[i]) <1])*10))/10)

            ax[spectra].set_xlabel('Binding Energy (eV)',fontsize = 20)
            ax[spectra].set_ylabel('Counts/s',fontsize = 20)

            ax[spectra].tick_params(labelsize=16)
            ax[spectra].tick_params(labelsize=16)

            if self.bg_info[spectra][1] == 'shirley':
                ax[spectra].axvspan( np.min(self.bg_info[spectra][0]), np.max(self.bg_info[spectra][0]) , alpha=0.1, color='orange')
            elif self.bg_info[spectra][1] == 'linear':
                ax[spectra].axvspan( np.min(self.bg_info[spectra][0]), np.max(self.bg_info[spectra][0]) , alpha=0.1, color='green')
            elif self.bg_info[spectra][1] == 'UT2':
                ax[spectra].axvspan( np.min(self.bg_info[spectra][0]), np.max(self.bg_info[spectra][0]) , alpha=0.1, color='blue')


        # fig.legend(fit_legend,loc='center left', bbox_to_anchor=(1.0, 0.5),fontsize=20)

        fig.tight_layout()
        return fig, ax



    def plot_atomic_percent(self, infig = None, inax = None, **pltkwargs):
        """Plots the atomic percent

        Parameters
        ----------
        infig: matplotlib.fig
           Pass in a figure to plot multiple samples on same graph
        inax: matplotlib.axes
           Pass in axes to plot multiple samples on same graph

        Returns
        -------
        matplotlib figures : fig
           Multi-plot plots of the data.

        See Also
        --------
        :func: plot_all_spectra(), plot_all_sub(), plot_atomic_percent()
        """
        if (infig == None) and (inax == None):
            fig, ax = plt.subplots(figsize = (18,8))
        else:
            fig = infig
            ax = inax
        x = np.arange(len(self.__dict__['C1s'].atomic_percent))

        if self.sputter_time is not None:
            x = x * self.sputter_time
        
        for spectra in self.atomic_percent.keys():

            ax.plot(x,self.atomic_percent[spectra]*100, color = self.spectra_colors[spectra],linewidth = 3,**pltkwargs)


        # ax.title(sample,fontsize = 24)
        
        if self.sputter_time is not None:
            ax.set_xlabel('Sputter Time',fontsize = 30)
        else:
            ax.set_xlabel('Position',fontsize = 30)
        ax.set_ylabel('Atomic Percent',fontsize = 30)

        ax.tick_params(labelsize=20)

        ax.set_xticks(x)
        ax.set_ylim(ymin = 0)
        if hasattr(self.__dict__['C1s'],'positions'):
            ax.set_xticklabels(self.__dict__['C1s'].positions,rotation = 80)
        ax.legend(list(self.atomic_percent.keys()),bbox_to_anchor=(0.85, 0.4, 0.5, 0.5), loc='lower center',fontsize = 20)

        return fig, ax


    # Compound Spectra Methods
    def build_compound_spectra(self,name,spectra_list):
        self.__dict__[name] = CompoundSpectra(spectra_list)