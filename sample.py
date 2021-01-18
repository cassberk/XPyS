# Try to load individual packages at some point
import pandas as pd
from copy import deepcopy as dc
from xps.helper_functions import *
from xps import bkgrds as backsub
from xps.data_io import load_excel
import lmfit as lm
import numpy as np
import matplotlib.pyplot as plt

# from xps.spectra import xps_spec
import xps
import xps.spectra as sp
from xps.spectra import spectra
import xps.VAMAS
from ipywidgets.widgets import Checkbox, Button, Layout, HBox, VBox, Output, Text
import os


# def _buildsample(state, funcdefs=None):
#     """Build sample from saved state.

#     Intended for internal use only.

#     """
#     if len(state) != 3:
#         raise ValueError("Cannot restore Model")
#     known_funcs = {}
#     for fname in lineshapes.functions:
#         fcn = getattr(lineshapes, fname, None)
#         if callable(fcn):
#             known_funcs[fname] = fcn
#     if funcdefs is not None:
#         known_funcs.update(funcdefs)

#     left, right, op = state
#     if op is None and right is None:
#         (fname, fcndef, name, prefix, ivars, pnames,
#          phints, nan_policy, opts) = left
#         if not callable(fcndef) and fname in known_funcs:
#             fcndef = known_funcs[fname]

#         if fcndef is None:
#             raise ValueError("Cannot restore Model: model function not found")

#         model = Model(fcndef, name=name, prefix=prefix,
#                       independent_vars=ivars, param_names=pnames,
#                       nan_policy=nan_policy, **opts)

#         for name, hint in phints.items():
#             model.set_param_hint(name, **hint)
#         return model
#     else:
#         lmodel = _buildmodel(left, funcdefs=funcdefs)
#         rmodel = _buildmodel(right, funcdefs=funcdefs)
#         return CompositeModel(lmodel, rmodel, getattr(operator, op))




class sample:

    def __init__(self,dataload_obj = None ,bkgrd_subtraction_dict = None, data_dict_idx = None, overview = True, sputter_time = None, offval=0, plotflag = True, plotspan = False,\
        plot_legend = None,normalize_subtraction = False,name = None,spectra_colors = None,load_derk = False, **kws):
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
        self.bg_info = bkgrd_subtraction_dict
        dfSF = pd.read_excel(r'/Volumes/GoogleDrive/My Drive/XPS/XPS_Library/Sensitivity_Factors.xlsx', keep_default_na = False)

        self.rsf = {dfSF['element'][dfSF['SF'] != ''].iloc[i] : dfSF['SF'][dfSF['SF'] != ''].iloc[i] for i in range(len(dfSF[dfSF['SF'] != '']))}
        self.offval = offval
        self.plotflag = plotflag
        self.plotspan = plotspan
        self.normalize_subtraction = normalize_subtraction
        self.plot_legend = plot_legend
        self.sputter_time = sputter_time

        if load_derk ==True:
            self.load_derk_sample_object(dataload_obj)

        elif type(dataload_obj) == xps.VAMAS.VAMASExperiment:
            self.load_experiment_sample_from_vamas(dataload_obj)

        elif (type(dataload_obj) == str) or (type(dataload_obj) == dict):
            self.load_experiment_sample_from_excel(dataload_obj,name)

        if (spectra_colors is None) and (dataload_obj != None):
            self.spectra_colors = {spectra:None for spectra in self.element_scans}
        else:
            self.spectra_colors = spectra_colors

        if dataload_obj is None:
            overview = False
        if overview:
            self.fig_dict, self.ax_dict = self.xps_overview()


    def load_experiment_sample_from_vamas(self,vamas_obj):
        
        self.element_scans = list(dict.fromkeys([spectra for spectra in \
            [''.join(block.species_label.split()) for block in vamas_obj.blocks] if (spectra != 'V.B.') and (spectra != 'XPSSurvey')]))

        all_scans = list(dict.fromkeys([scan for scan in [''.join(block.species_label.split()) for block in vamas_obj.blocks]]))
        name_change = {'V.B.':'vb','XPSSurvey':'survey'}
        self.all_scans = [name_change.get(n, n) for n in all_scans]
        self.data_raw = vamas_obj
        
        for scan in self.all_scans:
            if scan =='vb':
                self.__dict__[scan] = sp.spectra()
                sp.spectra.load_experiment_spectra_from_vamas(self.__dict__[scan],vamas_obj,orbital = 'V.B.')
            elif scan =='survey':
                self.__dict__[scan] = sp.spectra()
                sp.spectra.load_experiment_spectra_from_vamas(self.__dict__[scan],vamas_obj,orbital = 'XPSSurvey')                
            else:
                self.__dict__[scan] = sp.spectra(bg_info = self.bg_info[scan])
                sp.spectra.load_experiment_spectra_from_vamas(self.__dict__[scan],vamas_obj,orbital = scan)

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


    def load_derk_sample_object(self,dataload):
        filepath = os.path.join('/Volumes/GoogleDrive/My Drive/XPS/XPS_Library/xps/samples',\
        dataload,'sample_attributes.pkl')
        f = open(filepath, 'rb')   # 'r' for reading; can be omitted
        savedict_load = pickle.load(f)         # pickled dictionary with sample attributes
        f.close() 

        for key in savedict_load.keys():
            if savedict_load[key] != 'spectra_object':
                self.__dict__[key] = savedict_load[key]

            elif savedict_load[key] =='spectra_object':
                print(savedict_load['sample_name']+'-'+key)

                self.__dict__[key] = sp.spectra(savedict_load['sample_name'],key,load_spectra_object = True)

        # analyze = False








    ####
    def xps_overview(self):
        """Returns an overview of the spectra that are held in sample.

       The overview plots all of the raw signals, the background subtracted
       signals and the calculated atomic percent

        Parameters
        ----------

        Returns
        -------
        matplotlib figures : fig
           Multi-plot plots of the data.
        ipython widgets : widgets
            for interactively saving the plots

        See Also
        --------
        :func: plot_all_spectra(), plot_all_sub(), plot_atomic_percent()

        """
        self.bg_sub_all()
        self.calc_atomic_percent()

        if self.plotflag == True:

            save_figs_button = Button(description="Save Figures") 

            saved_root_name = Text(
                value=self.sample_name,
                placeholder='Save filename',
                disabled=False,
                layout = Layout(width = '200px', margin = '0 5px 0 0')
                )
            save_figs_chkbxs = {init_fig: Checkbox(
                value= False,
                description=str(init_fig),
                style = {'description_width': 'initial'},
                disabled=False,
                indent=False
                ) for init_fig in ['Raw','Subtracted','Atomic_Percent'] }

            display( VBox( [saved_root_name,HBox( [HBox( list( save_figs_chkbxs[chks] for chks in save_figs_chkbxs.keys() ) ), save_figs_button] ) ] ) )
            # out = Output()
            # display(out)

            fig_dict = {}
            ax_dict = {}
            fig_dict['Raw'], ax_dict['Raw']= self.plot_all_spectra(offval = self.offval, plotspan = self.plotspan, saveflag=0,filepath = '',figdim=(15,10))
            fig_dict['Subtracted'], ax_dict['Subtracted'] = self.plot_all_sub(offval = self.offval)
            fig_dict['Atomic_Percent'], ax_dict['Atomic_Percent'] = self.plot_atomic_percent()

            @save_figs_button.on_click
            def save_figs_on_click(b):

                if not os.path.exists(os.path.join(os.getcwd(),'figures')):
                    os.makedirs(os.path.join(os.getcwd(),'figures'))

                for figure in save_figs_chkbxs.keys():
                    if save_figs_chkbxs[figure].value:

                        save_location = os.path.join( os.getcwd(),'figures',saved_root_name.value  + '_' + str(figure) )     
                        fig_dict[figure].savefig(save_location, bbox_inches='tight')

        return fig_dict, ax_dict


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
            self.__dict__[spectra].bg_sub()


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
    def plot_all_spectra(self,offval=0, plotspan=False, saveflag=0,filepath = '',figdim = (15,20)):

        fig,ax = plt.subplots(int(np.ceil((len(self.element_scans)+2)/3)) ,3, figsize = figdim)
        ax = ax.ravel()
        # print(len(ax))
        j = 0
        for spectra in self.all_scans:
            for i in range(len(self.__dict__[spectra].I)):
                ax[j].plot(self.__dict__[spectra].E,self.__dict__[spectra].I[i]+i*offval,label = spectra, color = self.spectra_colors[spectra])

            ax[j].set_title(spectra,fontsize=24)
            ax[j].set_xlim(max(self.__dict__[spectra].E),min(self.__dict__[spectra].E))
            ax[j].set_xlabel('Binding Energy (eV)',fontsize = 20)
            ax[j].set_ylabel('Counts/s',fontsize = 20)

            ax[j].tick_params(labelsize=16)
            ax[j].tick_params(labelsize=16)

            if (plotspan==True) and (spectra in self.element_scans):
                if self.bg_info[spectra][1] == 'shirley':
                    ax[j].axvspan( np.min(self.bg_info[spectra][0]), np.max(self.bg_info[spectra][0]) , alpha=0.1, color='orange')
                elif self.bg_info[spectra][1] == 'linear':
                    ax[j].axvspan( np.min(self.bg_info[spectra][0]), np.max(self.bg_info[spectra][0]) , alpha=0.1, color='green')
                elif self.bg_info[spectra][1] == 'UT2':
                    ax[j].axvspan( np.min(self.bg_info[spectra][0]), np.max(self.bg_info[spectra][0]) , alpha=0.1, color='blue')
            j+=1

        fig.tight_layout(pad=2)

        if saveflag ==True:
            plt.savefig(filepath,bbox_inches = "tight")
        
        return fig, ax
            
            
            
    def plot_all_sub(self,offval=0):

        fig,ax = plt.subplots(int(np.ceil(len(self.element_scans)/2)) ,2, figsize = (15,15))
        ax = ax.ravel()


        # fit_legend = dc(self.__dict__['C1s']['pos names'])

        j = 0
        for spectra in self.element_scans:

            for i in range(len(self.__dict__[spectra].I)):
                
                ax[j].plot(self.__dict__[spectra].esub,self.__dict__[spectra].isub[i] + offval*i,label = spectra, color = self.spectra_colors[spectra])


            ax[j].set_title(spectra,fontsize=24)
            ax[j].set_xlim(max(self.__dict__[spectra].esub),min(self.__dict__[spectra].esub))

            # if self.normalize_subtraction ==True:
            #     ax[j].set_ylim(-0.05,(np.ceil(np.max([np.max(self.__dict__[spectra].isub[i]) for i in range(len(self.__dict__[spectra].isub))\
            #                    if np.max(self.__dict__[spectra].isub[i]) <1])*10))/10)

            ax[j].set_xlabel('Binding Energy (eV)',fontsize = 20)
            ax[j].set_ylabel('Counts/s',fontsize = 20)

            ax[j].tick_params(labelsize=16)
            ax[j].tick_params(labelsize=16)

            if self.bg_info[spectra][1] == 'shirley':
                ax[j].axvspan( np.min(self.bg_info[spectra][0]), np.max(self.bg_info[spectra][0]) , alpha=0.1, color='orange')
            elif self.bg_info[spectra][1] == 'linear':
                ax[j].axvspan( np.min(self.bg_info[spectra][0]), np.max(self.bg_info[spectra][0]) , alpha=0.1, color='green')
            elif self.bg_info[spectra][1] == 'UT2':
                ax[j].axvspan( np.min(self.bg_info[spectra][0]), np.max(self.bg_info[spectra][0]) , alpha=0.1, color='blue')

            j+=1

        # fig.legend(fit_legend,loc='center left', bbox_to_anchor=(1.0, 0.5),fontsize=20)

        fig.tight_layout()
        return fig, ax



    def plot_atomic_percent(self, infig = None, inax = None, **pltkwargs):
        
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
        ax.set_xlabel('Position',fontsize = 30)
        ax.set_ylabel('Atomic Percent',fontsize = 30)

        ax.tick_params(labelsize=20)
        ax.tick_params(labelsize=20)

        ax.set_xticks(x)
        ax.set_ylim(ymin = 0)
        # if 'pos names' in self.data['C1s']:
        #     ax.set_xticklabels(self.data['C1s']['pos names'],rotation = 80);
        ax.legend(list(self.atomic_percent.keys()),bbox_to_anchor=(0.85, 0.4, 0.5, 0.5), loc='lower center',fontsize = 20)

        return fig, ax




















# """All of these use self.data dicitonary. I am going to try to migrate
# away from this and store everything in .vms files and just use these objects
# to hold the data
# """

    # ####
    # def xps_overview(self,analyze = True):
        
    #     if analyze == True:

    #         self.bksub_all(cropdic = self.bkgrd_sub_dict)
    #         self.calc_total_area()

    #     if self.plotflag == True:

    #         save_figs_button = Button(description="Save Figures") 

    #         saved_root_name = Text(
    #             value=self.sample_name,
    #             placeholder='Save filename',
    #             disabled=False,
    #             layout = Layout(width = '200px', margin = '0 5px 0 0')
    #             )
    #         save_figs_chkbxs = {init_fig: Checkbox(
    #             value= False,
    #             description=str(init_fig),
    #             style = {'description_width': 'initial'},
    #             disabled=False,
    #             indent=False
    #             ) for init_fig in ['Raw','Subtracted','Atomic Percent'] }

    #         display( VBox( [saved_root_name,HBox( [HBox( list( save_figs_chkbxs[chks] for chks in save_figs_chkbxs.keys() ) ), save_figs_button] ) ] ) )
    #         # out = Output()
    #         # display(out)

    #         fig_dict = {}
    #         ax_dict = {}
    #         fig_dict['Raw'], ax_dict['Raw']= self.plot_all_spectra(cropdic = self.bkgrd_sub_dict, offval = self.offval, plotspan = self.plotspan, saveflag=0,filepath = '',figdim=(15,10))
    #         fig_dict['Subtracted'], ax_dict['Subtracted'] = self.plot_all_sub(cropdic = self.bkgrd_sub_dict)
    #         fig_dict['Atomic Percent'], ax_dict['Atomic Percent'] = self.plot_atomic_percent()

    #         @save_figs_button.on_click
    #         def save_figs_on_click(b):
    #             # with out:

    #             if not os.path.exists(os.path.join(os.getcwd(),'figures')):
    #                 os.makedirs(os.path.join(os.getcwd(),'figures'))

    #             for figure in save_figs_chkbxs.keys():
    #                 if save_figs_chkbxs[figure].value:

    #                     save_location = os.path.join( os.getcwd(),'figures',saved_root_name.value  + '_' + str(figure) )     
    #                     # print(save_location)     
    #                     fig_dict[figure].savefig(save_location, bbox_inches='tight')



    # #### Analysis Functions

    # def bksub_all(self,cropdic = {}):
    #     # print('subbing')
    #     if not bool(cropdic):
    #         print('No background subtraction dictionary defined')
    #         return
        
    #     for spectra in self.element_scans:

    #         self.data[spectra]['isub'] = dc([[] for k in range(len(self.data[spectra]['intensity']))])
    #         self.data[spectra]['bkgd'] = dc([[] for k in range(len(self.data[spectra]['intensity']))])
    #         self.data[spectra]['area'] = dc([[] for k in range(len(self.data[spectra]['intensity']))]) 
    #         self.data[spectra]['params'] = dc([[] for k in range(len(self.data[spectra]['intensity']))])  
    #         if cropdic[spectra][1] =='UT2':
    #             self.data[spectra]['params'] = dc([[] for k in range(len(self.data[spectra]['intensity']))])  
        
    #         for i in range(len(self.data[spectra]['intensity'])):


    #             self.data[spectra]['esub'], self.data[spectra]['isub'][i],\
    #             self.data[spectra]['bkgd'][i],self.data[spectra]['params'][i],\
    #             self.data[spectra]['area'][i] = \
    #             xps_spec.bg_sub(self.__dict__[spectra],self.data[spectra]['energy'],self.data[spectra]['intensity'],i,cropdic[spectra])
                
    #             if self.normalize_subtraction == True:

    #                 if self.data[spectra]['area'][i] > 1000:
    #                     self.data[spectra]['isub'][i] = self.data[spectra]['isub'][i]/self.data[spectra]['area'][i]
    #                     self.data[spectra]['bkgd'][i] = self.data[spectra]['bkgd'][i]/self.data[spectra]['area'][i]
    #                 else:
    #                     self.data[spectra]['isub'][i] = self.data[spectra]['isub'][i]/1e6
    #                     self.data[spectra]['bkgd'][i] = self.data[spectra]['bkgd'][i]/1e6
                        
    # def calc_total_area(self):
        
    #     cycles = len(self.data[list(self.data.keys())[0]]['intensity'])

    #     self.data['total area'] = [[] for k in range(cycles)]

    #     for i in range(cycles):
    #         tota = 0

    #         for spectra in self.element_scans:
                    
    #             tota += np.abs(self.data[spectra]['area'][i])/self.rsf[spectra]

    #         self.data['total area'][i] = dc(tota)
            


    #     for spectra in self.element_scans:

    #         self.data[spectra]['atperc'] = np.empty(cycles)
            
    #         for i in range(cycles):  
                                    
    #             self.data[spectra]['atperc'][i] = (np.abs(self.data[spectra]['area'][i])/self.rsf[spectra])/self.data['total area'][i] 
                    










    # ### Plotting functions
    # def plot_all_spectra(self, cropdic,offval=0, plotspan=False, saveflag=0,filepath = '',figdim = (15,40)):

    #     fig,ax = plt.subplots(int(np.ceil((len(self.element_scans)+2)/3)) ,3, figsize = figdim)
    #     ax = ax.ravel()
    #     # print(len(ax))
    #     iter = 0
    #     for spectra in self.data.keys():

    #         if spectra not in ('total area'):

    #             for i in range(len(self.data[spectra]['intensity'])):
    #                 ax[iter].plot(self.data[spectra]['energy'],self.data[spectra]['intensity'][i]+i*offval,label = spectra)

    #             ax[iter].set_title(spectra,fontsize=24)
    #             ax[iter].set_xlim(max(self.data[spectra]['energy']),min(self.data[spectra]['energy']))
    #             ax[iter].set_xlabel('Binding Energy (eV)',fontsize = 20)
    #             ax[iter].set_ylabel('Counts/s',fontsize = 20)

    #             ax[iter].tick_params(labelsize=16)
    #             ax[iter].tick_params(labelsize=16)

    #             if plotspan==True:
    #                 if cropdic[spectra][1] == 'shirley':
    #                     ax[iter].axvspan( np.min(cropdic[spectra][0]), np.max(cropdic[spectra][0]) , alpha=0.1, color='orange')
    #                 elif cropdic[spectra][1] == 'linear':
    #                     ax[iter].axvspan( np.min(cropdic[spectra][0]), np.max(cropdic[spectra][0]) , alpha=0.1, color='green')
    #                 elif cropdic[spectra][1] == 'UT2':
    #                     ax[iter].axvspan( np.min(cropdic[spectra][0]), np.max(cropdic[spectra][0]) , alpha=0.1, color='blue')
    #             iter+=1


    #     fig.tight_layout(pad=2)


    #     if saveflag ==True:
    #         plt.savefig(filepath,bbox_inches = "tight")
        
    #     return fig, ax
            
            
            
    # def plot_all_sub(self,cropdic = {}, offval=0):

    #     if not bool(cropdic):
    #         print('No background subtraction dictionary defined')
    #         return


    #     fig,ax = plt.subplots(int(np.ceil(len(self.element_scans)/2)) ,2, figsize = (15,15))
    #     ax = ax.ravel()


    #     fit_legend = dc(self.data['C1s']['pos names'])

    #     iter = 0
    #     for spectra in self.element_scans:

    #         en = self.data[spectra]['esub']

    #         for i in range(len(self.data[spectra]['intensity'])):
                
    #             ax[iter].plot(self.data[spectra]['esub'],self.data[spectra]['isub'][i] + offval*i,label = spectra)


    #         ax[iter].set_title(spectra,fontsize=24)
    #         ax[iter].set_xlim(max(self.data[spectra]['esub']),min(self.data[spectra]['esub']))
    #         if self.normalize_subtraction ==True:
    #             ax[iter].set_ylim(-0.05,(np.ceil(np.max([np.max(self.data[spectra]['isub'][i]) for i in range(len(self.data[spectra]['isub']))\
    #                            if np.max(self.data[spectra]['isub'][i]) <1])*10))/10)

    #         ax[iter].set_xlabel('Binding Energy (eV)',fontsize = 20)
    #         ax[iter].set_ylabel('Counts/s',fontsize = 20)

    #         ax[iter].tick_params(labelsize=16)
    #         ax[iter].tick_params(labelsize=16)

    #         if cropdic[spectra][1] == 'shirley':
    #             ax[iter].axvspan( np.min(cropdic[spectra][0]), np.max(cropdic[spectra][0]) , alpha=0.1, color='orange')
    #         elif cropdic[spectra][1] == 'linear':
    #             ax[iter].axvspan( np.min(cropdic[spectra][0]), np.max(cropdic[spectra][0]) , alpha=0.1, color='green')
    #         elif cropdic[spectra][1] == 'UT2':
    #             ax[iter].axvspan( np.min(cropdic[spectra][0]), np.max(cropdic[spectra][0]) , alpha=0.1, color='blue')

    #         iter+=1

    #     fig.legend(fit_legend,loc='center left', bbox_to_anchor=(1.0, 0.5),fontsize=20)

    #     fig.tight_layout()
    #     return fig, ax



    # def plot_atomic_percent(self):
        
    #     leglist = []
    #     fig, ax = plt.subplots(figsize = (12,8))

    #     x = np.arange(len(self.data['C1s']['atperc']))

    #     for spectra in self.element_scans:
    #         ax.plot(x,self.data[spectra]['atperc']*100)
    #         leglist.append(spectra)


    #     # ax.title(sample,fontsize = 24)
    #     ax.set_xlabel('Position',fontsize = 30)
    #     ax.set_ylabel('Atomic Percent',fontsize = 30)

    #     ax.tick_params(labelsize=30)
    #     ax.tick_params(labelsize=30)

    #     ax.set_xticks(x)
    #     ax.set_ylim(ymin = 0)
    #     if 'pos names' in self.data['C1s']:
    #         ax.set_xticklabels(self.data['C1s']['pos names'],rotation = 80);
    #     ax.legend(leglist,bbox_to_anchor=(0.85, 0.4, 0.5, 0.5), loc='lower center',fontsize = 20)
        
    #     return fig, ax