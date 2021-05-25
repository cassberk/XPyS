import h5py
import json
import lmfit as lm
import os
import numpy as np
import XPyS
import XPyS.config as cfg
from XPyS.helper_functions import *
from lmfit.models import GaussianModel, LorentzianModel, PseudoVoigtModel, SkewedVoigtModel

def find_files(filename, search_path):
    result = []

    # Wlaking top-down from the root
    for root, directory, files in os.walk(search_path):
        if filename in files:
            result.append(os.path.join(root, filename))
    return result


def load_model(model):
    model_filename = model+'.hdf5'
    model_filepath = find_files(model_filename,os.path.join(cfg.package_location,'XPyS/saved_models'))

    if len(model_filepath) > 1:
        print('there are more than one files with that name',model_filepath)
        return
    print('Model loaded from:',model_filepath[0])

    f = h5py.File(model_filepath[0],'r')
    
    # model
    try:
        m = lm.model.Model(lambda x: x)
        mod = m.loads(f['mod'][...][0])
    except:
        print(model,'couldnt save model')

    # params
    p = lm.parameter.Parameters()
    try:
        pars = p.loads(f['params'][...][0])
    except:
        print(model,'couldnt load params')

    # pairlist
    try:
        plist = [tuple(f.attrs['pairlist'][i]) for i in range(len(f.attrs['pairlist']))]
        pairlist = []
        for pair in plist:
            if pair[1] == '':
                pairlist.append(tuple([pair[0]]))
            else:
                pairlist.append(pair)

    except:
        print(model,'couldnt load pairlist')


    # model_ctrl
    try:
        element_ctrl = list(f.attrs['element_ctrl'])
    except:
        print(model,'couldnt load element_ctrl')
        
    f.close()
    
    return mod, pars, pairlist, element_ctrl


def model_list(startpath = None):
    if startpath is None:
        start_dir = os.path.join(cfg.package_location,"XPyS/saved_models")
    else:
        start_dir = startpath
    result = []
    for root, dirs, files in os.walk(start_dir):
        for file in files:
            if file.endswith(".hdf5"):
                result.append(file.split('.')[-2])
                # print(file.split('.')[-2])
    return result


def dumper(obj):
    try:
        return obj.toJSON()
    except:
        return obj.tolist()


def model_to_hdf5(model_name,mod,pars,pairlist,element_ctrl):

    if model_name.split('.')[-1] =='hdf5':
        model_name = model_name.split('.')[0]
        
    with h5py.File(model_name+'.hdf5','w') as f:

        mod_attr = ('params','mod')
        dt = h5py.special_dtype(vlen=str) 

        # params
        try:
            dt = h5py.special_dtype(vlen=str)
            data_temp = np.asarray([pars.dumps()], dtype=dt)
            f.create_dataset('params', data=data_temp)
        except:
            print(model_name,'couldnt save params')

        # model
        try:
            data_temp = np.asarray([mod.dumps()], dtype=dt) 
            f.create_dataset('mod', data=data_temp)
        except:
            print(model_name,'couldnt save model')

        # pairlist
        try:
#             dt = h5py.special_dtype(vlen=str)
#             plist = np.asarray([pair for pair in pairlist], dtype=dt) 
            f.attrs['pairlist'] = pairlist
        except:
            print(model_name,'couldnt save pairlist')


        # element_ctrl
        try:
            f.attrs['element_ctrl'] = element_ctrl
        except:
            print(model_name,'couldnt save element_ctrl')

def save_spectra_model(spectra_model,element = None,name = None,filepath = None):
    """Save a SpectraModel object to .hdf5
    Parameters
    ----------
    spectra_model : SpectraModel instance
        SpectraModel to be saved.
    element: str
        Element that is being modeled. For example: 'C' or 'Nb' or 'O'. This should be in line with what the 
        instrument spits out. For instance, dont name a model for Nb, Niobium because then the model will be saved
        in the Niobium3d folder, and other functionality will not work since the folder names rely on the way in which the
        instrument exports the data. Perhaps it would be good to add a list of possible element names, but that is not
        included yet
    name : str
        Name of the model. This can be anything and will be saved in the appropriate folder. 
    filepath: str
        The loation of the folder can be specified. However, this is not recommended because to fully use
        the package and the GUI add ons the models should all be saved in the saved_models folder in the appropriately
        named subfolders.
        
    """
    if (name is None and filepath is None):
        raise NameError('You must either name the model or specify the filepath with the name as the last entry in the path')
    if (not name is None and element is None):
        raise NameError('You must specify the element. Ex: "C", "Nb" and "O"')
    if (not name is None and not filepath is None):
        raise NameError('You must choose either a name or a filepath')

    model = spectra_model.model
    pars = spectra_model.pars

    # singlets in tuples do not save using .hdf5. Need to add an empty string to the tuple to save.
    pairlist = []
    for pair in spectra_model.pairlist:
        if len(pair) == 1:
            pairlist.append((pair[0],''))
        else:
            pairlist.append(pair)
            
    element_ctrl = spectra_model.element_ctrl

    if filepath is None:
        model_folder = element+spectra_model.orbital
        fpath = os.path.join(cfg.package_location,'XPyS/saved_models',model_folder,name)
    else:
        fpath = filepath
    
    model_to_hdf5(fpath,model,pars,pairlist,spectra_model.element_ctrl)
    print('Saved model to: ',fpath)


class SpectraModel(lm.Model):

    jprefix = {'1s':['_','_'],'2p':['_32_','_12_'],'3d':['_52_','_32_']}
    jratio = {'1s':['',''],'2p':'(1/2)','3d':'(2/3)'}

    def __init__(self,orbital = None,name='',pars = None,SO_split=None, xdata = None, ydata = None,sigma = 0.6, n_comps = 1):
        """Build a Model to fit the spectra

        Parameters
        ----------
        orbital: str
            string of orbital to model: 1s,2p,3d or 4f
            
        name: string,list
            name of the compound being modeled. This will become a prefix to all of the model parameters
            If this is left empty and there are more than one components specified, then the prefix will be 
            comp0, comp1, ... comp(n-1) where n is n_comps.
            If a string is named then an int will be added to the end for each component.
            If a list is specified then each element in the list will be the prefix name for each component. In
            this case the length of the list and n_comps must be the same.
            
        pars: lmfit parameter object (ordered dict)
            lmfit parameter object used to fit the model
            
        SO_split: str
            spin orbit splitting of the doublet
        
        xdata: array
            Binding Energy data
            
        ydata: array
            Intensity count data
            
        sigma: float
            sigma of model peak, default is 0.6

        n_comps: int
            number of singlets/doublets to add to model
           

        Notes
        -----


        Examples
        --------
        """

        self.pars = pars
        self.orbital = orbital
        self.xdata = xdata
        self.ydata = ydata
        self.model_type = []

        if (type(name) is list and len(name) != n_comps):
            raise ValueError('The number of prefixes in name does not match the number of components in n_comp')

        if (not type(name) is list and name == ''):
            name = ['comp'+str(i) for i in range(n_comps)]
        elif (not type(name) is list and name != ''):
            name = [name+str(i) for i in range(n_comps)]
        
        if self.orbital == '1s':
            for i in range(n_comps):
                self.singlet(name = name[i],pars = self.pars)
        elif self.orbital in ['2p','3d','4f']:
            if xdata is None:
                raise ValueError('Need xdata')
            if ydata is None:
                raise ValueError('Need ydata')
            for i in range(n_comps):
                self.doublet(name=name[i],pars = self.pars,SO_split=SO_split)

    
    def doublet(self,name = '',prefix = None, pars = None,SO_split=None,peak_ratio = None, sigma = 0.6,center = None):

        if SO_split is None:
            raise ValueError('You Must specify a spin orbit splitting for the doublet')
        if (self.orbital == '1s' and peak_ratio is None):
            raise ValueError('You must specify a peak_ratio if you are adding a doublet to the 1s orbital')
        if (self.orbital == '1s' and prefix is None and name == ''):
            raise ValueError('You must specify either the prefixes or the name if you are adding a doublet to the 1s orbital')
        if (self.orbital =='1s' and prefix is None):
            peak1_prefix = name + '0'
            peak2_prefix = name + '1'            
        elif (self.orbital != '1s' and prefix is None):
            peak1_prefix = name + jprefix[self.orbital][0]
            peak2_prefix = name + jprefix[self.orbital][1]
        else:
            if (type(prefix)!= list or len(prefix)!=2):
                raise ValueError('Prefix must be a list with two elements for each doublet peakname')
            peak1_prefix = prefix[0]
            peak2_prefix = prefix[1]

        if peak1_prefix[-1] != '_':
            peak1_prefix +='_'
        if peak2_prefix[-1] != '_':
            peak2_prefix +='_'

        if self.pars is None:
            self.pars = lm.Parameters()      
        peak1 = PseudoVoigtModel(prefix = peak1_prefix)
        peak2 = PseudoVoigtModel(prefix = peak2_prefix)
        
        if center is None:
            center = self.xdata[index_of(self.ydata, np.max(self.ydata))]
        
        self.pars.update(peak1.make_params())
        self.pars.update(peak2.make_params())

        self.pars[peak1_prefix + 'amplitude'].set(np.max(self.ydata),min = 0,vary = 1)
        self.pars[peak1_prefix + 'center'].set(center, vary=1)
        self.pars[peak1_prefix + 'sigma'].set(sigma, vary=1)
        self.pars[peak1_prefix + 'fraction'].set(0.05, vary=1)

        if peak_ratio is None:
            peak_ratio = jratio[self.orbital]
        self.pars[peak2_prefix + 'amplitude'].set(expr = peak_ratio+'*' + peak1_prefix + 'amplitude')
        self.pars[peak2_prefix + 'center'].set(expr = peak1_prefix + 'center+' +str(SO_split))
        self.pars[peak2_prefix + 'sigma'].set(expr = peak1_prefix + 'sigma')
        self.pars[peak2_prefix + 'fraction'].set(expr = peak1_prefix + 'fraction')
        
        model = peak1 + peak2
        if not hasattr(self,'model'):
            self.model = model
            self.pairlist = [(peak1_prefix,peak2_prefix)]
            self.element_ctrl = [0]
            self.model_type.append('doublet')
        else:
            self.model += model
            self.pairlist.append((peak1_prefix,peak2_prefix))
            if self.model_type[-1] == 'singlet':
                next_ctrl_peak = self.element_ctrl[-1]+1
            elif self.model_type[-1] == 'doublet':
                next_ctrl_peak = self.element_ctrl[-1]+2
            self.element_ctrl.append(next_ctrl_peak)
            self.model_type.append('doublet')
        
    def singlet(self,name,pars = None,sigma = 0.6,center = None):
        
        peak1_prefix = name
        if peak1_prefix[-1] != '_':
            peak1_prefix +='_'
        
        if self.pars is None:
            self.pars = lm.Parameters()      
        peak1 = PseudoVoigtModel(prefix = peak1_prefix)
        
        if center is None:
            center = self.xdata[index_of(self.ydata, np.max(self.ydata))]
        
        self.pars.update(peak1.make_params())
        
        self.pars[peak1_prefix + 'amplitude'].set(np.max(self.ydata),min = 0,vary = 1)
        self.pars[peak1_prefix + 'center'].set(center, vary=1)
        self.pars[peak1_prefix + 'sigma'].set(sigma, vary=1)
        self.pars[peak1_prefix + 'fraction'].set(0.05, vary=1)
        
        model = peak1
        if not hasattr(self,'model'):
            self.model = model
            self.pairlist = [(peak1_prefix,)]
            self.element_ctrl = [0]
            self.model_type.append('singlet')
        else:
            self.model += model
            self.pairlist.append((peak1_prefix,))
            if self.model_type[-1] == 'singlet':
                next_ctrl_peak = self.element_ctrl[-1]+1
            elif self.model_type[-1] == 'doublet':
                next_ctrl_peak = self.element_ctrl[-1]+2
            self.element_ctrl.append(next_ctrl_peak)
            self.model_type.append('singlet')
                
#     def __add__(self, other):
#         """+"""
#         return SpectraModel(self, other, operator.add)