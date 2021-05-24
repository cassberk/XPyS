import h5py
import json
import lmfit as lm
import os
import numpy as np
from xps_peakfit.helper_functions import *
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
    model_filepath = find_files(model_filename,'/Users/cassberk/code/xps_peakfit/models')

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
        start_dir = "/Users/cassberk/code/xps_peakfit/models"
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


jprefix = {'1s':['_','_'],'2p':['_32_','_12_'],'3d':['_52_','_32_']}
jratio = {'1s':['',''],'2p':'(1/2)','3d':'(2/3)'}

class SpectraModel(lm.Model):
    
    def __init__(self,orbital,name='',pars = None,SO_split=None, xdata = None, ydata = None,n_comps = 1):
        """Build a Model to fit the spectra

        Parameters
        ----------
        orbital: str
            string of orbital to model: 1s,2p,3d or 4f
            
        name: string,list
            name of the compound being modeled. This will become a prefix to all of the model parameters
            If this is left empty and there are more than one components specified, then the prefix will be 
            comp0, comp1, ... comp(n-1) where n is n_comps.
            If a list is specified then each element in the list will be the prefix name for each component.
            
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
                self.singlet(orbital=orbital,name = name[i],pars = self.pars)
        else:
            if SO_split is None:
                raise ValueError('You Must specify a spin orbit splitting for the doublet')
            if xdata is None:
                raise ValueError('Need xdata')
            if ydata is None:
                raise ValueError('Need ydata')
            for i in range(n_comps):
                self.doublet(orbital=orbital,name=name[i],pars = self.pars,SO_split=SO_split, xdata = xdata, ydata = ydata)

    
    def doublet(self,orbital,name='',pars = None,SO_split=0.44,sigma = 0.6,center = None):
        peak1_prefix = name + jprefix[orbital][0]
        peak2_prefix = name + jprefix[orbital][1]
        
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

        self.pars[peak2_prefix + 'amplitude'].set(expr = jratio[orbital]+'*' + peak1_prefix + 'amplitude')
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
        
    def singlet(self,orbital,name='',pars = None,sigma = 0.6,center = None):
        peak1_prefix = name + jprefix[orbital][0]
        
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