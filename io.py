import os, sys
import numpy as np
import pandas as pd
from lmfit.model import save_modelresult, save_model, load_model
import pickle
# import xps
from xps.sample import sample


"""
Saving and loading sample objects and spectra objects
"""

def save_sample(sample_object,filepath = None):
    if filepath == None:
        samplepath = os.path.join('/Volumes/GoogleDrive/My Drive/XPS/XPS_Library/xps/samples',\
                            sample_object.sample_name)
    else:
        samplepath = os.path.join(filepath,\
                            sample_object.sample_name)        

    if not os.path.exists(samplepath):
        os.makedirs(samplepath)

    savedict = {}
    
    for key in sample_object.__dict__.keys():
        if (str(type(sample_object.__dict__[key])).split('.')[-1] == "xps_spec'>") or \
            (str(type(sample_object.__dict__[key])).split('.')[-1] == "guipyter'>"):  
            savedict[key] = 'spectra_object'
            save_spectra(sample_object.__dict__[key],filepath)
            
        else:
            savedict[key] = sample_object.__dict__[key]
            
    with open(os.path.join(samplepath,'sample_attributes.pkl'), 'wb') as f:
        pickle.dump(savedict, f)
    f.close()


def save_spectra(spectra_object,filepath = None):
    if filepath ==None:
        samplepath = os.path.join('/Volumes/GoogleDrive/My Drive/XPS/XPS_Library/xps/samples',\
                            spectra_object.parent_sample)
    else:
        samplepath = os.path.join(filepath,\
                            spectra_object.parent_sample)        

    if not os.path.exists(samplepath):
        os.makedirs(samplepath)

    orbitalpath = os.path.join(samplepath,spectra_object.orbital)
    if not os.path.exists(orbitalpath):
        os.makedirs(orbitalpath)

    
    savedict = {}
    for key in spectra_object.__dict__.keys():
        if str(type(spectra_object.__dict__[key])).split('.')[0] !="<class 'ipywidgets":

            if str(type(spectra_object.__dict__[key])) == "<class 'dict'>":

                if not any([str(type(spectra_object.__dict__[key][subkey])).split('.')[0] =="<class 'ipywidgets" \
                   for subkey in spectra_object.__dict__[key].keys()]):
                    # print((key, str(type(spectra_object.__dict__[key]))+'2'))

                    savedict[key] = spectra_object.__dict__[key]

            elif str(type(spectra_object.__dict__[key])) == "<class 'lmfit.model.CompositeModel'>":     # Need to make this include non_composite models as well
                # print('comp model')
                save_model(spectra_object.mod,os.path.join(orbitalpath,\
                        spectra_object.spectra_name+'_model.sav'))

            elif key == 'fit_results':
                # print('fit results')
                for i in spectra_object.fit_results_idx:
                    save_modelresult(spectra_object.fit_results[i],os.path.join(orbitalpath,\
                        spectra_object.spectra_name+'_fit_result_'+str(i)+'.sav'))
            else:
                # print((key, str(type(spectra_object.__dict__[key]))+'3'))
                savedict[key] = spectra_object.__dict__[key]

    with open(os.path.join(orbitalpath,'spectra_attributes.pkl'), 'wb') as f:
        pickle.dump(savedict, f)
    f.close()



def load_sample(sample, overview = False):
    filepath = os.path.join('/Volumes/GoogleDrive/My Drive/XPS/XPS_Library/xps/samples',\
    sample,'sample_attributes.pkl')
    f = open(filepath, 'rb')   # 'r' for reading; can be omitted
    savedict_load = pickle.load(f)         # pickled dictionary with sample attributes
    f.close() 

    return sample(savedict_load,load_derk = True,overview = overview)






