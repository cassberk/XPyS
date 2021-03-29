import json
import h5py
import numpy as np
import xps_peakfit.sample
import xps_peakfit.spectra
import json
import lmfit as lm
from pathlib import Path
import os

def dumper(obj):
    try:
        return obj.toJSON()
    except:
        return obj.tolist()

def save_sample(sample_obj,filepath = None, experiment_name = None,force = False):

    if experiment_name == None:
        print('Must Name the experiment')
        return

    if sample_obj.sample_name == None:
        print('You must name the sample')
        return


    f = h5py.File(filepath,'a')
    # if experiment_name in f.keys() and force == False:
    #     print('Experiment already exists with the same name. Set force = True to delete experiment and save a new one \
    #         or save the individual attribute you are interested in')
    #     return

    # elif experiment_name in f.keys() and force == True:
    #     del f[experiment_name]
    #     experiment_group = f.create_group(experiment_name)
    
    # else:
    #     experiment_group = f.require_group(experiment_name)


    if any(['total_area' == it for it in [i[0] for i in f[experiment_name].items()]]):
        del f[experiment_name]['total_area']
    for group_attr in ['positions','bg_info','atomic_percent']:
        if any([group_attr == it for it in [i for i in f[experiment_name].attrs]]):
            del f[experiment_name].attrs[group_attr]

    # total_area
    try:
        f[experiment_name].create_dataset('total_area', data = sample_obj.total_area)
    except:
        print('couldnt total area')
        pass
    
    # scan positions
    try:
        f[experiment_name].attrs['positions'] = sample_obj.positions  #prob want to change this eventually
    except:
        print('position')
        # f[experiment_name].attrs['positions'] = 'Not Specified'
        
    #json dictionaries
    try:
        f[experiment_name].attrs['bg_info'] = json.dumps(sample_obj.bg_info, default=dumper, indent=2)
    except:
        print('couldnt bg_info on sampleobj')
        pass
    try:
        f[experiment_name].attrs['atomic_percent'] =json.dumps(sample_obj.atomic_percent, default=dumper, indent=2)
    except:
        print('couldnt atomic_percent on sample obj')
        pass

    f.close()

    for spectra in sample_obj.element_scans:
        save_spectra_analysis(sample_obj.__dict__[spectra],filepath = filepath,experiment_name = experiment_name,force = force)

     

## Need to work on this since it is not necessary to save all the info each time
def save_spectra_analysis(spectra_obj,filepath = None, experiment_name = None,force = False):

    if experiment_name == None:
        print('Must Name the experiment')
        return

    # if sample_obj.sample_name == None:
    #     print('You must name the sample')
    #     return

    if not hasattr(spectra_obj,'orbital'):
        raise NameError('You must assign an orbital attribute to the spectra object as a string. Ex: "O1s"')
    
    spectra = spectra_obj.orbital

    with h5py.File(filepath,'r+') as f:
        f[experiment_name][spectra]
        datasets_NS = [f[experiment_name][spectra][d].name for d in f[experiment_name][spectra].keys() if str(f[experiment_name][spectra][d][...]) =='Not Specified']
        dataset_exist = [f[experiment_name][spectra][d].name for d in f[experiment_name][spectra].keys()]

        if force == False:
            if any(['fit_results' in ds for ds in dataset_exist]):
                if not any(['fit_results' in ds for ds in datasets_NS]):
                    raise AttributeError('There are currently fit_results saved, use force to overwrite them')
                elif any(['fit_results' in ds for ds in datasets_NS]):
                    del f[experiment_name][spectra]['fit_results']  
                    

        if force == True:
            for dset in ['fit_results','mod','isub','esub','area','bg','thickness','atomic_percent']:
                if hasattr(spectra_obj,dset):
                    if any([dset in ds for ds in dataset_exist]):
                        # print(dset)
                        del f[experiment_name][spectra][dset]

    with h5py.File(filepath,'r+') as f:

        ##################################
        # Datasets
        dsets = ('esub','isub','area','BE_adjust','atomic_percent')
        
        for data in dsets:
            try:
                # if hasattr(spectra_obj,data):
                #     del f[experiment_name][spectra][data]
                f[experiment_name][spectra][data] = spectra_obj.__dict__[data]
            except:
                print('couldnt',data)
    #                 f[experiment_name][spectra][data] = 'Not Specified'
            
        ##################################     
        # Background subtraction
        try:
            f[experiment_name][spectra]['bg'] = spectra_obj.__dict__['bg']
        except:
            print('couldnt bg')
            # f[experiment_name][spectra]['bg'] = 'Not Specified'
            pass
        # crop_info
        try:
            f[experiment_name][spectra]['bg'].attrs['bg_bounds'] = np.asarray(spectra_obj.__dict__['bg_info'][0])
            f[experiment_name][spectra]['bg'].attrs['bg_type'] = spectra_obj.__dict__['bg_info'][1]
            if spectra_obj.__dict__['bg_info'][1] =='UT2':
                f[experiment_name][spectra]['bg'].attrs['bg_starting_pars'] = np.asarray(spectra_obj.__dict__['bg_info'][2])
        except:
            pass

        # bgpars
        if ('bgpars' in spectra_obj.__dict__.keys()) and (any(spectra_obj.bgpars)):
            dt = h5py.special_dtype(vlen=str) 
            bgpars = np.array([spectra_obj.bgpars[i].dumps() for i in range(len(spectra_obj.bgpars))], dtype=dt) 
            f[experiment_name][spectra]['bg'].attrs['bgpars'] = bgpars

        ##################################
        # Fit Results
        # save lmfit spectra attributes as datasets using the lmfit method .dumps() which is a wrapper for json.dumps
        # the lmfit objects are serialied info JSON format and stored as a string array in hdf5
        dt = h5py.special_dtype(vlen=str) 
        try:
            data_temp = np.asarray([spectra_obj.__dict__['fit_results'][i].dumps()\
                    for i in range(len(spectra_obj.__dict__['fit_results']))], dtype=dt) 
            f[experiment_name][spectra].create_dataset('fit_results', data=data_temp)
        except:
            print('couldnt fit results')
            # f[experiment_name][spectra].create_dataset('fit_results',data = 'Not Specified')

        ##################################
        # Model and model attributes
        # Mod
        try:
            data_temp = np.asarray([spectra_obj.__dict__['mod'].dumps()], dtype=dt) 
            f[experiment_name][spectra].create_dataset('mod', data=data_temp)
        except:
            print(spectra,'couldnt save','mod')
            # f[experiment_name][spectra].create_dataset('mod',data = 'Not Specified')
            
        #Params attribute
        try:
            data_temp = np.asarray([spectra_obj.__dict__['params'].dumps()], dtype=dt) 
            f[experiment_name][spectra]['mod'].attrs['params'] = data_temp
        except:
            print(spectra,'couldnt save','params')
            # f[experiment_name][spectra]['mod'].attrs['params']= 'Not Specified'

        # Other Attributes   
        attributes = ('element_ctrl','orbital','pairlist','parent_sample','prefixlist')
        for attr in attributes:
            try:
                f[experiment_name][spectra]['mod'].attrs[attr] = spectra_obj.__dict__[attr]
            except:
                print('couldnt',attr)
                # f[experiment_name][spectra]['mod'].attrs[attr] = 'Not Specified'

        ##################################
        # thickness
        try:
            f[experiment_name][spectra].create_dataset('thickness', data = spectra_obj.thickness)
        except:
            print('couldnt thickness')
    #             experiment_group[spectra].attrs['thickness'] = 'Not Specified'
            pass
        
        try:
            f[experiment_name][spectra]['thickness'].attrs['oxide_thickness'] = json.dumps(spectra_obj.oxide_thickness, default=dumper, indent=2)
        except:
            print('couldnt oxide thickness')
            pass
    #             experiment_group[spectra].attrs['oxide_thickness'] = 'Not Specified'
        try:
            f[experiment_name][spectra]['thickness'].attrs['oxide_thickness_err'] = json.dumps(spectra_obj.oxide_thickness_err, default=dumper, indent=2)
        except:
    #             experiment_group[spectra].attrs['oxide_thickness_err'] = 'Not Specified'
            pass     





def write_vamas_to_hdf5(vamas_obj, hdf5file):
    
    """Function to write the vamas python object to hdf5.
    The .vms text file is first opened and stored as a string array 
    as raw_vamas_data. then the vamas_obj is parsed through in order to 
    individually store the data of the whole vamas file as well as the 
    individual blocks. This second part is performed so that the data 
    can be viewed in an hdf5 viewer
    """
#     TODO:
#     May want to individually pick out the relevant attributes....
    
    with open(vamas_obj.file.name,'r') as v:
        store = ''.join([line for line in v.readlines()])
        
    raw = hdf5file.create_group('raw_data')
    
    dt = h5py.special_dtype(vlen=str) 
    vamfile = np.array([store], dtype=dt) 
    raw.create_dataset('raw_vamas_data', data=vamfile)
    
    for key in list(vamas_obj.__dict__.keys()):

        if key =='file':

            raw.attrs[key] = vamas_obj.__dict__[key].name

        elif key =='blocks':

            grps = {}
            for block in vamas_obj.__dict__['blocks']:

                for block_key in list(block.__dict__.keys()):
                    if block.block_identifier not in grps.keys():

                        blkcnt = 0
                        grps[block.block_identifier] = raw.create_group(block.block_identifier)

                    try:
                        raw[block.block_identifier]['block'+str(blkcnt)]
                    except:
                        raw[block.block_identifier].create_group('block'+str(blkcnt))


                    if block_key =='date':

                        raw[block.block_identifier]['block'+str(blkcnt)].attrs[block_key] = block.__dict__['date'].strftime("%m/%d/%Y, %H:%M:%S")

                    else:

                        raw[block.block_identifier]['block'+str(blkcnt)].attrs[block_key] = block.__dict__[block_key]

                blkcnt+=1

        else:
            raw.attrs[key] = vamas_obj.__dict__[key]
            
#     f.close()



def load_sample(filepath = None, experiment_name = None):

    if experiment_name is None:
        with h5py.File(filepath,'r') as f:
            experiment_options = [exp for exp in f.keys()]
        print('You must choose an experiment, The options are: ',experiment_options)
        return

    sample_obj = xps_peakfit.sample.sample(overview=False)
    sample_obj.load_path = filepath
    sample_obj.experiment_name = experiment_name

    f= h5py.File(filepath,"r")



    exp_attr = ('data_path','element_scans','all_scans','sample_name','positions')
    for attr in exp_attr:
        try:
            sample_obj.__dict__[attr] = f[experiment_name].attrs[attr]
            print(sample_obj.__dict__[attr])
        except:
            print('couldnt',attr)
            pass

    try:
        sample_obj.atomic_percent = json.loads(f[experiment_name].attrs['atomic_percent'])
        for spec in sample_obj.atomic_percent.keys():
            sample_obj.atomic_percent[spec] = np.asarray(sample_obj.atomic_percent[spec])
    except:
        pass

    try:
        sample_obj.data_raw = f[experiment_name]['raw_data']['raw_vamas_data'][...]
    except:
        pass


    # Background subtraction dictionary
    try:
        sample_obj.bg_info = json.loads(f[experiment_name].attrs['bg_info'])
    except:
        pass
    try:
        sample_obj.total_area = f[experiment_name]['total_area'][...]
    except:
        pass

    # if sample_obj.all_scans =='Not Specified':
    #     print('All Scans not specified, please specify them first')
    #     f.close()
    #     return

    for spec in sample_obj.all_scans:

        print(spec)
        # Create a new group that is a spectra object
        # that contains all the spectra and analysis
        sample_obj.__dict__[spec] = load_spectra(filepath = filepath, experiment_name = experiment_name,spec = spec)
        sample_obj.__dict__[spec].spectra_name = sample_obj.sample_name
        # try:
        #     sample_obj.__dict__[spec].atomic_percent = sample_obj.atomic_percent[spec]
        # except:
        #     print('couldnt load atomic percent on',spec)
    f.close()

    return sample_obj






def load_spectra(filepath = None, experiment_name = None,spec = None, openhdf5 = False):

    spectra_obj = xps_peakfit.spectra.spectra(spectra_name = spec)
    spectra_obj.load_path = filepath
    spectra_obj.experiment_name = experiment_name

    # if openhdf5 == False:
    f= h5py.File(filepath,"r")
    try:
        spectra_obj.comments = f[experiment_name][spec].attrs['DS_EXT_SUPROPID_COMMENTS']
    except:
        pass
    try:
        spectra_obj.positions = f[experiment_name][spec]['I'].attrs['Position']
    except:
        print('couldnt load positions')
        pass
    ##################################
    # Datasets
    for data in ['E','I','esub','isub','area','atomic_percent','BE_adjust']:
        try:
            spectra_obj.__dict__[data] = f[experiment_name][spec][data][...]
        except:
            print('couldnt',data)
            pass

    # try:
    #     spectra_obj.parent_sample = f[experiment_name][spec].attrs['parent_sample']
    # except:
    #     print('couldnt parent_sample')
    #     pass

    ##################################
    #Background Subtraction data and parameters
    # bg
    try:
        spectra_obj.bg = f[experiment_name][spec]['bg'][...]
    # bgpars
        if 'bgpars' in f[experiment_name][spec]['bg'].attrs.keys():
            p = lm.parameter.Parameters()
            spectra_obj.bgpars = [p.loads(f[experiment_name][spec]['bg'].attrs['bgpars'][...][i]) for i in range(len(f[experiment_name][spec]['bg'].attrs['bgpars']))]
    except:
        print('couldnt bgpars')
        pass
    # bg_info
    try:
        spectra_obj.bg_info = json.loads(f[experiment_name].attrs['bg_info'])[spec]
        spectra_obj.bg_info[0] = tuple(spectra_obj.bg_info[0])
    except:
        print('couldnt bginfo')
        pass


    ##################################
    # fit_results
    try:
        spectra_obj.fit_results = [[] for i in range(len(f[experiment_name][spec]['fit_results'][...]))]
        for i in range(len(spectra_obj.fit_results)):
            params = lm.parameter.Parameters()
            modres = lm.model.ModelResult(lm.model.Model(lambda x: x, None), params)
            spectra_obj.fit_results[i] = modres.loads(f[experiment_name][spec]['fit_results'][...][i])
    except:
        pass


    ##################################
    # mod
    m = lm.model.Model(lambda x: x)
#         sample_obj.__dict__[spec].mod = [m.loads(f[experiment_name][spec]['mod'][...][i]) for i in range(len(f[experiment_name][spec]['mod'][...]))]
    try:
        spectra_obj.mod = m.loads(f[experiment_name][spec]['mod'][...][0])
    except:
        print('couldnt open mod')
        pass            
    
    # params
    p = lm.parameter.Parameters()
    try:
        spectra_obj.params = [p.loads(f[experiment_name][spec]['mod'].attrs['params'][i]) for i in range(len(f[experiment_name][spec]['mod'].attrs['params']))][0]
    except:
        print('couldnt open params')
        pass

    # Attributes
    # element_ctrl
    try:
        spectra_obj.element_ctrl = f[experiment_name][spec]['mod'].attrs['element_ctrl']
    except:
        pass
    # orbital
    try:
        spectra_obj.orbital = f[experiment_name][spec]['mod'].attrs['orbital']
    except:
        pass
    try:
        spectra_obj.orbital = spec
    except:
        pass
    # pairlist
    try:
        spectra_obj.pairlist = f[experiment_name][spec]['mod'].attrs['pairlist']
    except:
        pass

    # prefixlist
    try:
        spectra_obj.prefixlist = list(f[experiment_name][spec]['mod'].attrs['prefixlist'])
    except:
        pass

    ##################################
    # thickness
    try:
        spectra_obj.thickness = f[experiment_name][spec]['thickness'][...]
    except:
        pass
    
    try:
        tempdict = json.loads(f[experiment_name][spec].attrs['oxide_thickness'])
        spectra_obj.oxide_thickness = {oxide:np.array(tempdict[oxide]) for oxide in tempdict.keys()}
    except:
        pass     
    try:
        spectra_obj.oxide_thickness_err = json.loads(f[experiment_name][spec].attrs['oxide_thickness_err'])
    except:
        pass
    #I changed up where oxide thickness is stored, so until i fix it for all samples I have to do this...
    try:
        tempdict = json.loads(f[experiment_name][spec]['thickness'].attrs['oxide_thickness'])
        spectra_obj.oxide_thickness = {oxide:np.array(tempdict[oxide]) for oxide in tempdict.keys()}
    except:
        pass     
    try:
        spectra_obj.oxide_thickness_err = json.loads(f[experiment_name][spec]['thickness'].attrs['oxide_thickness_err'])
    except:
        pass    
    

    # if openhdf5 ==False:
    f.close()

    return spectra_obj
