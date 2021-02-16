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
    if experiment_name in f.keys() and force == False:
        print('Experiment already exists with the same name. Set force = True to delete experiment and save a new one \
            or save the individual attribute you are interested in')
        return

    elif experiment_name in f.keys() and force == True:
        del f[experiment_name]
        experiment_group = f.create_group(experiment_name)
    
    else:
        experiment_group = f.require_group(experiment_name)

    try:
        write_vamas_to_hdf5(sample_obj.data_raw, experiment_group)
    except:
        print('Couldnt write vamas file')
        pass



    exp_attr = ('data_path','element_scans','all_scans','sample_name','positions')
    for attr in exp_attr:
        try:
            experiment_group.attrs[attr] = sample_obj.__dict__[attr]
        except:
            experiment_group.attrs[attr] = 'Not Specified'

    # total_area
    try:
        experiment_group.create_dataset('total_area', data = sample_obj.total_area)
    except:
        print('couldnt total area')
        pass
    
    # scan positions
    try:
        experiment_group.attrs['positions'] = sample_obj.positions  #prob want to change this eventually
    except:
        print('position')
        experiment_group.attrs['positions'] = 'Not Specified'
        
    #json dictionaries
    try:
        experiment_group.attrs['bg_info'] = json.dumps(sample_obj.bg_info, default=dumper, indent=2)
    except:
        print('couldnt bg_info on sampleobj')
        pass
    try:
        experiment_group.attrs['atomic_percent'] =json.dumps(sample_obj.atomic_percent, default=dumper, indent=2)
    except:
        print('couldnt atomic_percent on sample obj')
        pass

    # for spectra in sample_obj.element_scans:
    #     save_spectra(sample_obj = sample_obj,filepath = filepath, experiment_name = experiment_name,\
    #         force = force,specify_spectra = sample_obj.element_scans)
    for spectra in sample_obj.element_scans:

         # Create a new spectra group that contains all the spectra and analysis
        experiment_group.require_group(spectra)
        try:
            experiment_group[spectra].attrs['parent_sample'] = sample_obj.__dict__[spectra].__dict__['parent_sample']
        except:
            print('couldnt parent sample')
            experiment_group[spectra].attrs['parent_sample'] = 'Not Specified'


        ##################################
        # Datasets
        dsets = ('E','I','esub','isub','area','BE_adjust')
        
        for data in dsets:
            try:
                experiment_group[spectra][data] = sample_obj.__dict__[spectra].__dict__[data]
            except:
                print('couldnt',data)
                experiment_group[spectra][data] = 'Not Specified'
            
        ##################################     
        # Background subtraction
        try:
            experiment_group[spectra]['bg'] = sample_obj.__dict__[spectra].__dict__['bg']
        except:
            print('couldnt bg')
            experiment_group[spectra]['bg'] = 'Not Specified'
        # crop_info
        try:
            experiment_group[spectra]['bg'].attrs['bg_bounds'] = np.asarray(sample_obj.__dict__[spectra].__dict__['bg_info'][0])
            experiment_group[spectra]['bg'].attrs['bg_type'] = sample_obj.__dict__[spectra].__dict__['bg_info'][1]
            if sample_obj.__dict__[spectra].__dict__['bg_info'][1] =='UT2':
                experiment_group[spectra]['bg'].attrs['bg_starting_pars'] = np.asarray(sample_obj.__dict__[spectra].__dict__['bg_info'][2])
        except:
            pass

        # bgpars
        if ('bgpars' in sample_obj.__dict__[spectra].__dict__.keys()) and (any(sample_obj.__dict__[spectra].bgpars)):
            dt = h5py.special_dtype(vlen=str) 
            bgpars = np.array([sample_obj.__dict__[spectra].bgpars[i].dumps() for i in range(len(sample_obj.__dict__[spectra].bgpars))], dtype=dt) 
            experiment_group[spectra]['bg'].attrs['bgpars'] = bgpars

        ##################################
        # Fit Results
        # save lmfit spectra attributes as datasets using the lmfit method .dumps() which is a wrapper for json.dumps
        # the lmfit objects are serialied info JSON format and stored as a string array in hdf5
        dt = h5py.special_dtype(vlen=str) 
        try:
            data_temp = np.asarray([sample_obj.__dict__[spectra].__dict__['fit_results'][i].dumps()\
                    for i in range(len(sample_obj.__dict__[spectra].__dict__['fit_results']))], dtype=dt) 
            experiment_group[spectra].create_dataset('fit_results', data=data_temp)
        except:
            print('couldnt fit results')
            experiment_group[spectra].create_dataset('fit_results',data = 'Not Specified')

        ##################################
        # Model and model attributes
        # Mod
        try:
            data_temp = np.asarray([sample_obj.__dict__[spectra].__dict__['mod'].dumps()], dtype=dt) 
            experiment_group[spectra].create_dataset('mod', data=data_temp)
        except:
            print(spectra,'couldnt save','mod')
            experiment_group[spectra].create_dataset('mod',data = 'Not Specified')
            
        #Params attribute
        try:
            data_temp = np.asarray([sample_obj.__dict__[spectra].__dict__['params'].dumps()], dtype=dt) 
            experiment_group[spectra]['mod'].attrs['params'] = data_temp
        except:
            print(spectra,'couldnt save','params')
            experiment_group[spectra]['mod'].attrs['params']= 'Not Specified'

        # Other Attributes   
        attributes = ('element_ctrl','orbital','pairlist','parent_sample','prefixlist')
        for attr in attributes:
            try:
                experiment_group [spectra]['mod'].attrs[attr] = sample_obj.__dict__[spectra].__dict__[attr]
            except:
                print('couldnt',attr)
                experiment_group[spectra]['mod'].attrs[attr] = 'Not Specified'

        ##################################
        # thickness
        try:
            experiment_group[spectra].create_dataset('thickness', data = sample_obj.__dict__[spectra].thickness)
        except:
#             experiment_group[spectra].attrs['thickness'] = 'Not Specified'
            pass
        
        try:
            experiment_group[spectra].attrs['oxide_thickness'] = json.dumps(sample_obj.__dict__[spectra].oxide_thickness, default=dumper, indent=2)
        except:
            pass
#             experiment_group[spectra].attrs['oxide_thickness'] = 'Not Specified'
        try:
            experiment_group[spectra].attrs['oxide_thickness_err'] = json.dumps(sample_obj.__dict__[spectra].oxide_thickness_err, default=dumper, indent=2)
        except:
#             experiment_group[spectra].attrs['oxide_thickness_err'] = 'Not Specified'
            pass     

    f.close()


# def save_spectra(sample_obj,filepath = None, experiment_name = None,force = False,specify_spectra = None):

#     if experiment_name == None:
#         print('Must Name the experiment')
#         return
#     if specify_spectra == None:
#         print('Must specify the spectra')
#         return
    
#     f = h5py.File(filepath,'r+')
#     experiment_group = f[experiment_name]

#     if any([spec in f[experiment_name].keys() for spec in specify_spectra]) and force == False:

#         print('Spectra',[spec for spec in specify_spectra if spec in f[experiment_name].keys()],'already exist in',experiment_name,\
#             ' Set force = True to delete spectra and save a new one or save the individual attribute you are interested in')
#         f.close()
#         return

#     elif any([spec in f[experiment_name].keys() for spec in specify_spectra]) and force == True:
#         for spectra in specify_spectra:
#             del f[experiment_name][spectra]

#     for spectra in specify_spectra:

#          # Create a new spectra group that contains all the spectra and analysis
#         experiment_group.require_group(spectra)
#         try:
#             experiment_group[spectra].attrs['parent_sample'] = sample_obj.__dict__[spectra].__dict__['parent_sample']
#         except:
#             print('couldnt parent sample')
#             experiment_group[spectra].attrs['parent_sample'] = 'Not Specified'


#         ##################################
#         # Datasets
#         dsets = ('E','I','esub','isub','area','BE_adjust')
        
#         for data in dsets:
#             try:
#                 experiment_group[spectra][data] = sample_obj.__dict__[spectra].__dict__[data]
#             except:
#                 print('couldnt',data)
#                 experiment_group[spectra][data] = 'Not Specified'
            
#         ##################################     
#         # Background subtraction
#         try:
#             experiment_group[spectra]['bg'] = sample_obj.__dict__[spectra].__dict__['bg']
#         except:
#             print('couldnt bg')
#             experiment_group[spectra]['bg'] = 'Not Specified'
#         # crop_info
#         try:
#             experiment_group[spectra]['bg'].attrs['bg_bounds'] = np.asarray(sample_obj.__dict__[spectra].__dict__['bg_info'][0])
#             experiment_group[spectra]['bg'].attrs['bg_type'] = sample_obj.__dict__[spectra].__dict__['bg_info'][1]
#             if sample_obj.__dict__[spectra].__dict__['bg_info'][1] =='UT2':
#                 experiment_group[spectra]['bg'].attrs['bg_starting_pars'] = np.asarray(sample_obj.__dict__[spectra].__dict__['bg_info'][2])
#         except:
#             pass

#         # bgpars
#         if ('bgpars' in sample_obj.__dict__[spectra].__dict__.keys()) and (any(sample_obj.__dict__[spectra].bgpars)):
#             dt = h5py.special_dtype(vlen=str) 
#             bgpars = np.array([sample_obj.__dict__[spectra].bgpars[i].dumps() for i in range(len(sample_obj.__dict__[spectra].bgpars))], dtype=dt) 
#             experiment_group[spectra]['bg'].attrs['bgpars'] = bgpars

#         ##################################
#         # Fit Results
#         # save lmfit spectra attributes as datasets using the lmfit method .dumps() which is a wrapper for json.dumps
#         # the lmfit objects are serialied info JSON format and stored as a string array in hdf5
#         dt = h5py.special_dtype(vlen=str) 
#         try:
#             data_temp = np.asarray([sample_obj.__dict__[spectra].__dict__['fit_results'][i].dumps()\
#                     for i in range(len(sample_obj.__dict__[spectra].__dict__['fit_results']))], dtype=dt) 
#             experiment_group[spectra].create_dataset('fit_results', data=data_temp)
#         except:
#             print('couldnt fit results')
#             experiment_group[spectra].create_dataset('fit_results',data = 'Not Specified')

#         ##################################
#         # Model and model attributes
#         # Mod
#         try:
#             data_temp = np.asarray([sample_obj.__dict__[spectra].__dict__['mod'].dumps()], dtype=dt) 
#             experiment_group[spectra].create_dataset('mod', data=data_temp)
#         except:
#             print(spectra,'couldnt save','mod')
#             experiment_group[spectra].create_dataset('mod',data = 'Not Specified')
            
#         #Params attribute
#         try:
#             data_temp = np.asarray([sample_obj.__dict__[spectra].__dict__['params'].dumps()], dtype=dt) 
#             experiment_group[spectra]['mod'].attrs['params'] = data_temp
#         except:
#             print(spectra,'couldnt save','params')
#             experiment_group[spectra]['mod'].attrs['params']= 'Not Specified'

#         # Other Attributes   
#         attributes = ('element_ctrl','orbital','pairlist','parent_sample','prefixlist')
#         for attr in attributes:
#             try:
#                 experiment_group [spectra]['mod'].attrs[attr] = sample_obj.__dict__[spectra].__dict__[attr]
#             except:
#                 print('couldnt',attr)
#                 experiment_group[spectra]['mod'].attrs[attr] = 'Not Specified'

#         ##################################
#         # thickness
#         try:
#             experiment_group[spectra].create_dataset('thickness', data = sample_obj.__dict__[spectra].thickness)
#         except:
# #             experiment_group[spectra].attrs['thickness'] = 'Not Specified'
#             pass
        
#         try:
#             experiment_group[spectra].attrs['oxide_thickness'] = json.dumps(sample_obj.__dict__[spectra].oxide_thickness, default=dumper, indent=2)
#         except:
#             pass
# #             experiment_group[spectra].attrs['oxide_thickness'] = 'Not Specified'
#         try:
#             experiment_group[spectra].attrs['oxide_thickness_err'] = json.dumps(sample_obj.__dict__[spectra].oxide_thickness_err, default=dumper, indent=2)
#         except:
# #             experiment_group[spectra].attrs['oxide_thickness_err'] = 'Not Specified'
#             pass     
        
#     f.close()




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

    sample_obj = xps_peakfit.sample.sample(overview=False)

    f= h5py.File(filepath,"r")

    exp_attr = ('data_path','element_scans','all_scans','sample_name','positions')
    for attr in exp_attr:
        try:
            sample_obj.__dict__[attr] = f[experiment_name].attrs[attr]
        except:
            pass

    try:
        sample_obj.atomic_percent = json.loads(f[experiment_name].attrs['atomic_percent'])
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

    if sample_obj.all_scans =='Not Specified':
        print('All Scans not specified, please specify them first')
        f.close()
        return

    for spec in sample_obj.all_scans:

        print(spec)
        # Create a new group that is a spectra object
        # that contains all the spectra and analysis
        sample_obj.__dict__[spec] = load_spectra(filepath = filepath, experiment_name = experiment_name,spec = spec)
        sample_obj.__dict__[spec].spectra_name = sample_obj.sample_name

    f.close()

    return sample_obj






def load_spectra(filepath = None, experiment_name = None,spec = None, openhdf5 = False):

    spectra_obj = xps_peakfit.spectra.spectra(spectra_name = spec)
    # if openhdf5 == False:
    f= h5py.File(filepath,"r")

    ##################################
    # Datasets
    for data in ['E','I','esub','isub','area']:
        try:
            spectra_obj.__dict__[data] = f[experiment_name][spec][data][...]
        except:
            print('couldnt',data)
            pass

    try:
        spectra_obj.parent_sample = f[experiment_name][spec].attrs['parent_sample']
    except:
        print('couldnt parent_sample')
        pass

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
    
    

    # if openhdf5 ==False:
    f.close()

    return spectra_obj
