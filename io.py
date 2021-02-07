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

    # Path(os.path.join(*filepath.split('/')[:-1])).mkdir(parents=True, exist_ok=True)
    try:
        os.makedirs('/'.join(filepath.split('/')[:-1]))
    except OSError as e:
        print(e)
        print('use force')
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
    experiment_group.create_dataset('total_area', data = sample_obj.total_area)

    
    # scan positions
    try:
        experiment_group.attrs['positions'] = sample_obj.positions  #prob want to change this eventually
    except:
        experiment_group.attrs['positions'] = 'Not Specified'
        
    #json dictionaries
    experiment_group.attrs['bg_info'] = json.dumps(sample_obj.bg_info, default=dumper, indent=2)
    experiment_group.attrs['atomic_percent'] =json.dumps(sample_obj.atomic_percent, default=dumper, indent=2)

    for spectra in sample_obj.element_scans:

        # Create a new spectra group that contains all the spectra and analysis
        experiment_group.require_group(spectra)

        # Datasets
        dsets = ('E','I','esub','isub','area','bg','BE_adjust')
        
        for attr in dsets:
            try:
                experiment_group[spectra][attr] = sample_obj.__dict__[spectra].__dict__[attr]
            except:
                experiment_group[spectra][attr] = 'Not Specified'

        # bgpars
        if ('bgpars' in sample_obj.__dict__[spectra].__dict__.keys()) and (any(sample_obj.__dict__[spectra].bgpars)):
            dt = h5py.special_dtype(vlen=str) 
            bgpars = np.array([sample_obj.__dict__[spectra].bgpars[i].dumps() for i in range(len(sample_obj.__dict__[spectra].bgpars))], dtype=dt) 
            experiment_group[spectra].create_dataset('bgpars', data=bgpars)


        
        # save lmfit spectra attributes as datasets using the lmfit method .dumps() which is a wrapper for json.dumps
        # the lmfit objects are serialied info JSON format and stored as a string array in hdf5
        
        dt = h5py.special_dtype(vlen=str) 
        try:
            data_temp = np.asarray([sample_obj.__dict__[spectra].__dict__['fit_results'][i].dumps()\
                    for i in range(len(sample_obj.__dict__[spectra].__dict__['fit_results']))], dtype=dt) 
            experiment_group[spectra].create_dataset('fit_results', data=data_temp)
        except:
            experiment_group[spectra].create_dataset('fit_results',data = 'Not Specified')


        lm_spec_attr = ('params','mod')
        for attr in lm_spec_attr:
            try:
                data_temp = np.asarray([sample_obj.__dict__[spectra].__dict__[attr].dumps()], dtype=dt) 
                experiment_group[spectra].create_dataset(attr, data=data_temp)
            except:
                print(spectra,'couldnt save',attr)
                experiment_group [spectra].create_dataset(attr,data = 'Not Specified')


        # Attributes   
        attributes = ('element_ctrl','orbital','pairlist','parent_sample','prefixlist')
        for attr in attributes:
            try:
                experiment_group [spectra].attrs[attr] = sample_obj.__dict__[spectra].__dict__[attr]
            except:
                experiment_group [spectra].attrs[attr] = 'Not Specified'

        # crop_info
        experiment_group[spectra].attrs['bg_bounds'] = np.asarray(sample_obj.__dict__[spectra].__dict__['bg_info'][0])
        experiment_group[spectra].attrs['bg_type'] = sample_obj.__dict__[spectra].__dict__['bg_info'][1]
        if sample_obj.__dict__[spectra].__dict__['bg_info'][1] =='UT2':
            experiment_group[spectra].attrs['bg_starting_pars'] = np.asarray(sample_obj.__dict__[spectra].__dict__['bg_info'][2])


        # thickness
        try:
            experiment_group[spectra].create_dataset('thickness', data = sample_obj.__dict__[spectra].thickness)
        except:
            experiment_group[spectra].attrs['thickness'] = 'Not Specified'
            pass
        
        try:
            experiment_group[spectra].attrs['oxide_thickness'] = json.dumps(sample_obj.__dict__[spectra].oxide_thickness, default=dumper, indent=2)
        except:
            experiment_group[spectra].attrs['oxide_thickness'] = 'Not Specified'
        try:
            experiment_group[spectra].attrs['oxide_thickness_err'] = json.dumps(sample_obj.__dict__[spectra].oxide_thickness_err, default=dumper, indent=2)
        except:
            experiment_group[spectra].attrs['oxide_thickness_err'] = 'Not Specified'
            pass     
        

    f.close()



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

    sample_obj.data_path = f[experiment_name].attrs['data_path']
    sample_obj.element_scans = f[experiment_name].attrs['element_scans']
    sample_obj.sample_name = f[experiment_name].attrs['sample_name']
    sample_obj.positions = f[experiment_name].attrs['positions']
    try:
        sample_obj.atomic_percent = json.loads(f[experiment_name].attrs['atomic_percent'])
    except:
        pass

    try:
        sample_obj.data_raw = f[experiment_name]['raw_data']['raw_vamas_data'][...]
    except:
        pass


    # Background subtraction dictionary
    sample_obj.bg_info = json.loads(f[experiment_name].attrs['bg_info'])

    sample_obj.total_area = f[experiment_name]['total_area'][...]
    # sample_obj.atomic_percent = json.loads(f[experiment_name]['atomic_percent'][...],dtype = h5py.special_dtype(vlen=str))

    for spec in sample_obj.element_scans:
        print(spec)
        # Create a new group that is a spectra object
        # that contains all the spectra and analysis

        sample_obj.__dict__[spec] = xps_peakfit.spectra.spectra(spectra_name = spec)
        sample_obj.__dict__[spec].spectra_name = sample_obj.sample_name
        # Datasets
        # E, I, esub, isub
        sample_obj.__dict__[spec].E = f[experiment_name][spec]['E'][...]
        sample_obj.__dict__[spec].I= f[experiment_name][spec]['I'][...]
        sample_obj.__dict__[spec].esub = f[experiment_name][spec]['esub'][...]
        sample_obj.__dict__[spec].isub = f[experiment_name][spec]['isub'][...]

        # bg
        sample_obj.__dict__[spec].bg = f[experiment_name][spec]['bg'][...]

        # bgpars
        if 'bgpars' in f[experiment_name][spec].keys():
            p = lm.parameter.Parameters()
            sample_obj.__dict__[spec].bgpars = [p.loads(f[experiment_name][spec]['bgpars'][...][i]) for i in range(len(f[experiment_name][spec]['bgpars'][...]))]


        # area 
        sample_obj.__dict__[spec].area = f[experiment_name][spec]['area'][...]


        # fit_results
        try:
            sample_obj.__dict__[spec].fit_results = [[] for i in range(len(f[experiment_name][spec]['fit_results'][...]))]
            for i in range(len(sample_obj.__dict__[spec].fit_results)):
                params = lm.parameter.Parameters()
                modres = lm.model.ModelResult(lm.model.Model(lambda x: x, None), params)
                sample_obj.__dict__[spec].fit_results[i] = modres.loads(f[experiment_name][spec]['fit_results'][...][i])
        except:
            pass

        # fit_results_idx (I dont think I need this anymore, but will keep it for a bit just to be safe)
        # try:
        #     sample_obj.__dict__[spec].fit_results_idx = f[experiment_name][spec]['fit_results_idx'][...]
        # except:
        #     print('couldnt open fit_results_idx')
        #     pass

        # params
        p = lm.parameter.Parameters()
        try:
            sample_obj.__dict__[spec].params = [p.loads(f[experiment_name][spec]['params'][...][i]) for i in range(len(f[experiment_name][spec]['params'][...]))][0]
        except:
            print('couldnt open params')
            pass

        # mod
        m = lm.model.Model(lambda x: x)
#         sample_obj.__dict__[spec].mod = [m.loads(f[experiment_name][spec]['mod'][...][i]) for i in range(len(f[experiment_name][spec]['mod'][...]))]
        try:
            sample_obj.__dict__[spec].mod = m.loads(f[experiment_name][spec]['mod'][...][0])
        except:
            print('couldnt open mod')
            pass

        # BE_adjust
        sample_obj.__dict__[spec].BE_adjust = f[experiment_name][spec]['BE_adjust']

        
                
#         thickness
        try:
            sample_obj.__dict__[spec].thickness = f[experiment_name][spec]['thickness'][...]
        except:
            pass
        
        try:
            tempdict = json.loads(f[experiment_name][spec].attrs['oxide_thickness'])
            sample_obj.__dict__[spec].oxide_thickness = {oxide:np.array(tempdict[oxide]) for oxide in tempdict.keys()}
        except:
            pass     
        
        try:
            sample_obj.__dict__[spec].oxide_thickness_err = json.loads(f[experiment_name][spec].attrs['oxide_thickness_err'])
        except:
            pass
        
        
        # Attributes
        # element_ctrl
        sample_obj.__dict__[spec].element_ctrl = f[experiment_name][spec].attrs['element_ctrl']

        # orbital
        sample_obj.__dict__[spec].orbital = f[experiment_name][spec].attrs['orbital']

        # pairlist
        sample_obj.__dict__[spec].pairlist = f[experiment_name][spec].attrs['pairlist']

        # parent_sample
        sample_obj.__dict__[spec].parent_sample = f[experiment_name].attrs['sample_name']

        # prefixlist
        sample_obj.__dict__[spec].prefixlist = list(f[experiment_name][spec].attrs['prefixlist'])

        # bg_info
        sample_obj.__dict__[spec].bg_info = sample_obj.bg_info[spec]


    f.close()

    return sample_obj



def save_spectra(sample_obj,filepath = None, experiment_name = None,force = False,specify_spectra = None):

    if experiment_name == None:
        print('Must Name the experiment')
        return
    if specify_spectra == None:
        print('Must specify the spectra')
        return
    
    f = h5py.File(filepath,'r+')
    experiment_group = f[experiment_name]

    if any([spec in f[experiment_name].keys() for spec in specify_spectra]) and force == False:

        print('Spectra',[spec for spec in specify_spectra if spec in f[experiment_name].keys()],'already exist in',experiment_name,\
            ' Set force = True to delete spectra and save a new one or save the individual attribute you are interested in')
        f.close()
        return

    elif any([spec in f[experiment_name].keys() for spec in specify_spectra]) and force == True:
        for spectra in specify_spectra:
            del f[experiment_name][spectra]
    #     experiment_group = f.create_group(experiment_name)
    
    # else:
    #     experiment_group = f.require_group(experiment_name)

    # try:
    #     write_vamas_to_hdf5(sample_obj.data_raw, experiment_group)
    # except:
    #     pass

    # exp_attr = ('data_path','element_scans','all_scans','sample_name','positions')
    # for attr in exp_attr:
    #     try:
    #         experiment_group.attrs[attr] = sample_obj.__dict__[attr]
    #     except:
    #         experiment_group.attrs[attr] = 'Not Specified'

    # # total_area
    # experiment_group.create_dataset('total_area', data = sample_obj.total_area)

    
    # # scan positions
    # try:
    #     experiment_group.attrs['positions'] = sample_obj.positions  #prob want to change this eventually
    # except:
    #     experiment_group.attrs['positions'] = 'Not Specified'
        
    # #json dictionaries
    # experiment_group.attrs['bg_info'] = json.dumps(sample_obj.bg_info, default=dumper, indent=2)
    # experiment_group.attrs['atomic_percent'] =json.dumps(sample_obj.atomic_percent, default=dumper, indent=2)

    for spectra in specify_spectra:

        # Create a new spectra group that contains all the spectra and analysis
        experiment_group.require_group(spectra)

        # Datasets
        dsets = ('E','I','esub','isub','area','bg','BE_adjust')
        
        for attr in dsets:
            try:
                experiment_group[spectra][attr] = sample_obj.__dict__[spectra].__dict__[attr]
            except:
                experiment_group[spectra][attr] = 'Not Specified'

        # bgpars
        if ('bgpars' in sample_obj.__dict__[spectra].__dict__.keys()) and (any(sample_obj.__dict__[spectra].bgpars)):
            dt = h5py.special_dtype(vlen=str) 
            bgpars = np.array([sample_obj.__dict__[spectra].bgpars[i].dumps() for i in range(len(sample_obj.__dict__[spectra].bgpars))], dtype=dt) 
            experiment_group[spectra].create_dataset('bgpars', data=bgpars)


        
        # save lmfit spectra attributes as datasets using the lmfit method .dumps() which is a wrapper for json.dumps
        # the lmfit objects are serialied info JSON format and stored as a string array in hdf5
        
        dt = h5py.special_dtype(vlen=str) 
        try:
            data_temp = np.asarray([sample_obj.__dict__[spectra].__dict__['fit_results'][i].dumps()\
                    for i in range(len(sample_obj.__dict__[spectra].__dict__['fit_results']))], dtype=dt) 
            experiment_group[spectra].create_dataset('fit_results', data=data_temp)
        except:
            experiment_group[spectra].create_dataset('fit_results',data = 'Not Specified')


        lm_spec_attr = ('params','mod')
        for attr in lm_spec_attr:
            try:
                data_temp = np.asarray([sample_obj.__dict__[spectra].__dict__[attr].dumps()], dtype=dt) 
                experiment_group[spectra].create_dataset(attr, data=data_temp)
            except:
                print(spectra,'couldnt save',attr)
                experiment_group [spectra].create_dataset(attr,data = 'Not Specified')


        # Attributes   
        attributes = ('element_ctrl','orbital','pairlist','parent_sample','prefixlist')
        for attr in attributes:
            try:
                experiment_group [spectra].attrs[attr] = sample_obj.__dict__[spectra].__dict__[attr]
            except:
                experiment_group [spectra].attrs[attr] = 'Not Specified'

        # crop_info
        experiment_group[spectra].attrs['bg_bounds'] = np.asarray(sample_obj.__dict__[spectra].__dict__['bg_info'][0])
        experiment_group[spectra].attrs['bg_type'] = sample_obj.__dict__[spectra].__dict__['bg_info'][1]
        if sample_obj.__dict__[spectra].__dict__['bg_info'][1] =='UT2':
            experiment_group [spectra].attrs['bg_starting_pars'] = np.asarray(sample_obj.__dict__[spectra].__dict__['bg_info'][2])

    f.close()


def load_spectra(filepath = None, experiment_name = None,spec = None):

    spectra_obj = xps_peakfit.spectra.spectra(spectra_name = spec)

    f= h5py.File(filepath,"r")

    # print(spec)

    # Datasets
    # E, I, esub, isub
    spectra_obj.E = f[experiment_name][spec]['E'][...]
    spectra_obj.I= f[experiment_name][spec]['I'][...]
    spectra_obj.esub = f[experiment_name][spec]['esub'][...]
    spectra_obj.isub = f[experiment_name][spec]['isub'][...]

    # bg
    spectra_obj.bg = f[experiment_name][spec]['bg'][...]

    # bgpars
    if 'bgpars' in f[experiment_name][spec].keys():
        p = lm.parameter.Parameters()
        spectra_obj.bgpars = [p.loads(f[experiment_name][spec]['bgpars'][...][i]) for i in range(len(f[experiment_name][spec]['bgpars'][...]))]


    # area 
    spectra_obj.area = f[experiment_name][spec]['area'][...]


    # fit_results
    try:
        spectra_obj.fit_results = [[] for i in range(len(f[experiment_name][spec]['fit_results'][...]))]
        for i in range(len(spectra_obj.fit_results)):
            params = lm.parameter.Parameters()
            modres = lm.model.ModelResult(lm.model.Model(lambda x: x, None), params)
            spectra_obj.fit_results[i] = modres.loads(f[experiment_name][spec]['fit_results'][...][i])
    except:
        pass

    # fit_results_idx (I dont think I need this anymore, but will keep it for a bit just to be safe)
    # try:
    #     spectra_obj.fit_results_idx = f[experiment_name][spec]['fit_results_idx'][...]
    # except:
    #     print('couldnt open fit_results_idx')
    #     pass

    # params
    p = lm.parameter.Parameters()
    try:
        spectra_obj.params = [p.loads(f[experiment_name][spec]['params'][...][i]) for i in range(len(f[experiment_name][spec]['params'][...]))][0]
    except:
        print('couldnt open params')
        pass

    # mod
    m = lm.model.Model(lambda x: x)
#         spectra_obj.mod = [m.loads(f[experiment_name][spec]['mod'][...][i]) for i in range(len(f[experiment_name][spec]['mod'][...]))]
    try:
        spectra_obj.mod = m.loads(f[experiment_name][spec]['mod'][...][0])
    except:
        print('couldnt open mod')
        pass

    # BE_adjust
    spectra_obj.BE_adjust = f[experiment_name][spec]['BE_adjust']

    
            
#         thickness
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
    
    
    # Attributes
    # element_ctrl
    spectra_obj.element_ctrl = f[experiment_name][spec].attrs['element_ctrl']

    # orbital
    spectra_obj.orbital = f[experiment_name][spec].attrs['orbital']

    # pairlist
    spectra_obj.pairlist = f[experiment_name][spec].attrs['pairlist']

    # parent_sample
#         spectra_obj.parent_sample = f[experiment_name][spec].attrs['parent_sample']
    spectra_obj.parent_sample = f[experiment_name].attrs['sample_name']

    # prefixlist
    spectra_obj.prefixlist = list(f[experiment_name][spec].attrs['prefixlist'])

    # bg_info
    spectra_obj.bg_info = json.loads(f[experiment_name].attrs['bg_info'])[spec]
    spectra_obj.bg_info[0] = tuple(spectra_obj.bg_info[0])

    f.close()

    return spectra_obj