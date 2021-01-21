import json
import h5py
import numpy as np

def dumper(obj):
    try:
        return obj.toJSON()
    except:
        return obj.tolist()

def save_sample(sample_obj,filepath = None, experiment_name = None):

    if experiment_name == None:
        print('Must Name the experiment')
        return

    f = h5py.File(filepath,'a')

    experiment_group = f.require_group(experiment_name)

    write_vamas_to_hdf5(sample_obj.data_raw, experiment_group)

    exp_attr = ('data_path','element_scans','all_scans','sample_name')
    for attr in exp_attr:
        try:
            experiment_group .attrs[attr] = sample_obj.__dict__[attr]
        except:
            experiment_group .attrs[attr] = 'Not Specified'

    experiment_group ['total_area'] = sample_obj.total_area

    #json dictionaries
    experiment_group.attrs['bg_info'] = json.dumps(sample_obj.bg_info, default=dumper, indent=2)
    experiment_group.attrs['atomic_percent'] = json.dumps(sample_obj.atomic_percent, default=dumper, indent=2)


    for spectra in sample_obj.element_scans:

        # Create a new spectra group that contains all the spectra and analysis
        experiment_group.require_group(spectra)

        # Datasets
        dsets = ('E','I','esub','isub','area','bg','BE_adjust')
        
        for attr in dsets:
                experiment_group [spectra][attr] = sample_obj.__dict__[spectra].__dict__[attr]

        # bgpars
        if ('bgpars' in sample_obj.__dict__[spectra].__dict__.keys()) and (any(sample_obj.__dict__[spectra].bgpars)):
            dt = h5py.special_dtype(vlen=str) 
            bgpars = np.array([sample_obj.__dict__[spectra].bgpars[i].dumps() for i in range(len(sample_obj.__dict__[spectra].bgpars))], dtype=dt) 
            experiment_group [spectra].create_dataset('bgpars', data=bgpars)


        
        # save lmfit spectra attributes as datasets using the lmfit method .dumps() which is a wrapper for json.dumps
        # the lmfit objects are serialied info JSON format and stored as a string array in hdf5

        lm_spec_attr = ('fit_results','params','mod')
        
        dt = h5py.special_dtype(vlen=str) 
        for attr in lm_spec_attr:
    #         if hasattr(sample_obj.__dict__[spectra],attr):
            try:
                data_temp = np.asarray([sample_obj.__dict__[spectra].__dict__[attr][i].dumps() for i in range(len(sample_obj.__dict__[spectra].__dict__[attr]))], dtype=dt) 
                experiment_group [spectra].create_dataset(attr, data=data_temp)
            except:
                experiment_group [spectra].create_dataset(attr,data = 'Not Specified')


        # Attributes   
        
        attributes = ('position_names','element_ctrl','orbital','pairlist','parent_sample','prefixlist')
        for attr in attributes:
            try:
                experiment_group [spectra].attrs[attr] = sample_obj.__dict__[spectra].__dict__[attr]
            except:
                experiment_group [spectra].attrs[attr] = 'Not Specified'

        # crop_info
        experiment_group [spectra].attrs['bg_bounds'] = np.asarray(sample_obj.__dict__[spectra].__dict__['bg_info'][0])
        experiment_group [spectra].attrs['bg_type'] = sample_obj.__dict__[spectra].__dict__['bg_info'][1]
        if sample_obj.__dict__[spectra].__dict__['bg_info'][1] =='UT2':
            experiment_group [spectra].attrs['bg_starting_pars'] = np.asarray(sample_obj.__dict__[spectra].__dict__['bg_info'][2])

    f.close()

    #     """fit_results"""
    #     dt = h5py.special_dtype(vlen=str) 
    #     fit_results = np.asarray([sample_obj.__dict__[spectra].fit_results[i].dumps() for i in range(len(sample_obj.__dict__[spectra].fit_results))], dtype=dt) 
    #     experiment_group [spectra].create_dataset('fit_results', data=fit_results)

    #     """fit_results_idx"""
    #     experiment_group [spectra]['fit_results_idx'] = sample_obj.__dict__[spectra].fit_results_idx

    #     """params"""
    #     dt = h5py.special_dtype(vlen=str) 
    #     params = np.array([sample_obj.__dict__[spectra].params.dumps()], dtype=dt) 
    #     experiment_group [spectra].create_dataset('params', data=params)

    #     """mod"""
    #     dt = h5py.special_dtype(vlen=str) 
    #     mod = np.array([sample_obj.__dict__[spectra].mod.dumps()], dtype=dt) 
    #     experiment_group [spectra].create_dataset('mod', data=mod)


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



def load_sample(sample_obj,filepath = None, experiment_name = None):

    f= h5py.File(filepath,"r")

    sample_obj.data_path = f[experiment_name].attrs['data_path']
    sample_obj.element_scans = f[experiment_name].attrs['element_scans']
    sample_obj.sample_name = f[experiment_name].attrs['sample_name']
    sample_obj.positions = f[experiment_name].attrs['positions']
    try:
        sample_obj.data_raw = f[experiment_name]['raw_data']['raw_vamas_data'][...]
    except:
        pass
#     sample_obj.__dict__['data'] = {}
#     sample_obj.__dict__['bg_info'] = {}

    # """Background subtraction dictionary"""
    sample_obj.bg_info = json.loads(f[experiment_name].attrs['bg_info'])

    for spec in sample_obj.element_scans:
        print(spec)
        # """Create a new group that is a spectra object
        # that contains all the spectra and analysis
        # """

        sample_obj.__dict__[spec] = xps_peakfit.spectra.spectra(spectra_name = spec)
#         sample_obj.__dict__['data'][spec] = {}
        # """Datasets"""

        # """E, I, esub, isub"""
        sample_obj.__dict__[spec].E = f[experiment_name][spec]['E'][...]
        sample_obj.__dict__[spec].I= f[experiment_name][spec]['I'][...]
        sample_obj.__dict__[spec].esub = f[experiment_name][spec]['esub'][...]
        sample_obj.__dict__[spec].isub = f[experiment_name][spec]['isub'][...]

        # """bg"""
        sample_obj.__dict__[spec].bg = f[experiment_name][spec]['bg'][...]

        # """bgpars"""
        if 'bgpars' in f[experiment_name][spec].keys():
            p = lm.parameter.Parameters()
            sample_obj.__dict__[spec].bgpars = [p.loads(f[experiment_name][spec]['bgpars'][...][i]) for i in range(len(f[experiment_name][spec]['bgpars'][...]))]


        # """area"""    
        sample_obj.__dict__[spec].area = f[experiment_name][spec]['area'][...]


        # """fit_results"""
        params = lm.parameter.Parameters()
        modres = lm.model.ModelResult(lm.model.Model(lambda x: x, None), params)
        try:
            sample_obj.__dict__[spec].fit_results = [modres.loads(f[experiment_name][spec]['fit_results'][...][i]) for i in range(len(f[experiment_name][spec]['fit_results'][...]))]
        except:
            pass

        # """fit_results_idx"""
        sample_obj.__dict__[spec].fit_results_idx = f[experiment_name][spec]['fit_results_idx'][...]

        # """params"""
        p = lm.parameter.Parameters()
        try:
            sample_obj.__dict__[spec].params = [p.loads(f[experiment_name][spec]['params'][...][i]) for i in range(len(f[experiment_name][spec]['params'][...]))][0]
        except:
            pass

        # """mod"""
        m = lm.model.Model(lambda x: x)
#         sample_obj.__dict__[spec].mod = [m.loads(f[experiment_name][spec]['mod'][...][i]) for i in range(len(f[experiment_name][spec]['mod'][...]))]
        try:
            sample_obj.__dict__[spec].mod = m.loads(f[experiment_name][spec]['mod'][...][0])
        except:
            pass
        # """BE_adjust"""
        sample_obj.__dict__[spec].BE_adjust = f[experiment_name][spec]['BE_adjust']

        
                
#         thickness
        try:
            sample_obj.__dict__[spec].thickness = f[experiment_name][spec]['thickness'][...]
        except:
            pass
        
        try:
            sample_obj.__dict__[spec].oxide_thickness = json.loads(f[experiment_name][spec].attrs['oxide_thickness'])
        except:
            pass     
        
        try:
            sample_obj.__dict__[spec].oxide_thickness_err = json.loads(f[experiment_name][spec].attrs['oxide_thickness_err'])
        except:
            pass
        
        
        
        
        # """Attributes"""
        # """element_ctrl"""
        sample_obj.__dict__[spec].element_ctrl = f[experiment_name][spec].attrs['element_ctrl']

        # """orbital"""
        sample_obj.__dict__[spec].orbital = f[experiment_name][spec].attrs['orbital']

        # """pairlist"""
        sample_obj.__dict__[spec].pairlist = f[experiment_name][spec].attrs['pairlist']

        # """parent_sample"""
#         sample_obj.__dict__[spec].parent_sample = f[experiment_name][spec].attrs['parent_sample']
        sample_obj.__dict__[spec].parent_sample = f[experiment_name].attrs['sample_name']
        # """prefixlist"""
        sample_obj.__dict__[spec].prefixlist = list(f[experiment_name][spec].attrs['prefixlist'])

        # """bg_info"""
        sample_obj.__dict__[spec].bg_info = sample_obj.bg_info[spec]
    
    # sample_obj.bkgrd_sub_dict[spec] = [tuple(f['XPS1'][spec].attrs['bkgrd_sub_bounds']),f['XPS1'][spec].attrs['bkgrd_sub_type']]
    # if 'bkgrd_sub_starting_pars' in f['XPS1'][spec].attrs:
    #     sample_obj.bkgrd_sub_dict[spec].append(tuple(f['XPS1'][spec].attrs['bkgrd_sub_starting_pars']))
    #     print('in')

    f.close()
