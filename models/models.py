import h5py
import json
import lmfit as lm
import os
import numpy as np

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
