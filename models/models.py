import h5py
import lmfit as lm
import os

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
        pairlist = [tuple(f.attrs['pairlist'][i]) for i in range(len(f.attrs['pairlist']))]
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