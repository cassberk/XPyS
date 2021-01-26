import h5py
import lmfit as lm

def load_model(model):

    f = h5py.File(model+'.hdf5','r')
    
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