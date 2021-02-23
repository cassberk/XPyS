import h5py
import os
import re
import numpy as np
import glob
import json
import os
from IPython import embed as shell

def dumper(obj):
    try:
        return obj.toJSON()
    except:
        return obj.tolist()



def read_avg(filepath):

    with open(filepath, encoding="utf8", errors='ignore') as f:
        lines = [line for line in f.readlines() if line != '\n']

    print(filepath)
    AXESDICT = {}
    PROPERTIES = {}
    DATA = {}
    DATA['I'] = []
    i=0
    while i < len(lines)-1:
        if 'Dump of DataSpace' in lines[i]:
            PROPERTIES.update({'VGD_Location':lines[i].split(';Dump of DataSpace')[1].strip().replace("'",'')})
            i+=1
        elif (lines[i][0] != ';') and (lines[i][0] != '\n'):

            if '$PROPERTIES' in lines[i]:
                i+=1
                while (lines[i][0] !='$'):

                    if lines[i][0] != ';':
                        props = [k.strip() for k in re.split('=|:',lines[i].strip())]
                        if props[1] =='VT_BSTR':
                            if props[0] == 'DS_EXT_SUPROPID_COMMENTS':
                                comments = props[2].replace("'","")
                                i+=1
                                while lines[i][0] == ' ':
                                    comments+=lines[i].replace("'","")
                                    i+=1
                                PROPERTIES.update({props[0].split(':')[0].strip():comments})
                            else:
                                PROPERTIES.update({props[0].split(':')[0].strip():props[2].replace("'","")})
                                i+=1
                        elif props[1] =='VT_DATE':
                            props = [k.strip() for k in re.split('=',lines[i].strip())]
                            PROPERTIES.update({props[0].split(':')[0].strip():props[1]})
                            i+=1
                        elif (props[1] =='VT_I4') or (props[1] =='VT_I2'):
                            PROPERTIES.update({props[0].split(':')[0].strip():int(props[2])})
                            i+=1
                        elif props[1] == 'VT_BOOL':
                            PROPERTIES.update({props[0].split(':')[0].strip():bool(props[2])})
                            i+=1
                        elif (props[1] =='VT_R4'):
                            PROPERTIES.update({props[0].split(':')[0].strip():np.float(props[2])})
                            i+=1
                        else:
                            PROPERTIES.update({props[0].split(':')[0].strip():props[2]})
                            print(props[1],'is a not specified dat format')
                            i+=1
                    else:
                        i+=1
                    

            elif '$SPACEAXES' in lines[i]:
                SPACEAXES = {}
                spaxes_pars = [j for j in [ k.strip() for k in re.split(',|;|=', lines[i-1].strip())] if j !=''] #Get SPACEAXES parameter names

                i+=1
                while (lines[i][0] !='$'):

                    if lines[i][0] != ';':
                        spax_vals = [re.split(',|=',lines[i].strip())[k].strip() for k in range(len(spaxes_pars))] #Get space axis parameter values
                        SPACEAXES.update({spax_vals[0]:{spaxes_pars[k]:spax_vals[k] for k in range(1,len(spax_vals))}}) #Organize into dictionary with each axis as a key
                    i+=1        

            elif '$AXISVALUE' in lines[i]:

                AXVAL = [x.strip() for x in re.split(r'\b$AXISVALUE\b|\bDATAXIS\b|\bSPACEAXIS\b|\bLABEL\b|\bPOINT\b|\bVALUE\b|=|;',lines[i].strip()) if x.strip() not in ['$AXISVALUE','']]
                try:
                    if (eval(AXVAL[2]) == 'Etch Time') or (eval(AXVAL[2]) == 'Etch Level'):
                        AXESDICT[eval(AXVAL[2])].append(np.float(AXVAL[4]))
                    elif eval(AXVAL[2]) == 'Position':
                        AXESDICT[eval(AXVAL[2])].append(eval(AXVAL[4]))
                except:
                    if (eval(AXVAL[2]) == 'Etch Time') or (eval(AXVAL[2]) == 'Etch Level'):
                        AXESDICT[eval(AXVAL[2])] = []
                        AXESDICT[eval(AXVAL[2])].append(np.float(AXVAL[4]))
                    elif eval(AXVAL[2]) == 'Position':
                        AXESDICT[eval(AXVAL[2])] = []
                        AXESDICT[eval(AXVAL[2])].append(eval(AXVAL[4]))


                AXVAL[2]
                i+=1         

            elif '$DATA=*' in lines[i]:
                data_temp = []
                i+=1
                while ('LIST@' in lines[i].split()[0]):

                    if lines[i][0] != ';':

                        data_temp.extend([np.float(k.strip()) for k in lines[i].split('=')[1].split(',')])                

                    i+=1
                    if i == len(lines):
                        break

                if int(SPACEAXES['0']['numPoints']) != len(data_temp):
                    print(lines[i-1].split()[0])
                    print(int(SPACEAXES['0']['numPoints']),len(data_temp))
                    print('data is not the same length as the numpoints')
                    break

                DATA['I'].append(data_temp)


            else:
                i+=1
        else:
            i+=1
            
    start = np.float(PROPERTIES['DS_SOPROPID_ENERGY']) - np.float(SPACEAXES['0']['start'])
    stop = np.float(PROPERTIES['DS_SOPROPID_ENERGY']) - np.float(SPACEAXES['0']['start']) -  np.float(SPACEAXES['0']['width'])*np.float(SPACEAXES['0']['numPoints'])
    interval = int(SPACEAXES['0']['numPoints'])
    
    DATA['I'] = np.array(DATA['I'])
    DATA[eval(SPACEAXES['0']['symbol'])] = np.linspace(start,stop,interval)
    
    if (int(SPACEAXES['1']['numPoints']),int(SPACEAXES['0']['numPoints'])) != np.array(DATA['I']).shape:
        
        print((int(SPACEAXES['1']['numPoints']),int(SPACEAXES['0']['numPoints'])),np.array(DATA).shape)
        print('AXES DO NOT MATCH DATA')
        return

    return PROPERTIES, SPACEAXES, AXESDICT, DATA
                
                

               
def avg_to_hdf5(sample_name,experiment_name,filepath = None,savepath=None,force = False):
    
    if filepath == None:
        filelist = glob.glob(os.getcwd()+'/*')
    elif type(filepath)==list:
        filelist = filepath
    elif type(filepath) == str:
        filelist = glob.glob(filepath+'/*')

    if savepath == None:
        save_location = '/'+os.path.join(os.path.join(*filelist[0].split('/')[0:-1]),sample_name+'.hdf5')
    else:
        save_location = savepath

    if force == False:
        hdf = h5py.File(save_location,'a')
    elif force == True:
        hdf = h5py.File(save_location,'w')
    
    if experiment_name in [grp for grp in hdf.keys()]:
        del hdf[experiment_name]
    experiment_group = hdf.require_group(experiment_name)
    element_scans = []
    all_scans = []
    for fp in filelist:
        if '.avg' in fp:

            if 'Survey' in fp.split('/')[-1].split('.')[0]:
                spectra = fp.split('/')[-1].split('.')[0].split()[1]
                all_scans.append(spectra)
                # print(spectra)
            else:
                spectra = fp.split('/')[-1].split('.')[0].split()[0]
                if spectra =='Valence':
                    all_scans.append(spectra)
                else:
                    element_scans.append(spectra)
                    all_scans.append(spectra)
                # print(spectra)
            try:
                experiment_group.attrs['data_path'] = fp
            except:
                print('datapath didnt work')
            try:
                experiment_group.attrs['sample_name'] = sample_name
            except:
                print('sample_name didnt work')

            PROPERTIES, SPACEAXES, AXESDICT, DATA = read_avg(fp)

            experiment_group.require_group(spectra)
            print(spectra)
            for props,val in PROPERTIES.items():
                experiment_group[spectra].attrs[props] = val
            print(experiment_name,spectra)
            experiment_group[spectra].create_dataset('E', data = DATA['E'])   
            experiment_group[spectra].create_dataset('I', data = DATA['I'])

            for axes,vals in AXESDICT.items():
                experiment_group[spectra]['I'].attrs[axes] = vals


            experiment_group[spectra].attrs['SPACEAXES'] = json.dumps(SPACEAXES, default=dumper, indent=2)
        
        experiment_group.attrs['element_scans'] = element_scans
        experiment_group.attrs['all_scans'] = all_scans

        
    hdf.close()

            
