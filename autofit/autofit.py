import numpy as np
from xps_peakfit.helper_functions import guess_from_data


def autofit(energy,intensity,orbital):
    if orbital == 'Nb3d':
        return Nb_autofit(energy = energy,intensity = intensity,\
            autofitpars_path = '/Users/cassberk/code/xps_peakfit/autofit/autofitNb.txt')
    else:
        print('No autofit for that spectra')
        return

def get_autofit_pars(autofitpars_path):
    f = open(autofitpars_path, "r")
    comdic = {}
    for line in f.readlines():
    #     print(line)
        comdic[line.split(' ')[0]] = []
        comdic[line.split(' ')[0]].append(np.float(line.split(' ')[1]))
        if len(line.split(' ')) >2:
                comdic[line.split(' ')[0]].append(np.float(line.split(' ')[2]))
    f.close()
    return comdic


def Nb_autofit(energy,intensity,autofitpars_path):

    nb_peak,nb_cen = guess_from_data(energy,intensity,negative = None,peakpos = 202.2)
    nb2o5_peak,nb2o5_cen = guess_from_data(energy,intensity,negative = None,peakpos = 207.5)

    comdic = get_autofit_pars(autofitpars_path)
    guessdic = {}
    guessdic['Nb_52_amplitude'] = comdic['Nb_52_amplitude'][0]*nb_peak
    
    guessdic['Nb2O5_52_amplitude'] = comdic['Nb2O5_52_amplitude'][0]*nb2o5_peak
    
    guessdic['NbO_52_amplitude'] = comdic['NbO_52_amplitude'][0]*np.log(nb_peak) + comdic['NbO_52_amplitude'][1]
    
    guessdic['NbO2_52_amplitude'] = comdic['NbO2_52_amplitude'][0]*guessdic['NbO_52_amplitude']

    return guessdic