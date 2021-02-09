import numpy as np
from copy import deepcopy as dc
from xps_peakfit.helper_functions import *


class autofit:
    
    def __init__(self,energy,intensity,orbital):
        self.energy = dc(energy)
        self.intensity = dc(intensity)
        self.orbital = dc(orbital)
        self.autofit_pars = self.get_autofit_pars(self.orbital)
        self.guess_params(energy = self.energy,intensity = self.intensity)


    def get_autofit_pars(self,orbital):
        if orbital == 'Nb3d':
            autofitpars_path = '/Users/cassberk/code/xps_peakfit/autofit/autofitNb.txt'
        elif orbital =='Si2p':
            autofitpars_path = '/Users/cassberk/code/xps_peakfit/autofit/autofitSi2p.txt'
        elif orbital =='C1s':
            autofitpars_path = '/Users/cassberk/code/xps_peakfit/autofit/autofitC1s.txt'
        else:
            print('No autofit yet')
            return

        f = open(autofitpars_path, "r")
        comdic = {}
        for line in f.readlines():
            # print(line)
            comdic[line.split(' ')[0]] = [l.rstrip('\n') for l in line.split(' ')[1:]]
        f.close()
        return comdic

    def guess_params(self,energy,intensity):
        
        self.energy = dc(energy)
        self.intensity = dc(intensity)
        guessamp = {}
        for par in self.autofit_pars.keys():
            
            # print(par)
            
            if self.autofit_pars[par][0] == 'lin':
                idx = index_of(self.energy, np.float(self.autofit_pars[par][1]))

                guessamp[par] = self.intensity[idx]*np.float(self.autofit_pars[par][2])
                
            elif self.autofit_pars[par][0] == 'log':
                idx = index_of(self.energy, np.float(self.autofit_pars[par][1]))
                
                guessamp[par] = np.float(self.autofit_pars[par][2])*np.log(self.intensity[idx]) + np.float(self.autofit_pars[par][3])

            elif self.autofit_pars[par][0] == 'par':
                dep_par = self.autofit_pars[par][1]
                guessamp[par] = guessamp[dep_par]*np.float(self.autofit_pars[par][2])

            elif self.autofit_pars[par][0] == 'mean':
                #  a,c = guess_from_data(self.energy,self.intensity,negative = None,peakpos = np.float(self.autofit_pars[par][1]))
                #  guessamp[par] = c
                guessamp[par] = np.float(self.autofit_pars[par][1])

        self.guess_pars = guessamp



