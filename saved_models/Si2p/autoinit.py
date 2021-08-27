import XPyS
import XPyS.config
from XPyS.helper_functions import guess_from_data, index_of
import os
import XPyS.config
import numpy as np

class AutoInit:
    
    def __init__(self,spectra_object):
        self.spectra_object = spectra_object
        self.E = self.spectra_object.E
        self.I = self.spectra_object.I

        self._load_autofit()


    def _load_autofit(self):
        autofitpars_path = os.path.join(XPyS.config.package_location,'XPyS/saved_models/Si2p/Si2p_BE_correllations.txt')

        f = open(autofitpars_path, "r")
        comdic = {}
        for line in f.readlines():
            comdic[line.split(' ')[0]] = [l.rstrip('\n') for l in line.split(' ')[1:]]
        f.close()
        
        self.autofit_pars = comdic

    def guess_params(self,idx):
        
        energy = self.spectra_object.esub
        intensity = self.spectra_object.isub[idx]
        guessamp = {}
        for par in self.autofit_pars.keys():
                        
            if self.autofit_pars[par][0] == 'lin':
                idx = index_of(energy, np.float(self.autofit_pars[par][1]))

                if len(self.autofit_pars[par]) == 3:
                    guessamp[par] = intensity[idx]*np.float(self.autofit_pars[par][2])
                elif len(self.autofit_pars[par]) ==4:
                    guessamp[par] = intensity[idx]*np.float(self.autofit_pars[par][2]) + np.float(self.autofit_pars[par][3])

            elif self.autofit_pars[par][0] == 'linadj':
                amp,cen = guess_from_data(x= energy, y = intensity,peakpos = np.float(self.autofit_pars[par][1]))
                idx = index_of(energy, cen)

                if len(self.autofit_pars[par]) == 3:
                    guessamp[par] = intensity[idx]*np.float(self.autofit_pars[par][2])
                elif len(self.autofit_pars[par]) ==4:
                    guessamp[par] = intensity[idx]*np.float(self.autofit_pars[par][2]) + np.float(self.autofit_pars[par][3])

            elif self.autofit_pars[par][0] == 'log':
                idx = index_of(energy, np.float(self.autofit_pars[par][1]))
                
                guessamp[par] = np.float(self.autofit_pars[par][2])*np.log(intensity[idx]) + np.float(self.autofit_pars[par][3])

            elif self.autofit_pars[par][0] == 'par':
                dep_par = self.autofit_pars[par][1]
                guessamp[par] = guessamp[dep_par]*np.float(self.autofit_pars[par][2])

            elif self.autofit_pars[par][0] == 'mean':
                #  a,c = guess_from_data(energy,intensity,negative = None,peakpos = np.float(self.autofit_pars[par][1]))
                #  guessamp[par] = c
                guessamp[par] = np.float(self.autofit_pars[par][1])

            elif self.autofit_pars[par][0] == 'guess':
                a,c = guess_from_data(energy,intensity,peakpos = np.float(self.autofit_pars[par][1]))
                print(c)
                #  guessamp[par] = c
                guessamp[par] = np.float(c)
                
        self.params = guessamp
