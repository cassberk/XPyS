import XPyS
import XPyS.autofit
import XPyS.config
from XPyS.autofit import SpectraModelNet
from XPyS.helper_functions import guess_from_data, index_of
import os
import XPyS.config
import numpy as np

class AutoInit:
    
    def __init__(self):
        self._load_autofit()


    def _load_autofit(self):
        autofitpars_path = os.path.join(XPyS.config.package_location,'XPyS/saved_models/O1s/O1s_BE_correllations.txt')

        f = open(autofitpars_path, "r")
        comdic = {}
        for line in f.readlines():
            # print(line)
            comdic[line.split(' ')[0]] = [l.rstrip('\n') for l in line.split(' ')[1:]]
        f.close()
        
        self.autofit_pars = comdic

    def guess_params(self,energy,intensity):

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
