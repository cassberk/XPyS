import sys
sys.path.append("/Users/cassberk/code/xps_peakfit/")
import pandas as pd

datarepo = {
    'stoqd':'/Volumes/GoogleDrive/Shared drives/StOQD/sample_library',
    'mahmut':'/Volumes/GoogleDrive/Shared drives/Quantum Coherence - Mahmut Sami Kavrik/Samples'
    }

# This background subtraction dictrionary is set up as follows:
# the element is the key, then the list elements are
# 1. background subtraction limits
# 2. type of backgroudn subtraction
# if 2 is UT2 then
# 3. the starting parameters for the UT2 filt
# 4. the indices to fit
bkgrd_subtraction = {
    'Si2p': [(100, 106.5), 'linear'],
    'Ti2p': [(452.5, 470), 'UT2', (681, 1, 355, 0), (0, -1)],
    'O1s': [(528, 538.5), 'shirley'],
    'N1s': [(395, 401), 'shirley'],
    'C1s': [(281.5, 294), 'linear'],
    'F1s': [(682, 693), 'linear'],
    'Nb3d': [(201, 213.5), 'shirley'],
    'Al2p': [(71, 79), 'shirley'],
    'Valence': [(0, 0), 'shirley'],
    'XPS': [(0, 0), 'shirley'],
    'Survey': [(0, 0), 'shirley']
    }


def spectra_colors():
    with open("/Users/cassberk/code/xps_peakfit/configuration/spectra_colors.txt", "r") as f:
        spectra_colors = {line.replace('\n','').split()[0]:line.replace('\n','').split()[1] for line in f.readlines() }
        return spectra_colors

def avantage_sensitivity_factors():
    dfSF = pd.read_excel(r'~/code/xps_peakfit/configuration/Sensitivity_Factors.xlsx', keep_default_na = False)#Do this better
    return {dfSF['element'][dfSF['SF'] != ''].iloc[i] : dfSF['SF'][dfSF['SF'] != ''].iloc[i] for i in range(len(dfSF[dfSF['SF'] != '']))}