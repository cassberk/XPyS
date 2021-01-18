import numpy as np
import matplotlib.pyplot as plt
from xps.gui_element_dicts import *

def plot_atomic_percents(sample_list,idx = None, error = 'std', width = 0.8, spectra_colors = None, specify_names = None, specify_spectra = None, capsize = 20):


    fig, ax = plt.subplots(figsize=(15,10))

    fig_list = []
    fig_legend = []

    for sample in enumerate(sample_list):
        if type(idx) == list:
            i = idx[sample[0]]
        else:
            i = idx

        spectra_width = width/len(sample[1].element_scans)

        if specify_spectra is None:
            spectra_list = sample[1].element_scans
        else:
            spectra_list = specify_spectra
            sample[1].calc_atomic_percent(specify_spectra = spectra_list)

        for spectra in enumerate(spectra_list):
            if spectra[1] not in fig_legend:
                fig_legend.append(spectra[1])
                fig_list.append(ax.bar(sample[0]+spectra[0]*spectra_width, 100*sample[1].__dict__[spectra[1]].atomic_percent[i], spectra_width, \
                    color = spectra_colors[spectra[1]]))
            else:
                ax.bar(sample[0]+spectra[0]*spectra_width, 100*sample[1].__dict__[spectra[1]].atomic_percent[i], spectra_width, \
                    color = spectra_colors[spectra[1]])


    ax.tick_params(labelsize = 40)
    ax.set_xticks(np.arange(len(sample_list))+width/2)
    
    if specify_names is None:
        xlabel_list = ['No Name' if label_list is None else label_list for label_list in [sample_list[i].sample_name for i in range(len(sample_list))]]
    else:
        xlabel_list = specify_names
    ax.set_xticklabels(xlabel_list,rotation = 80)

    ax.set_ylabel('Atomic Percent',fontsize=40);


    plt.legend(fig_list,fig_legend,bbox_to_anchor=(1.0, 0.4, 0.0, 0.5),fontsize=20)

    plt.grid() 

    return fig, ax


