import numpy as np
import matplotlib.pyplot as plt
from xps_peakfit.gui_element_dicts import *
import matplotlib.patches as mpatches

def plot_atomic_percents(sample_list,idx = None, error = 'std', width = 0.8, spectra_colors = None, specify_names = None, \
    specify_spectra = None, fig = None, ax = None, capsize = 20):

    # if type(sample_list[0]) == list:

    #     lt = list(zip(sample_list[0],sample_list[1]))

    #     sample_list = [item for t in lt for item in t] 

    if (fig == None) and (ax == None):
        _fig, _ax = plt.subplots(figsize=(15,10))
    else:
        _fig = fig
        _ax = ax

    for sample in enumerate(sample_list):

        if sample[0]%2 == 1:

            fade = 0.3
        else:
            fade = 1

        

        if specify_spectra is None:
            spectra_list = sample[1].element_scans
            spectra_width = width/len(sample[1].element_scans)
        else:
            spectra_list = specify_spectra
            spectra_width = width/len(specify_spectra)
            # sample[1].calc_atomic_percent(specify_spectra = spectra_list)

        for spectra in enumerate(spectra_list):
            


            if idx is None:
                at_pct = 100*sample[1].__dict__[spectra[1]].atomic_percent.mean()
                at_pct_err = 100*sample[1].__dict__[spectra[1]].atomic_percent.std()
            else:
                at_pct = 100*sample[1].__dict__[spectra[1]].atomic_percent[idx]
                at_pct_err = 0
                
            _ax.bar(sample[0]+spectra[0]*spectra_width, at_pct, spectra_width, yerr=at_pct_err,\
                              alpha = fade,error_kw=dict(lw=3, capsize=0, capthick=3), color = spectra_colors[spectra[1]])


    
    if ax == None:
        _ax.set_xticks(np.arange(len(sample_list))+width/2)

        _ax.tick_params(labelsize = 40)
        if specify_names is None:
            xlabel_list = ['No Name' if label_list is None else label_list for label_list in [sample_list[i].sample_name for i in range(len(sample_list))]]
        else:
            xlabel_list = specify_names
        _ax.set_xticklabels(xlabel_list,rotation = 80)

        _ax.set_ylabel('Atomic Percent',fontsize=40);

            
        patchlist = []    
        for orb in spectra_list:
            patchlist.append(mpatches.Patch(color=spectra_colors[orb], label=orb))

        _fig.legend(handles=patchlist,bbox_to_anchor=(1.05, 1.), loc='upper left',fontsize = 18)
        _fig.tight_layout()
        # plt.legend(fig_list,fig_legend,bbox_to_anchor=(1.0, 0.4, 0.0, 0.5),fontsize=20)

    _ax.grid() 

    return fig, ax


def compare_atomic_percents(sample_list,idx = None, error = 'std', spectra_colors = None, specify_names = None, \
    specify_spectra = None,  ticklabels = None, capsize = 20):


    if specify_spectra is None:
        spectra_list = sample_list[0].element_scans
        spectra_width = width/len(sample_list[0].element_scans)
    else:
        spectra_list = specify_spectra
        spectra_width = width/len(specify_spectra)

    fig,ax = plt.subplots(1,len(spectra_list),figsize = (16,4))
    ax = ax.ravel()
    for orb in enumerate(spectra_list):
        f,a = xps_peakfit.compare.atomic_percent.plot_atomic_percents([e3,e4] ,spectra_colors = spectra_colors,specify_spectra = [orb[1]],\
                                                                    fig = fig, ax = ax[orb[0]])
        a.set_xticks(np.linspace(0,18,10))
        a.set_xticklabels(ticklabels,rotation = 90,fontsize = 14)

    patchlist = []    
    for orb in spectra_list:
        patchlist.append(mpatches.Patch(color=spectra_colors[orb], label=orb))

    fig.legend(handles=patchlist,bbox_to_anchor=(1.01, 0.85), loc='upper left',fontsize = 18)
    fig.tight_layout()

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    plt.xlabel('Etching Time (sec)',fontsize = 14,labelpad=35)
    plt.ylabel('Atomic Percent',fontsize = 14,labelpad=20)
