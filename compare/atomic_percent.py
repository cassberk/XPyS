import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import XPyS
import XPyS.config as cfg


def plot_atomic_percents(sample_list,idx = None, error = 'std', width = 0.8, spectra_colors = None, specify_names = None, \
    specify_spectra = None, fig = None, ax = None, capsize = 20):

    # if type(sample_list[0]) == list:

    #     lt = list(zip(sample_list[0],sample_list[1]))

    #     sample_list = [item for t in lt for item in t] 
    if spectra_colors is None:
        spectra_colors = cfg.spectra_colors()

    if (fig == None) and (ax == None):
        _fig, _ax = plt.subplots(figsize=(15,10))
    else:
        _fig = fig
        _ax = ax

    for sample in enumerate(sample_list):
        # print(sample[1].sample_name)
        fade = 1
        # if sample[0]%2 == 1:

        #     fade = 0.3
        # else:
        #     fade = 1

        

        if specify_spectra is None:
            spectra_list = sample[1].element_scans
            spectra_width = width/len(sample[1].element_scans)
        else:
            spectra_list = specify_spectra
            spectra_width = width/len(specify_spectra)
            # sample[1].calc_atomic_percent(specify_spectra = spectra_list)

        for spectra in enumerate(spectra_list):
            # print(spectra)
            if idx is None:
                at_pct = 100*sample[1].__dict__[spectra[1]].atomic_percent.mean()
                at_pct_err = 100*sample[1].__dict__[spectra[1]].atomic_percent.std()
            elif type(idx[0]) is list:
                at_pct = 100*sample[1].__dict__[spectra[1]].atomic_percent[idx[sample[0]][0]:idx[sample[0]][1]].mean()
                at_pct_err = 100*sample[1].__dict__[spectra[1]].atomic_percent[idx[sample[0]][0]:idx[sample[0]][1]].std()            
            elif type(idx[0]) is int:
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
        _ax.set_xticklabels(xlabel_list,rotation = 0)

        _ax.set_ylabel('Atomic Percent',fontsize=40);

            
        patchlist = []    
        for orb in spectra_list:
            patchlist.append(mpatches.Patch(color=spectra_colors[orb], label=orb))

        _fig.legend(handles=patchlist,bbox_to_anchor=(1.05, 1.), loc='upper left',fontsize = 18)
        _fig.tight_layout()
        # plt.legend(fig_list,fig_legend,bbox_to_anchor=(1.0, 0.4, 0.0, 0.5),fontsize=20)

    _ax.grid() 

    return _fig, _ax


def compare_atomic_percents(sample_list,idx = None, error = 'std', width = 0.8, spectra_colors = cfg.spectra_colors(), specify_names = None, recalc = False,\
    specify_spectra = None, capsize = 20):



    if specify_spectra is None:
        spectra_list = sample_list[0].element_scans
        spectra_width = width/len(sample_list[0].element_scans)
    else:
        spectra_list = specify_spectra
        spectra_width = width/len(specify_spectra)
        # sample[1].calc_atomic_percent(specify_spectra = spectra_list)
        
    if recalc == True:
        for sample in sample_list:
            sample.calc_atomic_percent(specify_spectra = spectra_list)
    fig,ax = plt.subplots(1,len(spectra_list),figsize = (16,4))
    ax = ax.ravel()
    
    for spectra in enumerate(spectra_list):

        if idx is None:
            at_pct = [100*sample.__dict__[spectra[1]].atomic_percent.mean() for sample in sample_list]
            at_pct_err = [100*sample.__dict__[spectra[1]].atomic_percent.std() for sample in sample_list]
        elif type(idx[0]) is list:
            at_pct = [100*sample[1].__dict__[spectra[1]].atomic_percent[idx[sample[0]][0]:idx[sample[0]][1]].mean() for sample in enumerate(sample_list)]
            at_pct_err = [100*sample[1].__dict__[spectra[1]].atomic_percent[idx[sample[0]][0]:idx[sample[0]][1]].std() for sample in enumerate(sample_list)]
        elif type(idx[0]) is int:
            at_pct = [100*sample[1].__dict__[spectra[1]].atomic_percent[idx[sample[0]]] for sample in enumerate(sample_list)]
            at_pct_err = 0
        
        ax[spectra[0]].bar(np.arange(len(at_pct)), at_pct, yerr=at_pct_err,error_kw=dict(lw=3, capsize=0, capthick=3), color = spectra_colors[spectra[1]])


        ax[spectra[0]].set_xticks(np.arange(len(sample_list)))

        ax[spectra[0]].tick_params(labelsize = 20)
        if specify_names is None:
            xlabel_list = ['No Name' if label_list is None else label_list for label_list in [sample_list[i].sample_name for i in range(len(sample_list))]]
        else:
            xlabel_list = specify_names
        ax[spectra[0]].set_xticklabels(xlabel_list,rotation = 80)
        if spectra[0] == 0:
            ax[spectra[0]].set_ylabel('Atomic Percent',fontsize=20);
        ax[spectra[0]].set_ylim([0,70])
        ax[spectra[0]].grid() 

    patchlist = []    
    for orb in spectra_list:
        patchlist.append(mpatches.Patch(color=spectra_colors[orb], label=orb))

    fig.legend(handles=patchlist,bbox_to_anchor=(1.05, 1.), loc='upper left',fontsize = 18)
    fig.tight_layout()
#     plt.legend(fig_list,fig_legend,bbox_to_anchor=(1.0, 0.4, 0.0, 0.5),fontsize=20)


    return fig, ax