import numpy as np
import matplotlib.pyplot as plt
import XPyS
import XPyS.config as cfg
# from embed import shell()

def plot_oxides(sample_spectra,set_idx = None, error = 'std', specify_names = None, width = 0.8,capsize = 20):
    """Function for comparing the oxides of different spectra objects"""


    # Check to see if any of the names repeat. If they do then the dictionaries which get build below will 
    # have their keys overwritten
    listrepeat = [spectra.spectra_name for spectra in sample_spectra]
    if (not len(listrepeat) == len(set(listrepeat))) and (specify_names == None):
        print(listrepeat)
        print('There are duplicate sample names!')
        return
    
    sample_dic_mean = {}
    sample_dic_err = {}

    # If there are no repeats the dictionary keys as well as the axis labels will just be the spectra_name attributes
    # of the samples in the list
    if specify_names == None:
        axis_names = [sample.spectra_name for sample in sample_spectra]
    else:
        axis_names = specify_names

    # There is an option to specify the indexes for a specific sample. set_idx should be a dictionary
    if set_idx == None:
        set_idx = {axis_names[i]:None for i in range(len(sample_spectra))}
    else:
        for sample in [name for name in axis_names if name not in list(set_idx.keys())]:
            set_idx[sample] = None

    for i in range(len(sample_spectra)):

        if set_idx[axis_names[i]] == None:
            
            idxs = {oxide:np.arange(len(sample_spectra[i].oxide_thickness[oxide]))[~np.isnan(sample_spectra[i].oxide_thickness[oxide])] \
                for oxide in list(sample_spectra[i].oxide_thickness.keys())}
                
        else:
            idxs = {oxide: np.array(set_idx[axis_names[i]]) for oxide in list(sample_spectra[i].oxide_thickness.keys())}

        
        sample_dic_mean[axis_names[i]] = {oxide:sample_spectra[i].oxide_thickness[oxide]\
                                                    [tuple(idxs[oxide]),].mean()\
                                                    for oxide in list(sample_spectra[i].oxide_thickness.keys())}

        if error == 'std':
            sample_dic_err[axis_names[i]] = {oxide:sample_spectra[i].oxide_thickness[oxide]\
                                                        [tuple(idxs[oxide]),].std()\
                                                        for oxide in list(sample_spectra[i].oxide_thickness.keys())}
        elif error == 'max-min':
            sample_dic_err[axis_names[i]] = {oxide:np.max(sample_spectra[i].oxide_thickness[oxide]\
                                                        [tuple(idxs[oxide]),] ) - \
                                                            np.min(sample_spectra[i].oxide_thickness[oxide]\
                                                        [tuple(idxs[oxide]),] ) 
                                                        for oxide in list(sample_spectra[i].oxide_thickness.keys())}        
        if error == 'var':
            sample_dic_err[axis_names[i]] = {oxide:sample_spectra[i].oxide_thickness[oxide]\
                                                        [tuple(idxs[oxide]),].var()\
                                                        for oxide in list(sample_spectra[i].oxide_thickness.keys())}          
        sample_spectra[i].spectra_name
        
        

    ind = np.arange(len(sample_spectra))
    if type(width) == float:
        width = width *np.ones(len(ind))
    
    fig, ax = plt.subplots(figsize=(15,10))

    p = []
    fit_legend = []
    
    plot_iter = 0
    for i in range(len(sample_spectra)):
        comps_so_far = []

        for ox in sample_spectra[i].oxide_thickness.keys():

            bottom_iter = sum(comps_so_far)

            if ox not in fit_legend:
                fit_legend.append(ox)
                p.append(ax.bar(ind[i],sample_dic_mean[axis_names[i]][ox], width[i], yerr=sample_dic_err[axis_names[i]][ox],\
                              error_kw=dict(lw=5, capsize=20, capthick=3), bottom = bottom_iter,\
                          color = cfg.element_color[ox]))
                plot_iter+=1

            else:
                ax.bar(ind[i],sample_dic_mean[axis_names[i]][ox], width[i], yerr=sample_dic_err[axis_names[i]][ox],\
                              error_kw=dict(lw=5, capsize=20, capthick=3), bottom = bottom_iter,\
                          color = cfg.element_color[ox])


            comps_so_far.append(np.asarray([sample_dic_mean[axis_names[i]][ox]]))


    ax.set_xticks(ind)
    ax.tick_params(labelsize = 40)
    ax.set_xticklabels([axis_names[i] for i in range(len(sample_spectra))],rotation = 80)
    ax.set_ylabel('Thickness (nm)',fontsize=40)
    # ax.set_xscale('log')


    plt.legend(p,[cfg.element_text[oxides] for oxides in fit_legend],bbox_to_anchor=(1.0, 0.4, 0.0, 0.5),fontsize=30)
    # plt.grid() 
    # return color_list, oxplotlist
    return fig, ax
