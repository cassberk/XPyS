import numpy as np
import matplotlib.pyplot as plt
from xps.gui_element_dicts import *

def plot_oxides(sample_spectra,set_idx = None, error = 'std', width = 0.8,capsize = 20):

    # listrepeat = [spectra.spectra_name for spectra in sample_spectra]
    # print(listrepeat)
    # if not len(listrepeat) == len(set(listrepeat)):
    #     print('There are duplicate sample names!')
    #     return
    
    sample_dic_mean = {}
    sample_dic_err = {}


    for i in range(len(sample_spectra)):
        if hasattr(sample_spectra[i],'adjusted_oxide_percentage'):
            sample_dic_mean[sample_spectra[i].spectra_name] = {oxide:sample_spectra[i].adjusted_oxide_percentage[oxide]\
                                                        [~np.isnan(sample_spectra[i].adjusted_oxide_percentage[oxide])].mean()\
                                                        for oxide in list(sample_spectra[i].adjusted_oxide_percentage.keys())}

            sample_dic_err[sample_spectra[i].spectra_name] = {oxide:sample_spectra[i].adjusted_oxide_percentage[oxide]\
                                                            [~np.isnan(sample_spectra[i].adjusted_oxide_percentage[oxide])].std()\
                                                            for oxide in list(sample_spectra[i].adjusted_oxide_percentage.keys())}
        
        elif hasattr(sample_spectra[i],'adjusted_oxide_thickness'):
       
            sample_dic_mean[sample_spectra[i].spectra_name] = {oxide:sample_spectra[i].adjusted_oxide_thickness[oxide]\
                                                        [~np.isnan(sample_spectra[i].adjusted_oxide_thickness[oxide])].mean()\
                                                        for oxide in list(sample_spectra[i].adjusted_oxide_thickness.keys())}

            if error == 'std':
                sample_dic_err[sample_spectra[i].spectra_name] = {oxide:sample_spectra[i].adjusted_oxide_thickness[oxide]\
                                                         [~np.isnan(sample_spectra[i].adjusted_oxide_thickness[oxide])].std()\
                                                         for oxide in list(sample_spectra[i].adjusted_oxide_thickness.keys())}
            elif error == 'max-min':
                sample_dic_err[sample_spectra[i].spectra_name] = {oxide:np.max(sample_spectra[i].adjusted_oxide_thickness[oxide]\
                                                         [~np.isnan(sample_spectra[i].adjusted_oxide_thickness[oxide])] ) - \
                                                             np.min(sample_spectra[i].adjusted_oxide_thickness[oxide]\
                                                         [~np.isnan(sample_spectra[i].adjusted_oxide_thickness[oxide])] ) 
                                                         for oxide in list(sample_spectra[i].adjusted_oxide_thickness.keys())}        
            if error == 'var':
                sample_dic_err[sample_spectra[i].spectra_name] = {oxide:sample_spectra[i].adjusted_oxide_thickness[oxide]\
                                                         [~np.isnan(sample_spectra[i].adjusted_oxide_thickness[oxide])].var()\
                                                         for oxide in list(sample_spectra[i].adjusted_oxide_thickness.keys())}          
        sample_spectra[i].spectra_name
        
        
    if set_idx == None:
        ind = np.arange(len(sample_spectra))
    else:
        ind = set_idx
        
#     width = 0.45
    
    fig, ax = plt.subplots(figsize=(15,10))

    p = []
    fit_legend = []

    plot_iter = 0
    for sample in enumerate(sample_spectra):

        comps_so_far = []

        if hasattr(sample[1],'adjusted_oxide_percentage'):
            oxplotlist = list(sample[1].adjusted_oxide_percentage.keys())

        elif hasattr(sample[1],'adjusted_oxide_thickness'):
            oxplotlist = list(sample[1].adjusted_oxide_thickness.keys())

        for ox in oxplotlist:

            bottom_iter = sum(comps_so_far)

            if ox not in fit_legend:
                fit_legend.append(ox)
                p.append(ax.bar(ind[sample[0]],sample_dic_mean[sample[1].spectra_name][ox], width, yerr=sample_dic_err[sample[1].spectra_name][ox],\
                              error_kw=dict(lw=5, capsize=20, capthick=3), bottom = bottom_iter,\
                          color = element_color[ox]))
                plot_iter+=1

            else:
                ax.bar(ind[sample[0]],sample_dic_mean[sample[1].spectra_name][ox], width, yerr=sample_dic_err[sample[1].spectra_name][ox],\
                              error_kw=dict(lw=5, capsize=20, capthick=3), bottom = bottom_iter,\
                          color = element_color[ox])


            comps_so_far.append(np.asarray([sample_dic_mean[sample[1].spectra_name][ox]]))


    ax.set_xticks(ind)
    ax.tick_params(labelsize = 40)
    ax.set_xticklabels([sample_spectra[i].spectra_name for i in range(len(sample_spectra))],rotation = 80)
    ax.set_ylabel('Thickness (nm)',fontsize=40);


    plt.legend(p,[element_text[oxides] for oxides in fit_legend],bbox_to_anchor=(1.0, 0.4, 0.0, 0.5),fontsize=20)
    plt.grid() 
    # return color_list, oxplotlist
    return fig, ax
