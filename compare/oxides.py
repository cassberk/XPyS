import numpy as np
import matplotlib.pyplot as plt
from xps_peakfit.gui_element_dicts import *
# from embed import shell()

def plot_oxides(sample_spectra,set_idx = None, error = 'std', width = 0.8,capsize = 20):

    # listrepeat = [spectra.spectra_name for spectra in sample_spectra]
    # print(listrepeat)
    # if not len(listrepeat) == len(set(listrepeat)):
    #     print('There are duplicate sample names!')
    #     return
    
    sample_dic_mean = {}
    sample_dic_err = {}


    for i in range(len(sample_spectra)):
       
        sample_dic_mean[sample_spectra[i].spectra_name] = {oxide:sample_spectra[i].oxide_thickness[oxide]\
                                                    [~np.isnan(sample_spectra[i].oxide_thickness[oxide])].mean()\
                                                    for oxide in list(sample_spectra[i].oxide_thickness.keys())}

        if error == 'std':
            sample_dic_err[sample_spectra[i].spectra_name] = {oxide:sample_spectra[i].oxide_thickness[oxide]\
                                                        [~np.isnan(sample_spectra[i].oxide_thickness[oxide])].std()\
                                                        for oxide in list(sample_spectra[i].oxide_thickness.keys())}
        elif error == 'max-min':
            sample_dic_err[sample_spectra[i].spectra_name] = {oxide:np.max(sample_spectra[i].oxide_thickness[oxide]\
                                                        [~np.isnan(sample_spectra[i].oxide_thickness[oxide])] ) - \
                                                            np.min(sample_spectra[i].oxide_thickness[oxide]\
                                                        [~np.isnan(sample_spectra[i].oxide_thickness[oxide])] ) 
                                                        for oxide in list(sample_spectra[i].oxide_thickness.keys())}        
        if error == 'var':
            sample_dic_err[sample_spectra[i].spectra_name] = {oxide:sample_spectra[i].oxide_thickness[oxide]\
                                                        [~np.isnan(sample_spectra[i].oxide_thickness[oxide])].var()\
                                                        for oxide in list(sample_spectra[i].oxide_thickness.keys())}          
        sample_spectra[i].spectra_name
        
        
    if set_idx is None:
        ind = np.arange(len(sample_spectra))
    else:
        ind = set_idx
    if type(width) == float:
        # print(ind)
        width = width *np.ones(len(ind))
#     width = 0.45
    
    fig, ax = plt.subplots(figsize=(15,10))

    p = []
    fit_legend = []

    plot_iter = 0
    for sample in enumerate(sample_spectra):

        comps_so_far = []

        for ox in sample[1].oxide_thickness.keys():

            bottom_iter = sum(comps_so_far)

            if ox not in fit_legend:
                fit_legend.append(ox)
                p.append(ax.bar(ind[sample[0]],sample_dic_mean[sample[1].spectra_name][ox], width[sample[0]], yerr=sample_dic_err[sample[1].spectra_name][ox],\
                              error_kw=dict(lw=5, capsize=20, capthick=3), bottom = bottom_iter,\
                          color = element_color[ox]))
                plot_iter+=1

            else:
                ax.bar(ind[sample[0]],sample_dic_mean[sample[1].spectra_name][ox], width[sample[0]], yerr=sample_dic_err[sample[1].spectra_name][ox],\
                              error_kw=dict(lw=5, capsize=20, capthick=3), bottom = bottom_iter,\
                          color = element_color[ox])


            comps_so_far.append(np.asarray([sample_dic_mean[sample[1].spectra_name][ox]]))


    ax.set_xticks(ind)
    ax.tick_params(labelsize = 40)
    ax.set_xticklabels([sample_spectra[i].spectra_name for i in range(len(sample_spectra))],rotation = 80)
    ax.set_ylabel('Thickness (nm)',fontsize=40)
    # ax.set_xscale('log')


    plt.legend(p,[element_text[oxides] for oxides in fit_legend],bbox_to_anchor=(1.0, 0.4, 0.0, 0.5),fontsize=20)
    plt.grid() 
    # return color_list, oxplotlist
    return fig, ax
