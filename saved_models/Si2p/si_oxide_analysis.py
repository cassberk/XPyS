import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy as dc
import XPyS
import XPyS.config as cfg

def calc_oxide_thickness(sample,oxides=['SiOx1_32_','SiOx2_32_','SiOx3_32_','SiOx4_32_'],substrate='Si_32_',\
    S_oxide=1,S_substrate=1,EAL=2.84,specific_points = None,SFactors = None, plotflag = True):

    """Plots the thickness of each fit of the spectra object and then breaks the thickness into
    the corresponding components making up the fit. This is not adjusted for the depth of each oxide 
    component. However, SFactors can be used to adjust the contribution from the different oxides.

    Parameters
    ----------


    Returns
    -------
    matplotlib figures : fig
        Multi-plot plots of the data.
    matplotlib axes : ax
        axes object
    See Also
    --------
    
    """


    if specific_points == None:
        pts = [j for j,x in enumerate(sample.fit_results) if x] 
        # else:
        #     pts = [j for j,x in enumerate(sample.params_full) if x] 

    else:
        pts = dc(specific_points)

    if SFactors is None:
        # sfact = {ox:1 for ox in oxides}
        sfact = {'SiOx1_32_':1/0.9,'SiOx2_32_':1/0.71,'SiOx3_32_':1/0.4,'SiOx4_32_':1/0.44,'Si_32_':1}

    else:
        sfact = SFactors


    areas = np.empty([len(sample.pairlist),len(pts)])

    fit_component = {key: np.empty(len(pts)) for key in [sample.pairlist[i][0] for i in range(len(sample.pairlist))]} 

    sample.oxide_thickness = {key: np.empty(len(pts)) for key in oxides} 
    sample.thickness =np.empty(len(pts))

    iter=0

    for k in enumerate(pts):

        for pairs in sample.pairlist:
            # print(pairs)
            fit_component[pairs[0]][k[0]] = sum( [sfact[pairs[0]]*sample.fit_results[k[1]].params[pairs[0] + 'amplitude'].value for i in range(len(pairs))] )
               

        sample.thickness[k[0]] = np.mean( EAL*np.log( 1 + S_substrate*sum([fit_component[ox][k[0]] for ox in oxides])/(S_oxide*fit_component[substrate][k[0]]) ) ) 

        for pairs in oxides:

            sample.oxide_thickness[pairs][k[0]] = sample.thickness[k[0]] * ( fit_component[pairs][k[0]]/sum([fit_component[ox][k[0]] for ox in oxides]) )

    if plotflag == True:
        width = 0.8
        fig, ax = plt.subplots(figsize=(15,10))

        p = [[] for i in range(len(oxides))]

        fit_legend = [cfg.element_text[element] for element in oxides]

        comps_so_far = []
        for ox in enumerate(oxides):

            bottom_iter = sum([sample.oxide_thickness[i] for i in comps_so_far])

            p[ox[0]] = ax.bar(pts,sample.oxide_thickness[ox[1]],width, bottom = bottom_iter, \
                            color = cfg.element_color[ox[1]])


            comps_so_far.append(ox[1])

        ax.set_xticks(pts)
        if hasattr(sample,'positions'):
            ax.set_xticklabels(sample.positions,rotation = 90)
        ax.tick_params(labelsize = 40)
        ax.set_ylabel('Thickness (nm)',fontsize=40);


        plt.legend(p,fit_legend,bbox_to_anchor=(0.9, 0.6, 0.0, 0.5),fontsize=20)
        plt.grid() 

        return fig, ax