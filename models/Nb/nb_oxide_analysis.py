import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy as dc
import lmfit as lm
from xps.gui_element_dicts import *

"""
Series of funcitons for analyzing thicknesses of the different oxides for NbOxides
"""

def transfer_func(depth,depth_bins,EAL,angle):
    trans = np.ones([4,depth_bins])
#     trans = np.ones(depth_bins)

#     iter = 0
#     for angle in np.arange(0,70,10):
    for thick in range(depth_bins):

        trans[:,thick] = np.exp( -(depth/depth_bins) /(EAL*np.cos(angle*(np.pi/180))))**thick
#         iter+=1

    return trans


# def oxide_concentration(x,Nb2O5_T,NbO_T,NbO2_T):
def oxide_concentration(x,pars):
    parsval = pars.valuesdict()
    NbO_T = parsval['NbO_T']
    NbO2_T = parsval['NbO2_T']
    Nb2O5_T = parsval['Nb2O5_T']
    
    ox_c = np.ones([4,len(x)])
    
    """Nb2O5"""
    ox_c[0,:] = 0.46*(np.ones(len(x))- np.heaviside(x - Nb2O5_T,1 ))
    """NbO2"""
    ox_c[1,:] = 0.51*(np.heaviside(x - Nb2O5_T,1) - np.heaviside(x-(Nb2O5_T + NbO2_T),1)) 
    """NbO"""
#     ox_c[2,:] = np.heaviside(x-(thickness - NbO_T),1)
    ox_c[2,:] = 0.73*(np.heaviside(x - (Nb2O5_T + NbO2_T),1) - np.heaviside(x-(Nb2O5_T + NbO2_T + NbO_T),1) )
    """Nb"""
    ox_c[3,:] = np.heaviside(x-(Nb2O5_T + NbO2_T + NbO_T),1)
    return ox_c


def thickness_residual(pars,x,data,error,transfer,EAL,obj_fun = 'Chi', split=False):
#     parsval = pars.valuesdict()
#     NbO_T = parsval['NbO_T']
#     NbO2_T = parsval['NbO2_T']
#     Nb2O5_T = parsval['Nb2O5_T']
    
#     calc_c = np.sum(transfer_func(thickness,len(x),EAL,0) * oxide_concentration(x,thickness,NbO_T,NbO2_T),axis = 1)/\
#     np.sum(transfer_func(thickness,len(x),EAL,0)*oxide_concentration(x,thickness,NbO_T,NbO2_T) )
    
#     calc_c = np.sum(transfer_func(np.max(x),len(x),EAL,0) * oxide_concentration(x,pars),axis = 1)/\
#     np.sum(transfer_func(np.max(x),len(x),EAL,0)*oxide_concentration(x,pars ))

    calc_c = np.sum(transfer * oxide_concentration(x,pars),axis = 1)/\
    np.sum(transfer*oxide_concentration(x,pars ))    
    
    
    err_adj = np.empty(len(error.keys()))
    err_adj[0] = error['Nb2O5_52_']
    err_adj[1] = error['NbO2_52_']
    err_adj[2] = error['NbO_52_']
    err_adj[3] = error['Nb_52_']
    err_adj = np.where(np.isnan(err_adj), 1, err_adj)
    if split==True:
        return calc_c
    if obj_fun == 'Chi' :
        return (calc_c - data)/err_adj
    if obj_fun =='S':
        return calc_c - data
#     return calc_c - data



def oxide_thickness(sample,oxides=None,substrate=None,S_oxide=None,S_substrate=None,\
    EAL=None,pars=None, fitting_alg = 'powell',obj_fun = 'S',specific_points = None,colors = None,fitflag = True,\
        plotflag = True):

    if pars is None:
        pars = lm.Parameters()
        pars.add('NbO_T',value = 1,min = 0,vary = 1)
        pars.add('NbO2_T',value = 1,min = 0, vary = 1)
        pars.add('Nb2O5_T',value = 4,min = 0, vary = 1)
    if oxides is None:
        oxides = ['NbO_52_','NbO2_52_','Nb2O5_52_']
    if substrate is None:
        substrate = 'Nb_52_'
    if S_oxide is None:
        S_oxide = 10583
    if S_substrate is None:
        S_substrate = 22917
    if EAL is None:
        EAL = 1.7
            
        
    if specific_points == None:
        if hasattr(sample,'fit_results_idx'):
            pts = [j for j,x in enumerate(sample.fit_results) if x] 
        else:
            pts = [j for j,x in enumerate(sample.params_full) if x] 

    else:
        pts = dc(specific_points)
    print(pts)


    """Fit the thicknesses of NbOxides"""
    if fitflag ==True:
        # areas = np.empty([len(sample.pairlist),len(pts)])

        sample.fit_component = {key: np.empty(len(pts)) for key in [sample.pairlist[i][0] for i in range(len(sample.pairlist))]} 

        sample.adjusted_oxide_thickness = {key: np.empty(len(pts)) for key in oxides}
        sample.adjusted_oxide_thickness_err = {key: np.empty(len(pts)) for key in oxides} 
        sample.thickness =np.empty(len(pts))
        oxide_thick_fit_result = [[] for i in range(len(pts))]

        iter=0

        for k in enumerate(pts):

            for pairs in sample.pairlist:

                if hasattr(sample,'fit_results_idx'):
                    sample.fit_component[pairs[0]][k[0]] = sum( [sample.fit_results[k[1]].params[pairs[i] + 'amplitude'].value for i in range(len(pairs))] )
                else:
                    sample.fit_component[pairs[0]][k[0]] = sum( [sample.params_full[k[1]][pairs[i] + 'amplitude'].value for i in range(len(pairs))] )                    

            sample.thickness[k[0]] = np.mean( EAL*np.log( 1 + S_substrate*sum([sample.fit_component[ox][k[0]] for ox in oxides])/(S_oxide*sample.fit_component[substrate][k[0]]) ) ) 
            
            ex_c = np.empty(4)
            ex_c[0] = sample.fit_component['Nb2O5_52_'][k[0]]
            ex_c[1] = sample.fit_component['NbO2_52_'][k[0]]
            ex_c[2] = sample.fit_component['NbO_52_'][k[0]]
            ex_c[3] = sample.fit_component['Nb_52_'][k[0]]
            ex_c = ex_c/np.sum(ex_c)

            x = np.linspace(0,10,10001)
            
            if hasattr(sample,'fit_results_idx'):
                err = {key[0]: sample.fit_results[k[0]].params[key[0] + 'amplitude'].stderr for key in sample.pairlist}
            else:
                err = {key[0]: sample.params_full[k[0]][key[0] + 'amplitude'].stderr for key in sample.pairlist} 
                    
                        
            tran = transfer_func(np.max(x),len(x),EAL,0)
            fitter = lm.Minimizer(thickness_residual, pars,fcn_args=(x,),fcn_kws={'data': ex_c,'error':err,\
                                                                                'transfer': tran, 'EAL':EAL,'obj_fun':obj_fun})
            result = fitter.minimize(method = fitting_alg)

            # sample.oxide_thick_fit_result[k[0]] = dc(result)
            sample.adjusted_oxide_thickness['Nb2O5_52_'][k[0]] = result.params['Nb2O5_T'].value
            sample.adjusted_oxide_thickness['NbO2_52_'][k[0]] = result.params['NbO2_T'].value
            sample.adjusted_oxide_thickness['NbO_52_'][k[0]] = result.params['NbO_T'].value
            
            sample.adjusted_oxide_thickness_err['Nb2O5_52_'][k[0]] = result.params['Nb2O5_T'].stderr
            sample.adjusted_oxide_thickness_err['NbO2_52_'][k[0]] = result.params['NbO2_T'].stderr
            sample.adjusted_oxide_thickness_err['NbO_52_'][k[0]] = result.params['NbO_T'].stderr
    #         print(result.params['Nb2O5_T'].value)
    #         print(result.params['Nb2O5_T'].stderr)
    
    """Plot the calculated thicknesses"""
    if plotflag:
        if colors is None:
            hue = element_color
        else:
            hue = colors
        width = 0.8
        fig, ax = plt.subplots(figsize=(15,10))

        p = [[] for i in range(len(oxides))]

        fit_legend = [element_text[element] for element in oxides]

        comps_so_far = []
        for ox in enumerate(oxides):

            bottom_iter = sum([sample.adjusted_oxide_thickness[i] for i in comps_so_far])

            p[ox[0]] = ax.bar(np.arange(0,len(pts)),sample.adjusted_oxide_thickness[ox[1]],width, bottom = bottom_iter, \
                          yerr = sample.adjusted_oxide_thickness_err[ox[1]], color = hue[ox[1]],capsize = 5)


            comps_so_far.append(ox[1])

        ax.set_xticks(np.arange(0,len(pts)))
        
        # ax.set_xticklabels(sample.data['pos names'],rotation = 90)
        ax.tick_params(labelsize = 40)
        ax.set_ylabel('Thickness (nm)',fontsize=40);


        plt.legend(p,fit_legend,bbox_to_anchor=(0.9, 0.6, 0.0, 0.5),fontsize=20)
        plt.grid()
        plt.tight_layout()
        return fig, ax

