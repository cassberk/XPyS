import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from lmfit.lineshapes import gaussian
from IPython import embed as shell

def fit_spectogram(input_samples,sputter_time = None,specify_scans = None, sigma_weight = 1,c_weight = 0.5,multiple_samples = False,\
    normalize = 'all',axis = 0,pltlims = 0,fig = False, ax = False):

    if multiple_samples:
        numsamples = len(input_samples.keys())
    else:
        numsamples = 1
        input_samples = {input_samples.sample_name:input_samples}
        
    if specify_scans is not None:
        spectra_list = specify_scans
    else:
        spectra_list = input_samples[list(input_samples.keys())[0]].element_scans

    if sputter_time == None:
        sputter_time = {sample_name: None for sample_name in input_samples.keys()}

    if axis == 0:
        # figure, axes = plt.subplots(numsamples,len(spectra_list),figsize = (25,1*numsamples))
        figure, axes = plt.subplots(numsamples,len(spectra_list),figsize = (25,4*numsamples))

    elif axis ==1:
        figure, axes = plt.subplots(len(spectra_list),1,figsize = (3,25))

    axes = axes.ravel()

    df = pd.DataFrame()
    for sample_name,sample in input_samples.items():
        df_temp = pd.DataFrame()
        for scan in spectra_list:
            # print(scan)
            spec = sample.__dict__[scan]
            for pairs in spec.pairlist:
                df_temp[sample_name+'_'+spec.orbital+'_'+pairs[0]+'amp'] = pd.Series([sum([spec.fit_results[i].params[p+'amplitude'].value for p in pairs]) for i in range(len(spec.fit_results))])
                df_temp[sample_name+'_'+spec.orbital+'_'+pairs[0]+'cent'] = pd.Series([spec.fit_results[i].params[pairs[0]+'center'].value  for i in range(len(spec.fit_results))])
                df_temp[sample_name+'_'+spec.orbital+'_'+pairs[0]+'sig'] = pd.Series([spec.fit_results[i].params[pairs[0]+'sigma'].value  for i in range(len(spec.fit_results))])
        df = pd.concat([df,df_temp], axis=1)

    j = 0
    samplenum = 0
    for sample_name,sample in input_samples.items():
        orderlist = [(orbital,np.max(sample.__dict__[orbital].esub)) for orbital in spectra_list]
        orderlist.sort(key=lambda x:x[1])
        orderlist = [spec[0] for spec in orderlist]

        scannum = 0
        for scan in orderlist[::-1]:
            spec = sample.__dict__[scan]
            if sputter_time[sample_name] == None:
                e, d = np.meshgrid(np.linspace(np.min(spec.esub), np.max(spec.esub), len(spec.esub)), np.linspace(0, len(spec.isub), len(spec.isub)))
            else:
                if pltlims > sputter_time[sample_name]*len(spec.isub):
                    yrange = int(np.ceil(pltlims/sputter_time[sample_name]))
                    e, d = np.meshgrid(np.linspace(np.min(spec.esub), np.max(spec.esub), len(spec.esub)), np.linspace(0, yrange*sputter_time[sample_name], yrange))
                else:
                    e, d = np.meshgrid(np.linspace(np.min(spec.esub), np.max(spec.esub), len(spec.esub)), np.linspace(0, sputter_time[sample_name]*len(spec.isub), len(spec.isub)))
                    

            c = np.zeros(e.shape)
            amplist = [k for k in [i for i in df.keys() if scan in i] if ('amp' in k)]
            maxamp = np.max([np.array(df[amplist[i]].dropna()).max() for i in range(len(amplist))])
            # print(scan,maxamp)
            for pairs in spec.pairlist:
                specpars = [k for k in [i for i in df.keys() if scan in i] if (pairs[0] in k) and (sample_name in k)]
                for i in range(len(spec.isub)):
                    specmap =gaussian(e[i,:],amplitude =  df[specpars[0]][i],center = df[specpars[1]][i],sigma = df[specpars[2]][i]*sigma_weight)
                    c[i,:] += specmap
                    if normalize =='all':
                        c[i,:] += specmap/maxamp

            # pc = axes[j].pcolormesh(e, d, c,vmin=0.0,vmax=c.max()*c_weight)
            if normalize == 'all':
                pc = axes[j].pcolormesh(e, d, c,vmin=0.0,vmax=maxamp*c_weight)
            else:
                pc = axes[j].pcolormesh(e, d, c,vmin=0.0,vmax=np.max(c))

            if pltlims != 0:
                axes[j].set_ylim([pltlims,0])
            else:
                axes[j].set_ylim(axes[j].get_ylim()[::-1])
            axes[j].tick_params('y',labelsize = 20)
            # axes[j].get_yaxis().set_ticks(np.arange(0,sputter_time*len(spec.isub)+1,20))

            axes[j].set_xlim(axes[j].get_xlim()[::-1])
            axes[j].set_label(scan)

            if multiple_samples:
                axes[j].xaxis.set_visible(False)
                if samplenum == 0:
                    axes[j].set_title(scan,fontsize = 40)
                if samplenum ==len(input_samples.keys())-1:
                    axes[j].xaxis.set_visible(True)  
                    axes[j].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
                    # axes[j].set_xticks(np.arange(np.min(axes[j].get_xlim()),np.max(axes[j].get_xlim())))
                    # for axis in [ax.xaxis, ax.yaxis]:
                    #     axis.set_major_locator(ticker.MaxNLocator(integer=True))

                    axes[j].tick_params('x',labelsize = 20,labelrotation=80)
                if j%len(orderlist) == 0:
                    axes[j].set_ylabel(sample_name,fontsize = 20)
                else:
                    axes[j].set_yticks([])
                    # axes[j].set_xlabel('Binding Energy (eV)',fontsize = 18,labelpad=20)



            elif not multiple_samples:
                # axes[j].set_title(scan)
                axes[j].set_title(scan,fontsize = 40)
                for ax in axes:
                    ax.tick_params('x',labelrotation=80)
                    # ax.set_xlabel('Binding Energy (eV)',fontsize = 18,labelpad=20)
                    if (ax ==axes[0]):
                        if sputter_time == None:
                            ax.set_ylabel('Scan#')
                        # else:
                            # ax.set_ylabel('Sputter Time (sec)')
                    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                                ax.get_xticklabels() + ax.get_yticklabels()):
                        item.set_fontsize(24)

            j+=1
        samplenum+=1

    figure.tight_layout()
    figure.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    plt.xlabel('Binding Energy (eV)',fontsize = 40,labelpad=85)
    plt.ylabel('Sputter Time (sec)',fontsize = 30,labelpad=50)

    return figure, axes,df
