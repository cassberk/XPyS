### Functions for importing the data into a dictionary
import os, sys
import numpy as np
import pandas as pd

def load_excel(filename):
    df = pd.read_excel(filename, None)
    sheetlist = list(df.keys())
    return builddatadict(df,sheetlist)


### Function for splitting up a path
def splitall(path):
    
    allparts = []
    
    while 1:
        parts = os.path.split(path)
        
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
            
        elif parts[1] == path: # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
            
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
            
    return allparts


### Function for getting the pathname from the excel file along with an index for where the searched for string is
def getpathlist(data,sheets,strsearch,excel_col):
    
    pathind = np.empty(len(sheets))
    pathlist = [0 for x in range(len(sheets))]
    
    for i in range(len(sheets)):

        if not data[sheets[i]].keys().tolist():
            print(sheets[i], 'is empty')
            break
        
        
        
        colheads = data[sheets[i]].keys()
        indfind = data[sheets[i]][data[sheets[i]][colheads[excel_col]].str.contains(strsearch)==True].index
        
        if len(indfind) == 0:
            
            pathind[i] = int(0)
            pathlist[i] = 'No Match'
            
        else:
            
            pathind[i] = int(indfind.values[0])            
            pathlist[i] = data[sheets[i]][colheads[excel_col]][pathind[i]]
            
    return pathlist, pathind


### Function to build the data dictionary
def builddatadict(data,sheets):
    import copy
    pathl, pathi = getpathlist(data,sheets,'C:',0)
    evl,evi = getpathlist(data,sheets,'Binding Energy',0)
    posl,posi = getpathlist(data,sheets,'Position ',1)


    ky1 = []
    ky2 = []
    relind = []

    for j in range(len(pathl)):
        
        if pathl[j] and evl[j] != 'No Match':

            relind.append(int(j))       ### Keep track of the indices of the path list that should have the xps data

            if pathl[j].split('\\')[-2] in ('Depth Profile','Iteration'):
            
                ky1.append(pathl[j].split('\\')[-3])    ### Create a list of keys (points in the XPS scans)
                
            else:
                
                ky1.append(pathl[j].split('\\')[-2])
                
    dd = list(dict.fromkeys(ky1))   ##get a list of all of the different points from XPS scans

    datad = {}
    for i in dd:
        datad[i] = {}    ### Create empty dictionary to store all the data 

        
    datadeV = {}
    datadI = {}
    
    for j in range(len(dd)):  

        for i in range(len(relind)):
            
            if pathl[relind[i]].split('\\')[-2] in ('Depth Profile', 'Iteration'):
                backind = int(-3)
            else:
                backind = int(-2)

            if dd[j] == pathl[relind[i]].split('\\')[backind]:   ### Find the paths for the same point

                spec = pathl[relind[i]].split('\\')[-1].split()[0].replace('.VGD','')   ### Create spectra key                   


                
                ### Store the spectral data in a dictionary format
                da = data[sheets[relind[i]]].drop(data[sheets[relind[i]]].index[0:int(evi[relind[i]])+2]).dropna(axis='columns').reset_index(drop=1).to_numpy(dtype = 'float32')
                
            
                
                datad[dd[j]][spec] = {}
                datad[dd[j]][spec] = {}        
                datad[dd[j]][spec]['energy'] = da[:,0]
                datad[dd[j]][spec]['intensity'] = np.transpose(da[:,1:])
                datad[dd[j]][spec]['info'] = pathl[relind[i]].split('\\')               
                datad[dd[j]][spec]['pos names'] = [data[sheets[relind[i]]].iloc[int(posi[relind[i]])][_+2] for _ in range(len(data[sheets[relind[i]]].keys())-2)]
                # print(int( posi[relind[i]]+2 ) )
                # print(data[sheets[relind[i]]].iloc[int(posi[relind[i]])])
                # print([data[sheets[relind[i]]].iloc[int(posi[relind[i]])][_+2] for _ in range(len(data[sheets[relind[i]]].keys())-2)])


#     return datad, pathl
    return datad


def del_empty_sheet(file):

    data_f = pd.read_excel(file, None)
    sheets = list(data_f.keys())

    for i in range(len(sheets)):

        if not data_f[sheets[i]].keys().tolist():

    #         print(sheets[i], 'is empty')
            wb=openpyxl.load_workbook(file)
            # create new sheet
            wb.remove(wb.get_sheet_by_name(sheets[i]))
            # save workbook
            wb.save(file)
         