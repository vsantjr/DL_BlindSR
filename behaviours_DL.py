"""

DL_BlindSR: Supporting Code
----

** Author: Valdivino Alexandre de Santiago Júnior

** Licence: GNU GENERAL PUBLIC LICENSE, Version 3 (GPLv3)

** Description: This code helps perceiving whether the two top approaches present similar behaviours when deriving HR images. There is also a correlation analysis related to this question. 

** Running the code:
   
    "python behaviours_DL.py". 
    		
   Input of this code is the file "df_maniqa.json" generated by "calculate_MANIQA.py". This will obtain the results in addition to creating the correlation analysis plots. 
    
"""

import numpy as np
import scipy.misc
import scipy.io
from os.path import dirname
from os.path import join
import scipy
from PIL import Image
import numpy as np
import scipy.ndimage
import numpy as np
import scipy.special
import math
import pandas as pd
import glob
import json
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import os
import scipy.stats 

def get_DomainDset(dd):
    domain, dset = '', ''
    if dd == 'acon':
        domain = 'Aerial'
        dset = 'condoaerial'
    elif dd == 'amas':
        domain = 'Aerial'
        dset = 'massachbuildings'
    elif dd == 'ashi':
        domain = 'Aerial'
        dset = 'ships'
    elif dd == 'aufs':
        domain = 'Aerial'
        dset = 'ufsm-flame'
    elif dd == 'fcat':
        domain = 'Fauna'
        dset = 'catsfaces'
    elif dd == 'fdog':
        domain = 'Fauna'
        dset = 'dogsfaces'
    elif dd == 'fflw': 
        domain = 'Flora'
        dset = 'flowers'   
    elif dd == 'fplp': 
        domain = 'Flora'
        dset = 'plantpat'
    elif dd == 'mmel': 
        domain = 'Medical'
        dset = 'melanomaisic'  
    elif dd == 'mstr': 
        domain = 'Medical'
        dset = 'structretina'
    elif dd == 'sama': 
        domain = 'Satellite'
        dset = 'amazonia1'
    elif dd == 'scbe': 
        domain = 'Satellite'
        dset = 'cbers4a'
    elif dd == 'sdee': 
        domain = 'Satellite'
        dset = 'deepglobe'
    elif dd == 'sisa': 
        domain = 'Satellite'
        dset = 'isaid'
    else:
        print('Invalid option!')
        
    return domain, dset

def get_Technique(tc):
    technique = ''
    if tc == 'apa':
        technique = 'APA'
    elif tc == 'bli':
        technique = 'BlindSR'
    elif tc == 'dan':
        technique = 'DAN'
    elif tc == 'fas':
        technique = 'FastGAN'
    elif tc == 'moe':
        technique = 'MoESR'
    else:
        print('Invalid option!')
        
    return technique    

# Save files with the list of the 10 images with highest and 10 images with lowest MANIQA scores.
def save_files(dat, tec, n, df_s, df_g):
    fs = open('xai_' + dat + '-' + tec + '_' + str(n) + '.txt',"w+")
    fs.write('Small {}!'.format(n))
    fs.write('\n'+str(df_s))
    fs.write('\n\nGreat {}!'.format(n))
    fs.write('\n'+str(df_g))
    fs.close()

# Compare dataframes.    
def compare_dfs(op, domc, dsetc, df1, df2):
    print('\nCompare dfs!')
    print('Option:', op)
    
    c = df1['file'].isin(df2['file']).value_counts()
    comp = pd.DataFrame([c.rename(None)])
    comp.insert(0, "pos", [op], False)
    comp.insert(1, "dom", [domc], False)
    comp.insert(2, "dset", [dsetc], False)
    print(comp)
    return comp
        
# This function helps obtaining the cardinalities of the common (C) and non-common (N) sets.
def get_Low_High_Card(f, sdo, sda, ste1, ste2, nv, orientation = 'index'):
    df_rec = pd.read_json(f, orient = orientation)
    print('#'*50)
    
    ## Get specific sda, ste1.
    print('Specific -> dset: {} - tech1: {}'.format(sda, ste1))
    df_spec1 = df_rec.loc[(df_rec['dset'] == sda) & (df_rec['tech'] == ste1)]
    #print(df_spec1)
        
    print('\nSmall {} !'.format(nv))
    small_v1 = df_spec1.nsmallest(nv, 'maniqa')
    print(small_v1)
        
    print('\nGreat {} !'.format(nv))
    great_v1 = df_spec1.nlargest(nv, 'maniqa')
    print(great_v1)
    
    save_files(sda, ste1, nv, small_v1, great_v1)
    
    ## Get specific sda, ste2.
    print('\n\nSpecific -> dset: {} - tech2: {}'.format(sda, ste2))
    df_spec2 = df_rec.loc[(df_rec['dset'] == sda) & (df_rec['tech'] == ste2)]
    #print(df_spec2)
        
    print('\nSmall {} !'.format(nv))
    small_v2 = df_spec2.nsmallest(nv, 'maniqa')
    print(small_v2)
        
    print('\nGreat {} !'.format(nv))
    great_v2 = df_spec2.nlargest(nv, 'maniqa')
    print(great_v2)
    
    save_files(sda, ste2, nv, small_v2, great_v2)
    
    ## Compare dataframes.
    comp_small = compare_dfs('Small', sdo, sda, small_v1, small_v2)
    comp_great = compare_dfs('Great', sdo, sda, great_v1, great_v2)
    return comp_small, comp_great

# This function performs the correlation analysis.        
def corr_analysis(df1, df2, option, fname, title):
    print('&&&')
    f_df1 = pd.to_numeric(df1[option], downcast='float') 
    f_df2 = pd.to_numeric(df2[option], downcast='float') 
    print(f_df1)
    print(f_df2)
    
    
    # Check normality of data.
    shap_df1 = scipy.stats.shapiro(f_df1)
    shap_df2 = scipy.stats.shapiro(f_df2)

    print('Shapiro-Wilk DF1: statistic {} and p-value {} '.format(shap_df1[0], shap_df1[1]))
    print('Shapiro-Wilk DF2: statistic {} and p-value {} '.format(shap_df2[0], shap_df2[1]))
    
    kendall = scipy.stats.kendalltau(f_df1, f_df2)
    print("Kendall's tau {} and p-value {}".format(kendall[0],kendall[1]))
    
    pearson = scipy.stats.pearsonr(f_df1, f_df2)
    print("Pearson's r/rho {} and p-value {}".format(pearson[0],pearson[1]))

    plt.figure()
    a, b = np.polyfit(f_df1, f_df2, 1)
    plt.scatter(f_df1, f_df2);
    plt.title("Kendall Rank Correlation Coefficient - "+title)
    plt.plot(f_df1, a*f_df1+b);
    plt.savefig(fname+".png", bbox_inches='tight')
    plt.show()
       
    
if __name__ == "__main__":
            
    dataset = ['acon', 'amas', 'ashi', 'aufs', 'fcat', 'fdog', 'fflw', 'fplp',
               'mmel', 'mstr', 'sama', 'scbe', 'sdee', 'sisa']

    technique = ['moe', 'dan']
    
    print('Behaviours DL Techniques !!!!')
    
    n_values = 10
    i = 0
    all_csmall = pd.DataFrame()
    all_cgreat = pd.DataFrame()
    for d in dataset:
        spec_dom , spec_data = get_DomainDset(d) 
        spec_tech1 = get_Technique(technique[0])
        spec_tech2 = get_Technique(technique[1])
        csmall, cgreat = get_Low_High_Card('df_maniqa.json', spec_dom, spec_data, spec_tech1, spec_tech2, n_values, 'index')
        all_csmall = all_csmall.append(csmall)
        all_cgreat = all_cgreat.append(cgreat)
        print('@'*100)
        
    
    print('\n\n')
    print('%'*700)
    print('\nAll Small:\n')
    all_csmall.reset_index(drop=True, inplace=True)
    all_csmall.fillna(0, inplace=True)
    print(all_csmall)
    print('\nSmall TRUE - Mean: {} and Std: {}'.format(all_csmall[True].mean(), all_csmall[True].std()))
    print('\nSmall FALSE - Mean: {} and Std: {}'.format(all_csmall[False].mean(), all_csmall[False].std()))
    all_csmall.to_json('df_maniqa_XAI_SMALL.json', orient = 'index')
    print('%'*100)
    
    print('\nAll Great:\n')
    all_cgreat.reset_index(drop=True, inplace=True)
    all_cgreat.fillna(0, inplace=True)
    print(all_cgreat)
    print('\nGreat TRUE - Mean: {} and Std: {}'.format(all_cgreat[True].mean(), all_cgreat[True].std()))
    print('\nGreat FALSE - Mean: {} and Std: {}'.format(all_cgreat[False].mean(), all_cgreat[False].std()))
    all_cgreat.to_json('df_maniqa_XAI_GREAT.json', orient = 'index')
    print('%'*100)  
    
    # Correlation analysis.
    print('\n\nCorrelation Analysis - TRUE!')
    corr_analysis(all_csmall, all_cgreat, True, 'corr_TRUE', 'True')
    print('\n\nCorrelation Analysis - FALSE!')
    corr_analysis(all_csmall, all_cgreat, False, 'corr_FALSE', 'False')
  
    
    
