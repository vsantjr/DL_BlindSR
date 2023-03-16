"""

DL_BlindSR: Supporting Code
----

** Author: Valdivino Alexandre de Santiago Júnior

** Licence: GNU GENERAL PUBLIC LICENSE, Version 3 (GPLv3)

** Description: This code handles the values of the MANIQA scores of the HR images obtained by executing the DL techniques for blind image SR. 

** Running the code:
   
    1.) Firstly, run:
    
    		"python calculate_MANIQA.py --activity gen". 
    		
    No that all .csv files, generated by the DNN MANIQA, must be in the same directory of this code. These .csv files are necessary for the "gen" option to work properly. After running the code above, this will generate the main json file with MANIQA scores;
    
    2.) Then run:
    		
    		"python calculate_MANIQA.py --activity ana".
    		
    This will obtain the results related to the BROADER DOMAIN and PER DATASET perspectives, in addition to creating the plots. 
    
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

# Mapping the broader domain and dataset options to the respective dirs.
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

# Mapping the the technique option to the respective dir.
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

# Generate the plots with MANIQA scores.
def generate_plots(df, opt, read = False):
    sns.set_style('darkgrid')
    sns.set(rc={'figure.figsize':(18,16)})
    domain = ['Aerial', 'Fauna', 'Flora', 'Medical', 'Satellite']
    datas = ['condoaerial', 'massachbuildings', 'ships', 'ufsm-flame', 
                   'catsfaces', 'dogsfaces',
                   'flowers', 'plantpat', 
                   'melanomaisic', 'structretina', 
                   'amazonia1', 'cbers4a', 'deepglobe', 'isaid']
    techn = ['APA', 'BlindSR', 'DAN', 'FastGAN', 'MoESR']
    # Domain        
    if opt == 'dom':
        print('\n\n **** STARTING DOMAIN ****')
        dom_min_max = []
        for d in domain:
            dom_data = []
            dom_data = df.loc[df[opt] == d] 
            print(dom_data)
           
            # Find min (worst).
            dom_data_ind = dom_data['mean'].idxmin()
            itemi = {}
            itemi['dom'] = dom_data.loc[dom_data_ind].at['dom']
            itemi['dset'] = dom_data.loc[dom_data_ind].at['dset']
            itemi['tech'] = dom_data.loc[dom_data_ind].at['tech']
            itemi['min'] = dom_data.loc[dom_data_ind].at['mean']
            dom_min_max.append(itemi)
            print('Minimum mean value Domain - dom: {} - dset: {} - tech: {} - mean {}'.format(itemi['dom'],itemi['dset'],itemi['tech'],itemi['min']))
            # Find max (best).
            dom_data_ind_max = dom_data['mean'].idxmax()
            itema = {}
            itema['dom'] = dom_data.loc[dom_data_ind_max].at['dom']
            itema['dset'] = dom_data.loc[dom_data_ind_max].at['dset']
            itema['tech'] = dom_data.loc[dom_data_ind_max].at['tech']
            itema['max'] = dom_data.loc[dom_data_ind_max].at['mean']
            dom_min_max.append(itema)
            print('Maximum mean value Domain - dom: {} - dset: {} - tech: {} - mean {}'.format(itema['dom'],itema['dset'],itema['tech'],itema['max']))
            print('@'*15)
            sns.catplot(
                x='dset', 
                y='mean', 
                data=dom_data,
                palette='bright',
                kind='bar',
                hue='tech');
            plt.xlabel('Dataset')
            plt.ylabel('Mean MANIQA')
            plt.title(d)
            plt.savefig(d+".png", bbox_inches='tight')
            plt.show()
            
        dom_min_max_df = pd.DataFrame(dom_min_max)
        dom_min_max_df.to_json('dom_min_max.json', orient = 'index')
        if read:
            print('@'*50)
            df_dom_read = pd.read_json('dom_min_max.json', orient = 'index')
            print('Domain: Min and Max')
            print(df_dom_read)
            print('@'*50)
            print('Find Occurences Domain!')
            for c in techn:
                counts_mm = []
                counts_mm = df_dom_read.apply(lambda x : True
                            if (x['tech'] == c) else False, axis = 1)
                num_rows = 0
                num_rows = len(counts_mm[counts_mm == True].index)
                print('Domain Min/Max - {} : {}'.format(c, num_rows))
          
    # Dataset
    elif opt == 'dset':
        print('\n\n **** STARTING DATASET ****')
        dset_min_max = []
        for d in datas:
            dset_data = []
            dset_data = df.loc[df[opt] == d] 
            print(dset_data)
           
            # Find min (best)
            dset_data_ind = dset_data['mean'].idxmin()
            itemi = {}
            itemi['dom'] = dset_data.loc[dset_data_ind].at['dom']
            itemi['dset'] = dset_data.loc[dset_data_ind].at['dset']
            itemi['tech'] = dset_data.loc[dset_data_ind].at['tech']
            itemi['min'] = dset_data.loc[dset_data_ind].at['mean']
            dset_min_max.append(itemi)
            print('Minimum mean value Dataset - dom: {} - dset: {} - tech: {} - mean {}'.format(itemi['dom'],itemi['dset'],itemi['tech'],itemi['min']))
            # Find max (worse)
            dset_data_ind_max = dset_data['mean'].idxmax()
            itema = {}
            itema['dom'] = dset_data.loc[dset_data_ind_max].at['dom']
            itema['dset'] = dset_data.loc[dset_data_ind_max].at['dset']
            itema['tech'] = dset_data.loc[dset_data_ind_max].at['tech']
            itema['max'] = dset_data.loc[dset_data_ind_max].at['mean']
            dset_min_max.append(itema)
            print('Maximum mean value Dataset - dom: {} - dset: {} - tech: {} - mean {}'.format(itema['dom'],itema['dset'],itema['tech'],itema['max']))
            print('%'*15)
            
                        
        dset_min_max_df = pd.DataFrame(dset_min_max)
        dset_min_max_df.to_json('dset_min_max.json', orient = 'index')
        if read:
            print('%'*50)
            df_dset_read = pd.read_json('dset_min_max.json', orient = 'index')
            print('Dataset: Min and Max')
            print(df_dset_read)
            print('Find Occurences Dataset!')
            for c in techn:
                counts_mm = []
                counts_mm = df_dset_read.apply(lambda x : True
                            if (x['tech'] == c) else False, axis = 1)
                num_rows = 0
                num_rows = len(counts_mm[counts_mm == True].index)
                print('Dataset Min/Max - {} : {}'.format(c, num_rows))
        
    
# This function receives a json file with all calculated MANIQA scores and outputs the results related to the BROADER DOMAIN and PER DATASET perspectives.
def analyse_maniqa(f, data, tech, orientation = 'index'):
    df_rec = pd.read_json(f, orient = orientation)
    print('Data Frame Read - RAW Values')
    print(df_rec)
            
    all_min_ind = df_rec['maniqa'].idxmin()
    print('Minimum RAW Value - dom: {} - dset: {} - tech: {} - file: {} - maniqa: {} '.format(df_rec.loc[all_min_ind].at['dom'],df_rec.loc[all_min_ind].at['dset'],df_rec.loc[all_min_ind].at['tech'],df_rec.loc[all_min_ind].at['file'],df_rec.loc[all_min_ind].at['maniqa']))
            
    all_max_ind = df_rec['maniqa'].idxmax()
    print('Maximum RAW Value - dom: {} - dset: {} - tech: {} - file: {} - maniqa: {} '.format(df_rec.loc[all_max_ind].at['dom'],df_rec.loc[all_max_ind].at['dset'],df_rec.loc[all_max_ind].at['tech'],df_rec.loc[all_max_ind].at['file'],df_rec.loc[all_max_ind].at['maniqa']))
       
    print('#'*30)
    # Get mean MANIQA.
    all_d = []
    for d in data:
        for t in tech:
            item = {}
            item['dom'] = d.rpartition('/')[0]
            item['dset'] = d.rpartition('/')[2]
            item['tech'] = t
            item['mean'] = df_rec.loc[(df_rec['dset'] == item['dset']) & (df_rec['tech'] == item['tech']), 'maniqa'].mean()
            item['std'] = df_rec.loc[(df_rec['dset'] == item['dset']) & (df_rec['tech'] == item['tech']), 'maniqa'].std()
            all_d.append(item)
        
    df_rec_met = pd.DataFrame(all_d)
    print('Data Frame - MEAN MANIQA Values')
    print(df_rec_met)
    print('#'*30)
    
    # Generate plots and json.
    generate_plots(df_rec_met, 'dom', True)
    generate_plots(df_rec_met, 'dset', True)
    

    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='MANIQA')
    parser.add_argument('--activity', type=str, choices=['gen', 'ana'], default='gen')
    args = parser.parse_args()
    
    dataset = ['Aerial/condoaerial', 'Aerial/massachbuildings', 'Aerial/ships', 'Aerial/ufsm-flame', 
                   'Fauna/catsfaces', 'Fauna/dogsfaces',
                   'Flora/flowers', 'Flora/plantpat', 
                   'Medical/melanomaisic', 'Medical/structretina', 
                   'Satellite/amazonia1', 'Satellite/cbers4a', 'Satellite/deepglobe', 'Satellite/isaid']

    technique = ['APA', 'BlindSR', 'DAN', 'FastGAN', 'MoESR']
    
    # Domain and dataset values as defined in the .csv file names.
    dd_val = ['acon', 'amas', 'ashi', 'aufs', 'fcat', 'fdog', 'fflw', 'fplp',
             'mmel', 'mstr', 'sama', 'scbe', 'sdee', 'sisa']

    # Techniques values.
    t_val = ['apa', 'bli', 'dan', 'fas', 'moe']
    
    all_files = []

    for dv in dd_val:
        for tv in t_val:
            all_files.append('output_'+dv+'-'+tv+'.csv')

        
    if args.activity == 'gen':
        print('Generate json file !!!!')
        
        all_data = []
        num_files = 0 # total number of files
    
        for af in all_files:
            df = pd.read_csv(af, header=None)
            #print(df)
            print(len(df))
            df.reset_index()  # make sure indexes pair with number of rows

            for index, row in df.iterrows():
                num_files+=1
                fname, split = '', ''
                item = {}
                f_name = os.path.basename(af)
                split = f_name.rpartition('-')
                # Get the domain and dataset.
                domdset = split[0]
                item['dom'], item['dset'] = get_DomainDset(domdset.rpartition('_')[2])
                
                # Get the technique.
                tech = split[2]
                item['tech'] = get_Technique(tech.rpartition('.')[0])
                item['file'] = row[0]
                item['maniqa'] = row[1]
        
                all_data.append(item)
            
        print(all_data)
        print(len(all_data))

        print('Total number of files: ', num_files)   
        print('%'*7)
    
        df_met = pd.DataFrame(all_data)
        print(df_met)
    
        # Save to json.
        df_met.to_json('df_maniqa.json', orient = 'index')
      
       
    else: # Analyse MANIQA data.
        print('Analyse MANIQA data !!!!')
        analyse_maniqa('df_maniqa.json', dataset, technique, 'index')

    
    
    

   


