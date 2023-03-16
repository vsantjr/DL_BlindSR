"""

DL_BlindSR: Supporting Code
----

** Author: Valdivino Alexandre de Santiago Júnior

** Licence: GNU GENERAL PUBLIC LICENSE, Version 3 (GPLv3)

** Description: This code allows the calculation of the NR-IQA metric NIQE. It was adapted based on the NIQE's implementation of Praful Gupta (https://github.com/guptapraful/niqe).

** Running the code:
   
    1.) Firstly, run:
    
    		"python calculate_NIQE.py --activity gen". 
    		
    This will generate the main json file with NIQE values;
    
    2.) Then run:
    		
    		"python calculate_NIQE.py --activity ana".
    		
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

gamma_range = np.arange(0.2, 10, 0.001)
a = scipy.special.gamma(2.0/gamma_range)
a *= a
b = scipy.special.gamma(1.0/gamma_range)
c = scipy.special.gamma(3.0/gamma_range)
prec_gammas = a/(b*c)

def aggd_features(imdata):
    #flatten imdata
    imdata.shape = (len(imdata.flat),)
    imdata2 = imdata*imdata
    left_data = imdata2[imdata<0]
    right_data = imdata2[imdata>=0]
    left_mean_sqrt = 0
    right_mean_sqrt = 0
    if len(left_data) > 0:
        left_mean_sqrt = np.sqrt(np.average(left_data))
    if len(right_data) > 0:
        right_mean_sqrt = np.sqrt(np.average(right_data))

    if right_mean_sqrt != 0:
      gamma_hat = left_mean_sqrt/right_mean_sqrt
    else:
      gamma_hat = np.inf
    #solve r-hat norm

    imdata2_mean = np.mean(imdata2)
    if imdata2_mean != 0:
      r_hat = (np.average(np.abs(imdata))**2) / (np.average(imdata2))
    else:
      r_hat = np.inf
    rhat_norm = r_hat * (((math.pow(gamma_hat, 3) + 1)*(gamma_hat + 1)) / math.pow(math.pow(gamma_hat, 2) + 1, 2))

    #solve alpha by guessing values that minimize ro
    pos = np.argmin((prec_gammas - rhat_norm)**2);
    alpha = gamma_range[pos]

    gam1 = scipy.special.gamma(1.0/alpha)
    gam2 = scipy.special.gamma(2.0/alpha)
    gam3 = scipy.special.gamma(3.0/alpha)

    aggdratio = np.sqrt(gam1) / np.sqrt(gam3)
    bl = aggdratio * left_mean_sqrt
    br = aggdratio * right_mean_sqrt

    #mean parameter
    N = (br - bl)*(gam2 / gam1)#*aggdratio
    return (alpha, N, bl, br, left_mean_sqrt, right_mean_sqrt)

def ggd_features(imdata):
    nr_gam = 1/prec_gammas
    sigma_sq = np.var(imdata)
    E = np.mean(np.abs(imdata))
    rho = sigma_sq/E**2
    pos = np.argmin(np.abs(nr_gam - rho));
    return gamma_range[pos], sigma_sq

def paired_product(new_im):
    shift1 = np.roll(new_im.copy(), 1, axis=1)
    shift2 = np.roll(new_im.copy(), 1, axis=0)
    shift3 = np.roll(np.roll(new_im.copy(), 1, axis=0), 1, axis=1)
    shift4 = np.roll(np.roll(new_im.copy(), 1, axis=0), -1, axis=1)

    H_img = shift1 * new_im
    V_img = shift2 * new_im
    D1_img = shift3 * new_im
    D2_img = shift4 * new_im

    return (H_img, V_img, D1_img, D2_img)


def gen_gauss_window(lw, sigma):
    sd = np.float32(sigma)
    lw = int(lw)
    weights = [0.0] * (2 * lw + 1)
    weights[lw] = 1.0
    sum = 1.0
    sd *= sd
    for ii in range(1, lw + 1):
        tmp = np.exp(-0.5 * np.float32(ii * ii) / sd)
        weights[lw + ii] = tmp
        weights[lw - ii] = tmp
        sum += 2.0 * tmp
    for ii in range(2 * lw + 1):
        weights[ii] /= sum
    return weights

def compute_image_mscn_transform(image, C=1, avg_window=None, extend_mode='constant'):
    if avg_window is None:
      avg_window = gen_gauss_window(3, 7.0/6.0)
    assert len(np.shape(image)) == 2
    h, w = np.shape(image)
    mu_image = np.zeros((h, w), dtype=np.float32)
    var_image = np.zeros((h, w), dtype=np.float32)
    image = np.array(image).astype('float32')
    scipy.ndimage.correlate1d(image, avg_window, 0, mu_image, mode=extend_mode)
    scipy.ndimage.correlate1d(mu_image, avg_window, 1, mu_image, mode=extend_mode)
    scipy.ndimage.correlate1d(image**2, avg_window, 0, var_image, mode=extend_mode)
    scipy.ndimage.correlate1d(var_image, avg_window, 1, var_image, mode=extend_mode)
    var_image = np.sqrt(np.abs(var_image - mu_image**2))
    return (image - mu_image)/(var_image + C), var_image, mu_image


def _niqe_extract_subband_feats(mscncoefs):
    # alpha_m,  = extract_ggd_features(mscncoefs)
    alpha_m, N, bl, br, lsq, rsq = aggd_features(mscncoefs.copy())
    pps1, pps2, pps3, pps4 = paired_product(mscncoefs)
    alpha1, N1, bl1, br1, lsq1, rsq1 = aggd_features(pps1)
    alpha2, N2, bl2, br2, lsq2, rsq2 = aggd_features(pps2)
    alpha3, N3, bl3, br3, lsq3, rsq3 = aggd_features(pps3)
    alpha4, N4, bl4, br4, lsq4, rsq4 = aggd_features(pps4)
    return np.array([alpha_m, (bl+br)/2.0,
            alpha1, N1, bl1, br1,  # (V)
            alpha2, N2, bl2, br2,  # (H)
            alpha3, N3, bl3, bl3,  # (D1)
            alpha4, N4, bl4, bl4,  # (D2)
    ])

def get_patches_train_features(img, patch_size, stride=8):
    return _get_patches_generic(img, patch_size, 1, stride)

def get_patches_test_features(img, patch_size, stride=8):
    return _get_patches_generic(img, patch_size, 0, stride)

def extract_on_patches(img, patch_size):
    h, w = img.shape
    patch_size = int(patch_size)
    patches = []
    for j in range(0, h-patch_size+1, patch_size):
        for i in range(0, w-patch_size+1, patch_size):
            patch = img[j:j+patch_size, i:i+patch_size]
            patches.append(patch)

    patches = np.array(patches)
    
    patch_features = []
    for p in patches:
        patch_features.append(_niqe_extract_subband_feats(p))
    patch_features = np.array(patch_features)

    return patch_features

def _get_patches_generic(img, patch_size, is_train, stride):
    h, w = np.shape(img)
    
    
    if h < patch_size or w < patch_size:
        print("Input image is too small")
        exit(0)

       
    # ensure that the patch divides evenly into img
    hoffset = (h % patch_size)
    woffset = (w % patch_size)
    

    if hoffset > 0: 
        img = img[:-hoffset, :]
    if woffset > 0:
        img = img[:, :-woffset]


    img = img.astype(np.float32)
    him = int(0.5*img.shape[0])
    wim = him 
    img2 = np.array(Image.fromarray(img).resize((wim, him)))
    

    mscn1, var, mu = compute_image_mscn_transform(img)
    mscn1 = mscn1.astype(np.float32)

    mscn2, _, _ = compute_image_mscn_transform(img2)
    mscn2 = mscn2.astype(np.float32)


    feats_lvl1 = extract_on_patches(mscn1, patch_size)
    feats_lvl2 = extract_on_patches(mscn2, patch_size/2)

    feats = np.hstack((feats_lvl1, feats_lvl2))# feats_lvl3))

    return feats

def niqe(inputImgData):

    patch_size = 96
    module_path = dirname(__file__)

    # TODO: memoize
    params = scipy.io.loadmat(join(module_path, 'data', 'niqe_image_params.mat'))
    pop_mu = np.ravel(params["pop_mu"])
    pop_cov = params["pop_cov"]


    M, N = inputImgData.shape

    # assert C == 1, "niqe called with videos containing %d channels. Please supply only the luminance channel" % (C,)
    assert M > (patch_size*2+1), "niqe called with small frame size, requires > 192x192 resolution video using current training parameters"
    assert N > (patch_size*2+1), "niqe called with small frame size, requires > 192x192 resolution video using current training parameters"


    feats = get_patches_test_features(inputImgData, patch_size)
    sample_mu = np.mean(feats, axis=0)
    sample_cov = np.cov(feats.T)

    X = sample_mu - pop_mu
    covmat = ((pop_cov+sample_cov)/2.0)
    pinvmat = scipy.linalg.pinv(covmat)
    niqe_score = np.sqrt(np.dot(np.dot(X, pinvmat), X))

    return niqe_score

# Generate the plots with NIQE values.
def generate_plots(df, opt, read = False):
    sns.set_style('darkgrid')
    #sns.set(font_scale=1.1)
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
           
            # Find min (best)
            dom_data_ind = dom_data['mean'].idxmin()
            itemi = {}
            itemi['dom'] = dom_data.loc[dom_data_ind].at['dom']
            itemi['dset'] = dom_data.loc[dom_data_ind].at['dset']
            itemi['tech'] = dom_data.loc[dom_data_ind].at['tech']
            itemi['min'] = dom_data.loc[dom_data_ind].at['mean']
            dom_min_max.append(itemi)
            print('Minimum mean value Domain - dom: {} - dset: {} - tech: {} - mean {}'.format(itemi['dom'],itemi['dset'],itemi['tech'],itemi['min']))
            # Find max (worse)
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
            plt.ylabel('Mean NIQE')
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
        
    
# This function receives a json file with all calculated NIQE values and outputs the results related to the BROADER DOMAIN and PER DATASET perspectives.
def analyse_niqe(f, data, tech, orientation = 'index'):
    df_rec = pd.read_json(f, orient = orientation)
    print('Data Frame Read - RAW Values')
    print(df_rec)
    
    
    all_min_ind = df_rec['niqe'].idxmin()
    print('Minimum RAW Value - dom: {} - dset: {} - tech: {} - file: {} - niqe: {} '.format(df_rec.loc[all_min_ind].at['dom'],df_rec.loc[all_min_ind].at['dset'],df_rec.loc[all_min_ind].at['tech'],df_rec.loc[all_min_ind].at['file'],df_rec.loc[all_min_ind].at['niqe']))
    
        
    all_max_ind = df_rec['niqe'].idxmax()
    print('Maximum RAW Value - dom: {} - dset: {} - tech: {} - file: {} - niqe: {} '.format(df_rec.loc[all_max_ind].at['dom'],df_rec.loc[all_max_ind].at['dset'],df_rec.loc[all_max_ind].at['tech'],df_rec.loc[all_max_ind].at['file'],df_rec.loc[all_max_ind].at['niqe']))
       
    print('#'*30)
    # Get mean NIQE.
    all_d = []
    for d in data:
        for t in tech:
            item = {}
            item['dom'] = d.rpartition('/')[0]
            item['dset'] = d.rpartition('/')[2]
            item['tech'] = t
            item['mean'] = df_rec.loc[(df_rec['dset'] == item['dset']) & (df_rec['tech'] == item['tech']), 'niqe'].mean()
            item['std'] = df_rec.loc[(df_rec['dset'] == item['dset']) & (df_rec['tech'] == item['tech']), 'niqe'].std()
            all_d.append(item)
        
    df_rec_met = pd.DataFrame(all_d)
    print('Data Frame - MEAN NIQE Values')
    print(df_rec_met)
    print('#'*30)
    
    # Generate plots and other json files.
    generate_plots(df_rec_met, 'dom', True)
    generate_plots(df_rec_met, 'dset', True)
    

    
if __name__ == "__main__":
    
    
    
    parser = argparse.ArgumentParser(description='NIQE')
    parser.add_argument('--activity', type=str, choices=['gen', 'ana'], default='gen')
    args = parser.parse_args()
    
    dataset = ['Aerial/condoaerial', 'Aerial/massachbuildings', 'Aerial/ships', 'Aerial/ufsm-flame', 
                   'Fauna/catsfaces', 'Fauna/dogsfaces',
                   'Flora/flowers', 'Flora/plantpat', 
                   'Medical/melanomaisic', 'Medical/structretina', 
                   'Satellite/amazonia1', 'Satellite/cbers4a', 'Satellite/deepglobe', 'Satellite/isaid']

    technique = ['APA', 'BlindSR', 'DAN', 'FastGAN', 'MoESR']
    
    if args.activity == 'gen':
        print('Generate json file !!!!')
        dir_img = '/Users/valdivino/Documents/Des/pythonw/DL/pyTorch/SuperGAN/Results/'

        all_data = []
        num_files = 0 # Total number of files.
    
        for d in dataset:
            for t in technique:
                for f in glob.iglob(f'{dir_img}{d}/{t}/img/*.*'):
                    num_files+=1
                    print('File #', num_files)
                    item = {}
                    item['dom'] = d.rpartition('/')[0]
                    item['dset'] = d.rpartition('/')[2]
                    item['tech'] = t
                    item['file'] = f.rpartition('/')[2]
                    img_file = []
                    img_file =  np.array(Image.open(f).convert('LA'))[:,:,0] 
                    item['niqe'] = niqe(img_file) 
                    all_data.append(item)
                        
        print('Total number of files: ', num_files)   
        print('%'*7)
    
        df_met = pd.DataFrame(all_data)
        print(df_met)
    
        # Save to json all NIQE values.
        df_met.to_json('df_niqe.json', orient = 'index')
        
    else: # Analyse NIQE values.
        print('Analyse NIQE data !!!!')
        analyse_niqe('df_niqe.json', dataset, technique, 'index')

    
    
    

   



