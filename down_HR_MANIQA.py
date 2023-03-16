"""

DL_BlindSR: Supporting Code
----

** Author: Valdivino Alexandre de Santiago Júnior

** Licence: GNU GENERAL PUBLIC LICENSE, Version 3 (GPLv3)

** Description: This code performs the downsampling of the HR images produced by the DL techniques so that the MANIQA DNN can be properly used. The bicubic interpolation method was used (default of the resize - PIL's Image module).

** Running the code:
	
	"python down_HR_MANIQA.py"
	
"""

import glob
import os
from PIL import Image


# Auxiliary function to handle files.

def short_name(s):
    ind=s.rfind("/")
    new_s=s[ind+1:-4]
    return new_s

 
base_dir_in = './Results/' # Input dir: root.
base_dir = './Resultsx12Res/' # Output dir: root.

# Structure of the dataset dirs according to the broader domains.
data_in = ['Aerial/condoaerial/', 'Aerial/massachbuildings/', 'Aerial/ships/', 'Aerial/ufsm-flame/', 'Fauna/catsfaces/', 'Fauna/dogsfaces/', 'Flora/flowers/', 'Flora/plantpat/', 'Medical/melanomaisic/', 'Medical/structretina/', 'Satellite/amazonia1/','Satellite/cbers4a/', 'Satellite/deepglobe/', 'Satellite/isaid/']

# The DL techniques.          
algs_in = ['APA/', 'BlindSR/', 'DAN/', 'FastGAN/', 'MoESR/']

img_in = 'img/'
img_out = 'imgx12/'

# Downsampling to 224 x 224.
crop_d = 224  

print('Starting...')

for d in data_in:
          
    for a in algs_in:
  
        for f in glob.iglob(base_dir_in+d+a+img_in+'*.*'):
            im = []
            im = Image.open(f)
            width, height = im.size
            print('Image dimension - original: {}, {}'.format(width, height))
            
            im_res = []
            im_res = im.resize((crop_d, crop_d))
            width, height = im_res.size
            print('Image dimension - resized: {}, {}'.format(width, height))    
    
            print('Dir: {} - Image Short Name: {}'.format(d+a,short_name(f)))
               
            im_res.save(base_dir+d+a+img_out+short_name(f)+'_res.png', format="png")
                        
            print('*'*30)
            
            
            
    
