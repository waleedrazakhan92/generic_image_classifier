import shutil
import os
import pandas as pd


import shutil
import os
import pandas as pd
from glob import glob

def get_paths(path_dataset):
    all_paths = glob(path_dataset+'/'+'*'+'/'+'*')
    return all_paths

def check_skin_dataset(path_write, path_csv):
    
    
    all_paths = get_paths(path_write)
    data_df = pd.read_csv(path_csv)
    
    for i in range(0,len(all_paths)):
#         print('--------------------------------------------------')
#         print(i)
        
        folder_name = all_paths[i].split('/')[-2]
        img_name = all_paths[i].split('/')[-1].split('.')[0]

        
        if not float(data_df[data_df['image']==img_name][folder_name])==1.0:
            print(all_paths[i])
            print(data_df[data_df['image']==img_name][folder_name])
        
        else:
            pass
        
        


###path_write = '/media/waleed/904846EF4846D41E/ubuntu_data/skin_dataset_subset/'
###path_csv = '/media/waleed/904846EF4846D41E/ubuntu_data/ISIC_2019_Training_GroundTruth.csv'

###check_skin_dataset(path_write, path_csv)
