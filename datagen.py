from keras.utils.np_utils import to_categorical
from augmentor import *
import os
import cv2
import numpy as np
from glob import glob
import random	
random.seed(100) 

from config import num_classes

def train_val_split(path_dataset, path_validation,split=0.8):
    
    if path_validation==None:
        all_paths = set(get_paths(path_dataset))
        train_paths = random.sample(all_paths, int(len(all_paths)*split))
        train_paths = set(train_paths)
        val_paths = all_paths.difference(train_paths)
    else:
        train_paths = get_paths(path_dataset)
        val_paths = get_paths(path_validation)
       
    return list(train_paths), list(val_paths) 




def make_lab_dict(path_dataset):
    folders = os.listdir(path_dataset)
    folders = np.sort(folders)  
    labs = list(range(0,len(folders)))
    lab_dict = dict(zip(folders, labs))
    
    return lab_dict

def check_lab_dict(path_dataset, path_dict='lab_dict.npy'):
    if os.path.isfile(path_dict):
        print('Loading label dictionary. . .')
        lab_dict = np.load(path_dict,allow_pickle='TRUE').item()
    else:
        print('Creating label dictionary. . .')
        lab_dict = make_lab_dict(path_dataset)
        np.save(path_dict,lab_dict)
    return lab_dict

def get_paths(path_dataset):
    all_paths = glob(path_dataset+'/'+'*'+'/'+'*')
    return all_paths


def my_datagen(all_paths, lab_dict, batch_size=64, out_shape=(128,128,3), augm=False):
    
    if augm==True:
        augmentor = data_augmentor() 
    
    
    while True: 
        selected_paths = random.sample(all_paths, batch_size)
        
        batch_images = []
        batch_labels = []
        for i in range(0,len(selected_paths)):
            img = cv2.imread(selected_paths[i])
            img = cv2.resize(img,(out_shape[0],out_shape[1]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_category = selected_paths[i].split('/')[-2]
            lab = lab_dict[img_category]
            lab = to_categorical(lab, num_classes=num_classes)
            
            
            img = img/255.0
            if augm==True:
                itr = augmentor.flow(np.expand_dims(img,axis=0), batch_size=1)
                img = itr.next()
            
            batch_images.append(img.squeeze())
            batch_labels.append(lab)
            

        yield np.array(batch_images), np.array(batch_labels)
        
