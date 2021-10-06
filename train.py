import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger

import os

from datagen import *
from models import init_model

from config import *
from utils import calculate_class_weights

import random
random.seed(100) 


def train_model(path_dataset, path_validation, batch_size=64, epochs=2, img_shape=(128,128,3), model_select='vgg', augment=False):
    model, checkpoint_path = init_model(model_name=model_name[model_num] ,input_shape=img_shape, model_select=model_select)
    print(model.summary())
    
    checkpoint=ModelCheckpoint(filepath=checkpoint_path,monitor='loss',save_best_only=True,
                           save_weights_only=False,verbose=0,mode="auto")
    csv_logger = CSVLogger(model_name[model_num]+'_'+path_csv_logger, append=True, separator=',')    

    callbacks=[checkpoint, csv_logger]
    print('learning rate = ',learn_rate)
    
    opt = tf.keras.optimizers.Adam(lr = learn_rate)
    model.compile(optimizer=opt,loss="categorical_crossentropy")

    lab_dict = check_lab_dict(path_dataset=path_dataset)
    print('Labels dictionary', lab_dict)

    data_size = len(get_paths(path_dataset))
    print('Total images = ', data_size)

    class_weights = calculate_class_weights(path_dataset, lab_dict)
    print('Class weights = ',class_weights)
    
    train_paths, val_paths = train_val_split(path_dataset, path_validation, split=val_split)
    print('Total training images = ',len(train_paths))
    print('Total validation images = ',len(val_paths))    

    train_gen = my_datagen(train_paths, lab_dict, batch_size=batch_size, out_shape=img_shape, augm=augment)
    val_gen = my_datagen(val_paths, lab_dict, batch_size=batch_size, out_shape=img_shape, augm=augment)
    model.fit_generator(train_gen, epochs=epochs, validation_data=val_gen, validation_steps=validation_steps, callbacks = callbacks,steps_per_epoch=int(data_size/batch_size), class_weight=class_weights)

    # save trained model
    #model.save('checkpoint/' + 'trained_model.h5')


def main():
    global model_name
    model_name = {1:'vgg',2:'resnet',3:'densenet',4:'efficient'}
    train_model(path_dataset, path_validation, batch_size=batch_size, epochs=60, img_shape=input_shape,model_select=model_name[model_num], augment=True)

if __name__ == '__main__':
    main()
