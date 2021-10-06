
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import os

from config import num_classes

def make_classifier(model, num_classes):
    
    for l in model.layers:
        if 'bn' in l.name:
            #print(l.name)
            l.trainable = True
        else:
            l.trainable = False
    
    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    out = Dense(num_classes, activation='softmax')(x)
    
    new_model = Model(model.input, out)
    
    return new_model


def build_vgg(input_shape, num_classes):
      
    model_vgg = tf.keras.applications.VGG16(
    include_top=False,
    input_shape=input_shape,
    weights="imagenet")
    
    model = make_classifier(model_vgg, num_classes)
    
    return model


def build_resnet(input_shape, num_classes):

    model_resnet  = tf.keras.applications.ResNet50(
    include_top=False,
    input_shape=input_shape,
    weights="imagenet")
    
    model = make_classifier(model_resnet, num_classes)
       
    return model


def build_densenet(input_shape, num_classes):

    model_dense  = tf.keras.applications.DenseNet121(
    include_top=False,
    weights="imagenet",
    input_shape=input_shape)
    
    model = make_classifier(model_dense, num_classes)

    return model
    
def build_efficient_net(input_shape, num_classes):

    model_effi  = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=input_shape)
    
    model = make_classifier(model_effi, num_classes)
    
    return model


def init_model(model_name='my_model', path_ckpt='checkpoint/', input_shape=(128,128,3), model_select='vgg'):
    complete_path = path_ckpt+model_name+'.h5'
    
    if model_select=='vgg':
        print('vgg model selected')
        model = build_vgg(input_shape=input_shape, num_classes=num_classes)
        
    elif model_select=='resnet':
        print('resnet model selected')
        model = build_resnet(input_shape=input_shape, num_classes=num_classes)
        
    elif model_select=='densenet':
        print('densenet model selected')
        model = build_densenet(input_shape=input_shape, num_classes=num_classes)

    elif model_select=='efficient':
        print('efficientnet model selected')
        model = build_efficient_net(input_shape=input_shape, num_classes=num_classes)
    else:
        print('Please select valid model!')
        model = None
    
    
    if not os.path.isdir(path_ckpt):
        os.mkdir(path_ckpt)

    names = os.listdir(path_ckpt)
    if complete_path.split('/')[-1] in names:
        print('Loading checkpoint...')
        model.load_weights(complete_path)
    else:
        print('Training from scratch')

    return model, complete_path

    
