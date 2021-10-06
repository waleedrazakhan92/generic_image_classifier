from tensorflow.keras.preprocessing.image import ImageDataGenerator

def data_augmentor():
    
    aug = ImageDataGenerator(
    rotation_range=5,
    zoom_range=0.1,
    shear_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest")

#     width_shift_range=0.1,
#     height_shift_range=0.1,
    
    return aug


