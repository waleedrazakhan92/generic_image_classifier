# generic_image_classifier
An easy and clean repository to perform image classification tasks.

# Usage
1) Place the dataset in the folder and set the path_dataset variable to that folder. Do the same with path_validation.
2) Set validation steps to according to your dataset size.
3) Set the learning rate according to you GPU memory.
4) Set the batch size.
5) Select the model you want to train via model_num.
You can also turn on the augmentor in the datagen.py file.
run train.py
Ather the model is trained run inference.py file to get all the plots and results.
