# generic_image_classifier
An easy and clean repository to perform image classification tasks.

# Models Covered
So far four models are available to train namely:
1) VGG
2) ResNet
3) DenseNet
4) EfficientNet

# Usage
1) Place the dataset in the folder and set the **path_dataset** variable to that folder. Do the same with **path_validation**.
2) Set **validation** steps to according to your dataset size.
3) Set the **learning rate**.
4) Set the **batch size** according to you GPU memory.
5) Select the model you want to train via **model_num**.
6) You can also turn on the augmentor in the *datagen.py* file.
7) run *train.py*
8) After the model is trained run *inference.py* file to get all the plots and results.
