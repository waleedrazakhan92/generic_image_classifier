import os


learn_rate = 0.001
batch_size = 8
input_shape = (224,224,3)
path_dataset = 'dataset/train_data/'
path_validation = 'dataset/val_data/'
validation_steps = 2
model_num = 1
num_classes = len(os.listdir(path_dataset))

test_dataset = 'dataset/val_data/'
test_batch_size = 150

path_csv_logger = 'model_loss.csv'
path_plots = 'plots/'

val_split = 0.8

