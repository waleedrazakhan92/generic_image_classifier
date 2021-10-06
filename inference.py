import os

from datagen import *
from utils import *
from models import init_model

from config import *


def inference(path_dataset, lab_dict, batch_size=200, out_shape=(128,128,3), model_select='vgg'):
    
    model, checkpoint_path = init_model(model_name=model_name[model_num], input_shape=out_shape, model_select=model_select)
    print(model.summary())
    all_paths = get_paths(path_dataset)
    selected_paths = all_paths[0:batch_size]

    
    true_labs = []
    pred_labs = []
    true_one_hots = []
    pred_one_hots = []
    for i in range(0,len(selected_paths)):
        img = cv2.imread(selected_paths[i])
        img = cv2.resize(img,(out_shape[0],out_shape[1]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img/255.0
        img_category = selected_paths[i].split('/')[-2]
        lab = lab_dict[img_category]
        lab_one_hot = to_categorical(lab, num_classes=num_classes)

        pred = model.predict(np.expand_dims(img,axis=0)).squeeze()
        pred_idx = np.argmax(pred)
        print('---------------------------------')
        print('Actual = ',lab, '. predicted = ',pred_idx )
        print(pred)       
        pred_labs.append(pred_idx)
        true_labs.append(lab)

        true_one_hots.append(lab_one_hot)
        pred_one_hots.append(pred)
        print(i)
    
    return np.array(true_labs), np.array(pred_labs), true_one_hots, np.array(pred_one_hots)




global model_name
model_name = {1:'vgg',2:'resnet',3:'densenet',4:'efficient'}
lab_dict = check_lab_dict(path_dataset=path_dataset)
print (lab_dict)

true_labs, pred_labs, true_probs, pred_probs = inference(test_dataset,lab_dict,batch_size=test_batch_size,out_shape=input_shape,model_select=model_name[model_num])
cm = compute_conf_matrix(true_labs, pred_labs)
compute_prec_recall(true_labs, pred_labs)
calculate_accuracy(cm, true_probs, pred_probs)
plot_loss(model_name[model_num]+'_'+path_csv_logger)
plot_roc(true_labs, pred_labs, num_classes)

