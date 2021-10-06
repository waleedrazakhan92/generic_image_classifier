from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import average_precision_score

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from keras.utils import to_categorical
import os
from config import path_plots, model_num


global model_name
model_name = {1:'vgg',2:'resnet',3:'densenet',4:'efficient'}

def compute_conf_matrix(true_labs, pred_labs):
    conf = confusion_matrix(true_labs, pred_labs)
    print('Confusion Matrix:')
    print(conf)
    return conf
    
def calculate_accuracy(cm, true_probs, pred_probs):
    #cm = pd.DataFrame(cm)
    FP = cm.sum(axis=0) - np.diag(cm)  
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)
    # Positive Predictive Value
    PPV = (TP)/(TP+FP)
    # Negative Predictive Value
    NPV = (TN)/(TN+FN)
    # Mean Average Precision
    MAP = average_precision_score(true_probs, pred_probs, average=None)
    
    print('Senstivity = ', TPR)
    print('Specificity = ',TNR)
    print('Accuracy = ',ACC)
    print('Positive Predictive Value = ',PPV)
    print('Negative Predictive Value = ',NPV)
    print('Mean Average Precision = ', MAP)

def compute_prec_recall(true_labs, pred_labs):
    precision, recall, fscore, support = score(true_labs, pred_labs)
    weighted = np.sum(precision)/len(precision)
    print('Precision = ',precision)
    print('Recall = ',recall)
    print('Weighted Average = ',weighted)
    print('F1 Score = ',fscore)


def calculate_class_weights(path_dataset, lab_dict):
    classes = os.listdir(path_dataset)
    weights = []
    labels = []
    for i in range(0,len(classes)):
        weights.append(len(os.listdir(path_dataset+classes[i])))
        labels.append(lab_dict[classes[i]])
        
    class_weights = np.sum(weights)/weights
    normalized_weights = class_weights/np.min(class_weights)
    
    
    return dict(zip(labels, normalized_weights))


def plot_loss(path_csv):
    new_path_plots = model_name[model_num]+'_'+path_plots
    if not os.path.isdir(new_path_plots):
        os.mkdir(new_path_plots)
    loss_path = new_path_plots+'loss/'
    if not os.path.isdir(loss_path):
        os.mkdir(loss_path)

  
    loss_csv = pd.read_csv(path_csv)
    plt.title('Model Loss')
    plt.plot(loss_csv['loss'])
    plt.plot(loss_csv['val_loss'])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.gca().legend(('train_loss','val_loss'))
    plt.savefig(loss_path+'loss_curve.png')
    

def plot_roc(true_labs, pred_labs, n_classes, multi_plots=False):
    new_path_plots = model_name[model_num]+'_'+path_plots
    if not os.path.isdir(new_path_plots):
        os.mkdir(new_path_plots)
    roc_path = new_path_plots+'roc/'
    if not os.path.isdir(roc_path):
        os.mkdir(roc_path)

    y_test = to_categorical(true_labs, n_classes)
    y_score  = to_categorical(pred_labs, n_classes)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    
    classes = [*roc_auc]
    # Plot of a ROC curve for a specific class
    if multi_plots==False:
        plt.figure()
    for i in range(n_classes):
        if multi_plots==True:
            plt.figure()
        plt.plot(fpr[i], tpr[i], label='class '+str(classes[i])+' (area = %0.2f)' % roc_auc[i])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")        
        #plt.show()
        if multi_plots==True:
            plt.savefig(roc_path+'roc_curve_'+str(classes[i])+'.png')

    if multi_plots==False:
        plt.savefig(roc_path+'roc_curve.png')

