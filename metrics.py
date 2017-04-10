import numpy as np
from keras import backend as K
import tensorflow as tf

# Thanks to issacgerg on https://github.com/fchollet/keras/issues/3230

#-----------------------------------------------------------------------------------------------------------------------------------------------------
# AUC for a binary classifier
def auc(y_true, y_pred):   
    ptas = tf.stack([binary_PTA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.stack([binary_PFA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.concat([tf.ones((1,)) ,pfas],axis=0)
    binSizes = -(pfas[1:]-pfas[:-1])
    s = ptas*binSizes
    return K.sum(s, axis=0)

#-----------------------------------------------------------------------------------------------------------------------------------------------------
# PFA, prob false alert for binary classifier
def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # N = total number of negative labels
    N = K.sum(1 - y_true)
    # FP = total number of false alerts, alerts from the negative class labels
    FP = K.sum(y_pred - y_pred * y_true)    
    return FP/N
#-----------------------------------------------------------------------------------------------------------------------------------------------------
# P_TA prob true alerts for binary classifier
def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # P = total number of positive labels
    P = K.sum(y_true)
    # TP = total number of correct alerts, alerts from the positive class labels
    TP = K.sum(y_pred * y_true)    
    return TP/P

# def sklearnAUC(test_labels, test_prediction, n_classes=2 ):
#     '''
#     test_labels : one hot vector of actual labels looks something like [ [0,1] , [1,0] .... ]
#     test_prediction : probabilities of different classes, like [ [0.3,0.7] , [0.2,0.8] ....]    
#     returns : list of aucs for each class
#     sample : 
#         #auc1 and auc2 should be equal
#         auc1 , auc2 = sklearnAUC(  Y_test ,  Y_pred, 2 )
#     '''
#     # Compute ROC curve and ROC area for each class
#     fpr = dict()
#     tpr = dict()
#     roc_auc = list()
#     for i in range(n_classes):
#         # ( actual labels, predicted probabilities )
#         fpr[i], tpr[i], _ = roc_curve(test_labels[:, i], test_prediction[:, i])
#         roc_auc.append( auc(fpr[i], tpr[i]) )
#     return roc_auc
# 
# def roc_auc(y, y_pred):
#     '''
#     This is a wrapper for sklearnAUC to fit our data
#     y : true labels, catgorical, like [ 0, 1, ... ]
#     y_pred : predictred classes, like [ [0.3,0.7], [0.2,0.8], ...]
#     '''
#     y_labels = np.eye(2)[y]
#     auc1, auc2 = sklearnAUC(y_labels, y_pred)
#     logging.debug('aucs: {}, {}'.format(auc1, auc2))
#     return auc1 


