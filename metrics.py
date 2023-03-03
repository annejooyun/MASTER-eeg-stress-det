from sklearn import metrics
import numpy as np

def compute_metrics(y_true, y_pred):
    #print(metrics.classification_report(y_true, y_pred))

    conf_matrix = metrics.confusion_matrix(y_true, y_pred)
    print('\n Confusion matrix:')
    print(conf_matrix)

    TP = conf_matrix[0,0]
    TN = conf_matrix[1,1]
    FN = conf_matrix[1,0]
    FP = conf_matrix[0,1]

    # Accuracy = TP/ (TP + TN)
    accuracy = (TP+TN)/(TP+TN+FN+FP)

    # Sensitivity = TP / (TP + FN)
    if TP+FN < 1e-12:
        sensitivity = 0
        print("Null error in sensitivity")
    else:
        sensitivity = TP/(TP+FN)
    #print(f'\n Sensitivity: {sensitivity}')

    # Specificity = TN / (TN + FP)
    if TN+FP < 1e-12:
        specificity = 0
        print("Null error in specificity")
    else:  
        specificity = TN/(TN+FP)
    #print(f'\n Specificity: {specificity}')
    performance = np.array([accuracy, sensitivity, specificity])*100
    return np.round(performance, decimals=2)
    