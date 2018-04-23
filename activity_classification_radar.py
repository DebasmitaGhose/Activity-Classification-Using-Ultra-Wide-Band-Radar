# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 00:14:59 2018

@author: User
"""

import numpy as np
import pandas as pd
import glob
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,classification_report,confusion_matrix
import itertools
import matplotlib.pyplot as plt

filepath = r"C:/Users/User/Documents/UMass Amherst/Semester 2/COMPSCI 590U - Mobile and Ubiquitous Computing/Assignments/Assignment 4/Code (1)/Code/feature_vector/"
datafiles = os.listdir(filepath)
allFiles = glob.glob(filepath + "/*.csv")
data = []

for d in allFiles:
    print(d)
    #if "_1.csv" in d:
     #   print("hi")
      #  continue
    #if "_3.csv" in d:
     #   continue
    line = np.genfromtxt(d, delimiter=",")
    data.append(line)

final_data = np.concatenate(data)

radar = final_data[:,:8]
act_id = final_data[:,9]
sub_id = final_data[:,10]

radar_time =final_data[:,:2]
radar_freq = final_data[:,3:8]

def plot_confusion_matrix(cm, classes=[1,2,3,4,5,6,7],
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # if normalize:
    #     cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #     print("Normalized confusion matrix")
    # else:
    #     print('Confusion matrix, without normalization')
 
    print (cm)
 
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
 
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
 
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

print("Data Read")

num_subjects = 10
num_activities = 7

for i in range(1,num_subjects+1):
    print("subject = ",i)
    test_data = radar[sub_id==i]
    train_data = radar[sub_id!=i]
    test_label = act_id[sub_id==i]
    train_label = act_id[sub_id!=i]
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(train_data,train_label)
    y_pred = clf.predict(test_data)
    accuracy = accuracy_score(test_label,y_pred)
    f1 = f1_score(test_label,y_pred,average='weighted')
    precision = precision_score(test_label,y_pred,average='weighted')
    recall = recall_score(test_label,y_pred,average='weighted')
    print(classification_report(test_label,y_pred))
    C = confusion_matrix(test_label,y_pred)
    #C = C/C.astype(np.float).sum(axis=1)
    #print(C)
    plot_confusion_matrix(C)
    print("accuracy = ",accuracy)
    print("f1 score = ", f1)
    print("precision = ", precision)
    print("recall = ", recall)
    
