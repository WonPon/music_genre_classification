# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 21:03:32 2022

@author: abhin
"""
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn import metrics

from sklearn.metrics import classification_report

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import matplotlib.pyplot as plt
import joblib,  statistics
from sklearn.model_selection import GridSearchCV 
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm 
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import SelectKBest,chi2,mutual_info_classif,RFE
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from collections import Counter
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
import csv,os,re,sys,codecs
from sklearn.decomposition import PCA

class data_classification():
     def __init__(self,path='',clf_opt='lr',no_of_selected_features=None):
        self.path = path
        self.clf_opt=clf_opt
        self.no_of_selected_features=no_of_selected_features
        if self.no_of_selected_features!=None:
            self.no_of_selected_features=int(self.no_of_selected_features) 

# Selection of classifiers  
     def classification_pipeline(self):    
    # AdaBoost 
        if self.clf_opt=='ab':
            print('\n\t### Training AdaBoost Classifier ### \n')
            be1 = svm.LinearSVC(class_weight='balanced')            
            be2 = LogisticRegression(solver='liblinear',class_weight='balanced') 
            be3 =  DecisionTreeClassifier(max_depth=50)         
            clf = AdaBoostClassifier( DecisionTreeClassifier(max_depth=50), algorithm='SAMME.R',n_estimators=100,random_state=10)
            clf_parameters = {
            # 'clf__base_estimator':(be1,be2,be3),
            # 'clf__random_state':(0,10),
            # 'clf__n_estimators':(10,100,200)
             }      
 # Decision Tree
        elif self.clf_opt=='dt': 
            print('\n\t### Training Decision Tree Classifier ### \n')
            clf = DecisionTreeClassifier(random_state=40,criterion='entropy',max_features='auto',max_depth=40,ccp_alpha=0.01) 
            clf_parameters = {
            # 'clf__criterion':('gini', 'entropy'), 
            # 'clf__max_features':('auto', 'sqrt', 'log2'),
            # 'clf__max_depth':(10,40,45,60),
            # 'clf__ccp_alpha':(0.009,0.01,0.05,0.1),
            } 

    # Logistic Regression 
        elif self.clf_opt=='lr':
            print('\n\t### Training Logistic Regression Classifier ### \n')
            clf = LogisticRegression(solver='liblinear',class_weight='balanced',random_state=0) 
            clf_parameters = {
   #         'clf__random_state':(0,10),
            } 

    # Random Forest 
        elif self.clf_opt=='rf':
            print('\n\t ### Training Random Forest Classifier ### \n')
            clf = RandomForestClassifier(max_features=None,class_weight='balanced',criterion='entropy',max_depth=50,n_estimators=200)
            clf_parameters = {
          #  'clf__criterion':('entropy','gini'),       
          #  'clf__n_estimators':(30,50,100,200,300),
          #  'clf__max_depth':(10,20,30,50,100,200),
            }          
    # Support Vector Machine  
        elif self.clf_opt=='svm': 
            print('\n\t### Training SVM Classifier ### \n')
            clf = svm.SVC(kernel='rbf',class_weight='balanced',probability=True,C=100)  
            clf_parameters = {
          #  'clf__C':(0.01,0.1,1,100,1000),
            #'clf__kernel':('linear','rbf','polynomial'),
            }
    # KNN
        elif self.clf_opt=='knn':
            print('\n\t### Training KNN Classifier ### \n')
            clf=  KNeighborsClassifier(n_neighbors=3,weights='distance',metric='manhattan')
            clf_parameters = {
            # 'clf__n_neighbors':range(1,50),
            # 'clf__weights':('uniform','distance'),
      #      'clf__leaf_size':range(5,100),
     #       'clf__algorithm':('auto','ball_tree','kd_tree','brute'),
             # 'clf__metric':('euclidean','manhattan','chebyshev','minkowski')
            }
    # MLP
        elif self.clf_opt=='mlp':
            print('\n\t### Training MLP Classifier ### \n')
            clf=  MLPClassifier(hidden_layer_sizes=(1000,500,500), max_iter=600,solver='adam',activation='relu',learning_rate='constant')
            clf_parameters = {
            #'clf__activation':('identity', 'logistic', 'tanh', 'relu'),
            #'clf__solver':('lbfgs', 'sgd', 'adam'),
            #'clf__learning_rate':('constant','invscaling','adaptive')
            }
        else:
            print('Select a valid classifier \n')
            sys.exit(0)        
        return clf,clf_parameters     
# Load the data 
     def get_data(self):
        data = np.loadtxt('music30s_trainin.csv', delimiter=",",dtype=float,skiprows=1)
        labels  = np.loadtxt('music30s_trainlabel.csv', delimiter=",",dtype=object)

        data=np.delete(data,0,1)
        labels=np.delete(labels,0,0)
        labels=np.delete(labels,0,1)
        labels=np.reshape(labels,-1)
        stdscaler = preprocessing.StandardScaler()
        data = stdscaler.fit_transform(data)
        labels=labels.tolist()
              
        #Training and Test Split           
        trn_data, tst_data, trn_cat, tst_cat = train_test_split(data, labels, test_size=0.2, random_state=10)   

        return trn_data, tst_data, trn_cat, tst_cat
    
     def classification(self):  
   # the following commented code was used for experiment by dividing K-fold into training and test data
        trn_data, tst_data, trn_cat, tst_cat=self.get_data()

 #        skf = StratifiedKFold(n_splits=3)
 #        predicted_class_labels=[]; actual_class_labels=[]; 
 #        count=0; probs=[];
 #        for train_index, test_index in skf.split(trn_data,trn_cat):
 #            X_train=[]; y_train=[]; X_test=[]; y_test=[]
 #            for item in train_index:
 #                X_train.append(trn_data[item])
 #                y_train.append(trn_cat[item])
 #            for item in test_index:
 #                X_test.append(trn_data[item])
 #                y_test.append(trn_cat[item])
 #            count+=1                
 #            print('Training Phase '+str(count))
 #            clf,clf_parameters=self.classification_pipeline()
 #            pipeline = Pipeline([
 #                #        ('feature_selection', SelectKBest(chi2, k=self.no_of_selected_features)),                         # k=1000 is recommended 
 #                        ('feature_selection', SelectKBest(mutual_info_classif, k=self.no_of_selected_features)),        
 #                        ('clf', clf),])
 #            grid = GridSearchCV(pipeline,clf_parameters,scoring='f1_micro',cv=10)          
 #            grid.fit(X_train,y_train)     
 #            clf= grid.best_estimator_  
 #            # print('\n\n The best set of parameters of the pipiline are: ')
 #            # print(clf)     
 #            predicted=clf.predict(X_test)  
 #            predicted_probability = clf.predict_proba(X_test) 
 #            for item in predicted_probability:
 #                probs.append(float(max(item)))
 #            for item in y_test:
 #                actual_class_labels.append(item)
 #            for item in predicted:
 #                predicted_class_labels.append(item)           
 #        confidence_score=statistics.mean(probs)-statistics.variance(probs)
 #        confidence_score=round(confidence_score, 3)
 #        print ('\n The Probablity of Confidence of the Classifier: \t'+str(confidence_score)+'\n') 
       
 # #   Evaluation
        # class_names=list(Counter(tst_cat).keys())
        # class_names = [str(x) for x in class_names] 
 #        print('\n\n The classes are: ')
 #        print(class_names)      
 #        print('\n *************** Confusion Matrix ***************  \n')
 #        print (confusion_matrix(tst_cat, predicted))        
 #        print(classification_report(actual_class_labels, predicted_class_labels, target_names=class_names))        
        
        # Experiments on Given Test Data during Test Phase
   
    #    print('\n ***** Classifying Test Data ***** \n')   
        class_names=list(Counter(tst_cat).keys())
        class_names = [str(x) for x in class_names] 
        clf,clf_parameters=self.classification_pipeline()
        pipeline = Pipeline([
            #   ('feature_selection', SelectKBest(chi2, k=self.no_of_selected_features)),
            #  ('feature_selection', RFE(svm.SVC(kernel='linear'),n_features_to_select=self.no_of_selected_features)),                         # k=1000 is recommended 
            ('feature_selection', SelectKBest(mutual_info_classif, k=self.no_of_selected_features)),        
            ('clf', clf),])
        grid = GridSearchCV(pipeline,clf_parameters,scoring='accuracy',cv=10,n_jobs=-1)          
        grid.fit(trn_data,trn_cat)     
        clf= grid.best_estimator_ 
        # print(grid.best_params_)
        predicted=clf.predict(tst_data )
        # print(classification_report(tst_cat, predicted, target_names=class_names)) 
        # print('\n *************** Confusion Matrix ***************  \n')
        # cm = confusion_matrix(tst_cat, predicted, labels=clf.classes_) 
        print("Accuracy score is ", grid.best_score_)
        # disp = ConfusionMatrixDisplay(confusion_matrix=cm,
        #                       display_labels=clf.classes_)
        # disp.plot()
        # plt.show()   
        
        
classifiers=['ab',
              'dt',
              'lr',
              'rf',
              'svm',
              'knn',
              'mlp',
    ]

for clf in classifiers:
    data_classification('', clf_opt=clf,no_of_selected_features=45).classification()

#Best classifiers are adaboost and Random forest
data = np.loadtxt('music30s_trainin.csv', delimiter=",",dtype=float,skiprows=1)
labels  = np.loadtxt('music30s_trainlabel.csv', delimiter=",",dtype=object)
testdata= np.loadtxt('music30s_test.csv', delimiter=",",dtype=float,skiprows=1)
data=np.delete(data,0,1)
labels=np.delete(labels,0,0)
labels=np.delete(labels,0,1)
labels=np.reshape(labels,-1)
stdscaler = preprocessing.StandardScaler()
data = stdscaler.fit_transform(data)
labels=labels.tolist()
  
trn_data, tst_data, trn_cat, tst_cat = train_test_split(data, labels, test_size=0.2, random_state=10)   
clf = AdaBoostClassifier( DecisionTreeClassifier(max_depth=50), algorithm='SAMME.R',n_estimators=100,random_state=10).fit(trn_data,trn_cat)
predicted=clf.predict(tst_data )
class_names=list(Counter(tst_cat).keys())
class_names = [str(x) for x in class_names] 
print(classification_report(tst_cat, predicted, target_names=class_names)) 
cm = confusion_matrix(tst_cat, predicted, labels=clf.classes_) 
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                      display_labels=clf.classes_)
test_labels=clf.predict(testdata)

#printing test labels on ADAboost
DF = pd.DataFrame(test_labels)
  
# save the test data classes dataframe as a csv file
DF.to_csv("music30s_testlabels.csv", header=False, index=False)
