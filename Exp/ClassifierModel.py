#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 00:13:51 2018

@author: mgy
"""

import numpy as np
import pickle

from sklearn.linear_model import LogisticRegression as LR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.svm import SVC
#from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier as MLP

from sklearn.model_selection import train_test_split as TTS, GridSearchCV as GCV
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, classification_report

from joblib import Parallel, delayed
import multiprocessing

writefile = 'TrainingResult.txt'

# please pick names among models below
Decision = ['SVM.sav', 'LakerSVM.sav']
Probability = ['KNN.sav', 'DT.sav', 'LR.sav', 'AdaBoost.sav', 'RandomForest.sav', 'LakerAdaBoost.sav', 'LakerRandomForest.sav']

class TrainingModel():
    def Predict(self, X, ModelName):
        X = preprocessing.scale(X)
        try:
            loaded_model = pickle.load(open(ModelName, 'rb'))
            if ModelName in Decision:
                returnVector = loaded_model.decision_function(X)
            elif ModelName in Probability:
                returnVector = (loaded_model.predict_proba(X).T)[1].T
                # I am not sure about which function should XGBoost call
                # yet, dont try it
            return returnVector
        except Exception as e:
            print(e)
            return
