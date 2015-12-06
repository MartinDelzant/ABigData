# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 16:13:54 2015

@author: martin
"""

import pandas as pd 
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.cross_validation import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
import re
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, classification_report, auc
from nltk.stem.porter import PorterStemmer
import nltk
import string
from fonctions import *


print("Loading training set")
data, y = loadTrainSet()
cv = StratifiedKFold(y, n_folds=10, shuffle=True, random_state=41)

#TO UNCOMMENT : data = preprocess(data)

#TODO features a la main !!
print("Tfidf ...")
tfidf = TfidfVectorizer(ngram_range=(1,2))
X = tfidf.fit_transform(data)
inv_voc = {v:k for k,v in tfidf.vocabulary_.items()}

print("k Best...") # Selecting the Kbest to see which word come out first. 
kBest = SelectKBest(chi2, k=25)
kBest.fit(X,y)
print(' '.join([inv_voc[index] for index in kBest.get_support(indices=True)]))

## Printing scores and roc curve :
model = MultinomialNB(alpha=0.5)
scores_accuracy = cross_val_score(model, X,y,cv=cv, n_jobs=-1)
scores_roc_auc = cross_val_score(model, X,y,cv=cv, n_jobs=-1, scoring="roc_auc")
print("Accuracy :\n", round(np.mean(scores_accuracy), 4), "+/-", round(2*np.std(scores_accuracy),4))
print("ROC AUC : \n", round(np.mean(scores_roc_auc), 4), "+/-", round(2*np.std(scores_roc_auc),4))

#creating the cross_val predict and predict_proba :
y_pred_proba = np.zeros((y.shape[0],2))
for train_idx, test_idx in cv:
    model.fit(X[train_idx], y[train_idx])
    y_pred_proba[test_idx] = model.predict_proba(X[test_idx])
y_pred = np.argmax(y_pred_proba, axis=1)

# Reports and roc curve
print(classification_report(y,y_pred))
plot_roc_curve(y, y_pred_proba[:,1], fig_args=dict(figsize=(8,8)))

print("Done !")
