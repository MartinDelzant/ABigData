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

def loadData(train=True, verbose=False):
    """
    loadData() for the training set
    loadData(False) for the testing set
    """
    def loadTemp(path, verbose=False):
        data = []
        if verbose:
            i=0
        for _, _, files in os.walk(path):
            for file in files:
                if verbose and i%100 == 0:
                    print(i, file)
                with open(path+"/"+file, 'r') as content_file:
                    content = content_file.read() #assume that there are NO "new line characters"
                    data.append(content)
        return data
    data = []
    if train:
        data.extend(loadTemp('./data/train/pos', verbose=False))
        data.extend(loadTemp('./data/train/neg', verbose=False))
    else:
        data.extend(loadTemp('data/test'))
    return np.array(data)
    
def loadTrainSet(shuffle=False, dataFrame=False, verbose=False):
    data = loadData(verbose=verbose)
    label = np.array([1]*12500 + [0]*12500)
    
    if shuffle:
        pass #TODO maybe ?
        
    if dataFrame:
        return pd.DataFrame({'data':data, 'label':label})
        
    return data, label

def myFeatures(string):
    return [
    len(string),
    string.count('.'),
    string.count('!'),
    string.count('?'),
    len(re.findall(r'\W',string)),
    len(re.findall(r'10', string)),
    len(re.findall(r'[0-9]', string))
    ]

if __name__=='__main__':
    print("Loading training set")
    data, y = loadTrainSet()
    cv = StratifiedKFold(y, n_folds=10, shuffle=True, random_state=41)
    #TODO features a la mano !!
    print("Tfidf ...")
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(data)
    inv_voc = {v:k for k,v in tfidf.vocabulary_.items()}
    
    print("k Best...")
    kBest = SelectKBest(chi2, k=25)
    kBest.fit(X,y)
    print(' '.join([inv_voc[index] for index in kBest.get_support(indices=True)]))
    
    nb = MultinomialNB(alpha=0.5)
    scores_accuracy = cross_val_score(nb, X,y,cv=cv, n_jobs=-1)
    scores_roc_auc = cross_val_score(nb, X,y,cv=cv, n_jobs=-1, scoring="roc_auc")
    print("Accuracy :\n", round(np.mean(scores_accuracy), 4), "+/-", round(2*np.std(scores_accuracy),4))
    print("ROC AUC : \n", round(np.mean(scores_roc_auc), 4), "+/-", round(2*np.std(scores_roc_auc),4))
    print("Done !")