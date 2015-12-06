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

def preprocess(data):
    data = [ re.sub(r"<.*>"," ",text) for text in data ]
    punctuation = set(string.punctuation)
    stemmer = PorterStemmer()
    #data2 = [ [ stemmer.stem(m.lower() for m in re.sub(text,regex_punctuation," ").split() if m.lower() not in punctuation ]  for text in data2]
    data = [ " ".join([ stemmer.stem(m) for m in nltk.word_tokenize(text.decode("utf-8")) if m not in punctuation ]) for text in data]   
    # re.sub(r"<.*>","",review2)
    #TODO features a la mano !!
    #TODO : Count break ... 
    return data
    
def plot_roc_curve(y_true, probas, fig_args = dict(), **kwargs):
    """
    probas : Probability of having class 1
    """
    fpr, tpr, thres = roc_curve(y_true, probas)
    myauc = auc(fpr,tpr)
    plt.figure(**fig_args)
    plt.plot(fpr, tpr, label="AUC: %0.3f"%(myauc), **kwargs)
    plt.legend()
    plt.show()


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
