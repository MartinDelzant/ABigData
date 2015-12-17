# -*- coding: utf-8 -*-

import sys
sys.path.insert(0,"../../")


import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn
from fonctions import *
from sklearn.linear_model import LogisticRegressionCV

from nltk.stem.wordnet import WordNetLemmatizer

data, y = loadTrainSet()
cv = StratifiedKFold(y, n_folds=10, shuffle=True, random_state=41)


from load_unload_sparse_matrix import *

data_GoW = load_sparse_csr("Intermediate_data/TrainSet/GoW_features/data.txt.npz")

lem = WordNetLemmatizer()

myFeat,data,_ = preprocess(data, lemmatizer = lem)

GoW = data_Gow.shape[1]
n_features_tfidf = data_Gow.shape[1]

data_tot = scipy.sparse.hstack(data,data_GoW)
data_tot = scipy.sparse.hstack(data_tot,myFeat)

X_train,X_test,y_train,y_test =  train_test_split(data_tot,y)

model_tot = LogisticRegressionCV(Cs = np.logspace(-3,2),penalty = "l2",solver = "lbfgs")
model_GoW = LogisticRegressionCV(Cs = np.logspace(-3,2),penalty = "l2",solver = "lbfgs")
model_extraFeat = LogisticRegressionCV(Cs = np.logspace(-3,2),penalty = "l2",solver = "lbfgs")

n_tot_features = X_train.shape[1]

model_tot.train(X_train[,0:n_features],y_train)
model_GoW.train(X_train[,n_features:(n_features+n_features_tfidf),y_train)
model_extraFeat.train(X_train[,(n_features+n_features_tfidf):n_tot_features],y_train)

scores_accuracy = cross_val_score(model, data_GoW, y, cv=cv, n_jobs=-1)
scores_roc_auc = cross_val_score(model, data_GoW, y, cv=cv, n_jobs=-1, scoring="roc_auc")
print("Accuracy :\n", round(np.mean(scores_accuracy), 4), "+/-", round(2*np.std(scores_accuracy),4))
print("ROC AUC : \n", round(np.mean(scores_roc_auc), 4), "+/-", round(2*np.std(scores_roc_auc),4))
