# -*- coding: utf-8 -*-

import sys
sys.path.insert(0,"../../")


import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.cross_validation import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
# from sklearn.decomposition import NMF, LatentDirichletAllocation
from fonctions import *

print("Loading training set")
data, y = loadTrainSet()
cv = StratifiedKFold(y, n_folds=10, shuffle=True, random_state=41)

print("preprocess ...")
#myFeat,data = preprocess(data)
#myFeat, data, pos_tag = preprocess(data)
print("Tfidf ...")
# Stop words : Yes /No ?
# regex : Yes / No ?
# sublinear tf : Y/N ?
# ...

tfidfWord = TfidfVectorizer(# ngram_range=(1),
    min_df=2, max_df=0.95)
X = tfidfWord.fit_transform(data)

#from sklearn.grid_search import GridSearchCV
#from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV

model = LogisticRegressionCV(Cs = np.logspace(-5,4,10),penalty = "l2",solver=  "lbfgs")
#'liblinear')

#param_grid = [ {'C': [1, 10, 100, 1000]}] 


# Printing scores and roc curve :
#model = MultinomialNB(alpha=0.5)
scores_accuracy = cross_val_score(model, X, y, cv=cv, n_jobs=1)
scores_roc_auc = cross_val_score(model, X, y, cv=cv, n_jobs=1, scoring="roc_auc")
print("Accuracy :\n", round(np.mean(scores_accuracy), 4), "+/-", round(2*np.std(scores_accuracy),4))
print("ROC AUC : \n", round(np.mean(scores_roc_auc), 4), "+/-", round(2*np.std(scores_roc_auc),4))
