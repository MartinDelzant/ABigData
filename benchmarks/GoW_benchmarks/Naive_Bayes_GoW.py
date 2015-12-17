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

data, y = loadTrainSet()
cv = StratifiedKFold(y, n_folds=10, shuffle=True, random_state=41)

from load_unload_sparse_matrix import *

data_GoW = load_sparse_csr("Intermediate_data/TrainSet/GoW_features/data.txt.npz")

model = MultinomialNB(alpha=1.1)
scores_accuracy = cross_val_score(model, data_GoW, y, cv=cv, n_jobs=-1)
scores_roc_auc = cross_val_score(model, data_GoW, y, cv=cv, n_jobs=-1, scoring="roc_auc")
print("Accuracy :\n", round(np.mean(scores_accuracy), 4), "+/-", round(2*np.std(scores_accuracy),4))
print("ROC AUC : \n", round(np.mean(scores_roc_auc), 4), "+/-", round(2*np.std(scores_roc_auc),4))
