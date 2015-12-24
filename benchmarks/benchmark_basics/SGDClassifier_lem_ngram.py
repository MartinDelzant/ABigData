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
from nltk.stem.wordnet import WordNetLemmatizer

print("Loading training set")
data, y = loadTrainSet()
cv = StratifiedKFold(y, n_folds=10, shuffle=True, random_state=41)

lem = WordNetLemmatizer()

print("preprocess ...")
myFeat,data,_ = preprocess(data,lemmatizer = lem)


print("Tfidf ...")

tfidfWord = TfidfVectorizer( ngram_range=(1,2),
    min_df=2, max_df=0.95, strip_accents = True , stop_words = "english", binary = True)

X = tfidfWord.fit_transform(data)

from sklearn.grid_search import GridSearchCV
#from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

# GridSearch sur alpha ?
model = SGDClassifier(alpha = 10**(-5),loss="log", penalty = "elasticnet", n_iter = int(10**6/X.shape[0]))
# model = LogisticRegressionCV(Cs = np.logspace(-5,4,10),penalty = "l2",solver=  "lbfgs")
#'liblinear')

parameters = { "alpha" : np.logspace(-5,-3,5) }
# 10**(-5) seems like a good choice

#param_grid = [ {'C': [1, 10, 100, 1000]}] 
#clf = GridSearchCV(model,parameters,n_jobs = 3,cv = cv)



# Printing scores and roc curve :
#model = MultinomialNB(alpha=0.5)
scores_accuracy = cross_val_score(model, X, y, cv=cv, n_jobs=3)
#scores_roc_auc = cross_val_score(model, X, y, cv=cv, n_jobs=1, scoring="roc_auc")
print("Accuracy :\n", round(np.mean(scores_accuracy), 4), "+/-", round(2*np.std(scores_accuracy),4))
#print("ROC AUC : \n", round(np.mean(scores_roc_auc), 4), "+/-", round(2*np.std(scores_roc_auc),4))
