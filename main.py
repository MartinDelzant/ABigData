# -*- coding: utf-8 -*-

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.cross_validation import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.grid_search import GridSearchCV
# from sklearn.decomposition import NMF, LatentDirichletAllocation
from fonctions import *
from scipy import sparse

print("Loading training set")
data, y = loadTrainSet()
cv = StratifiedKFold(y, n_folds=5, shuffle=True, random_state=41)

print("preprocess ...")
myFeat,data, pos_tag = preprocess(data)

print("Tfidf ...")
# Stop words : Yes /No ?
# regex : Yes / No ?
# sublinear tf : Y/N ?
# ...
tfidfWord = TfidfVectorizer(ngram_range=(1, 2),
    min_df=2, max_df=0.95)
X = tfidfWord.fit_transform(data)
inv_voc = {v: k for k, v in tfidfWord.vocabulary_.items()}
print('tfidf char')
tfidfChar = TfidfVectorizer(ngram_range=(3, 5),
	min_df=2, max_df=0.95, analyzer='char')
X_char = tfidfChar.fit_transform(data)
inv_vocChar = {v :k for k, v in tfidfChar.vocabulary_.items()}

print('CountPOS tfidf')
# Not tested yet ...
countPOS = CountVectorizer(tokenizer=lambda x: x.split("##"),
	ngram_range=(1,4), lowercase=False, min_df=2)
pos_tag = countPOS.fit_transform(pos_tag)


print("k Best...")  # Selecting the Kbest to see which word come out first.
kBest = SelectKBest(chi2, k=25)
kBest.fit(X, y)
print('"\t"'.join([inv_voc[index] for index in np.argsort(kBest.scores_)[::-1][:25]]))

print("k Best...")  # Selecting the Kbest to see which char come out first.
kBestChar = SelectKBest(chi2, k=25)
kBestChar.fit(X_char, y)
print('"\t"'.join([inv_vocChar[index] for index in np.argsort(kBestChar.scores_)[::-1][:25]]))

# TODO : hstack the matrices
params = {"selectkbest__k":list(range(10000, 1000000, 5000)), "multinomialnb__alpha":np.logspace(-3,3,10)}
# Printing scores and roc curve :
models = [ # MultinomialNB(alpha=0.5), 
GridSearchCV(make_pipeline(SelectKBest(chi2), MultinomialNB()), params, cv=cv, verbose=1),
GridSearchCV(make_pipeline(SelectKBest(f_classif), MultinomialNB()), params, cv=cv, verbose=1)
]
for model in models:
    model.fit(sparse.hstack((X, X_char)),y)
    scores_accuracy = cross_val_score(model, X, y, cv=cv, n_jobs=-1)
    scores_roc_auc = cross_val_score(model, X, y, cv=cv, n_jobs=-1, scoring="roc_auc")

    print("Accuracy :\n", round(np.mean(scores_accuracy), 4), "+/-", round(2*np.std(scores_accuracy),4))
    print("ROC AUC : \n", round(np.mean(scores_roc_auc), 4), "+/-", round(2*np.std(scores_roc_auc),4))
"""
#creating the cross_val predict and predict_proba :
y_pred_proba = np.zeros((y.shape[0], 2))
for train_idx, test_idx in cv:
    model.fit(X[train_idx], y[train_idx])
    y_pred_proba[test_idx] = model.predict_proba(X[test_idx])
y_pred = np.argmax(y_pred_proba, axis=1)

# Reports and roc curve
print(classification_report(y,y_pred))
plot_roc_curve(y, y_pred_proba[:,1], fig_args=dict(figsize=(8,8)))
"""
print("Done !")
