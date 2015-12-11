# -*- coding: utf-8 -*-

import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, BaggingClassifier
from fonctions import loadTrainSet
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.multiclass import OneVsOneClassifier

_, y = loadTrainSet()
cv = StratifiedKFold(y, n_folds=5, shuffle=True, random_state=41)


def cross_val(clf_name, X, y, cv=cv, proba=False, score=accuracy_score, *params, **kwargs):
    if clf_name == "extra":
        c = ExtraTreesClassifier(12, max_depth=23, max_features=10, n_jobs=-1, *params, **kwargs)
    elif clf_name == "grad":
        c = GradientBoostingClassifier(n_estimators=40, learning_rate=0.1, *params, **kwargs)
    elif clf_name == "cgrad":
        c = CalibratedClassifierCV(base_estimator=GradientBoostingClassifier(n_estimators = 20,learning_rate= 0.1, *params, **kwargs), method='isotonic', cv=10) 
    elif clf_name == "cmulti":
        c = CalibratedClassifierCV(base_estimator=MultinomialNB(alpha = alpha_multi, *params, **kwargs), method='isotonic', cv=10) 
    elif clf_name == "multi":
        c = MultinomialNB(alpha=0.5, *params, **kwargs)
    elif clf_name == "bag":
        c = BaggingClassifier(base_estimator=MultinomialNB(alpha = 0.5, *params, **kwargs),n_estimators = 100,n_jobs = -1)
    elif clf_name == "bern":
        c = BernoulliNB(alpha=0.00000000001, *params, **kwargs)
    elif clf_name == "gauss":
        c = GaussianNB(*params, **kwargs)
    elif clf_name == "random":
        c = RandomForestClassifier(1200,max_depth= 23,max_features = 10,n_jobs = -1, *params, **kwargs)
    elif clf_name == "lda":
        c = LinearDiscriminantAnalysis(*params, **kwargs)
    elif clf_name == "logistic":
        c = LogisticRegression(C=1, *params, **kwargs)
    elif clf_name == "svm":
        c = LinearSVC(C=100, *params, **kwargs)
    elif clf_name == "knn":
        c = KNeighborsClassifier(n_neighbors=20, *params, **kwargs)
    elif clf_name == "near":
        c = NearestCentroid(*params, **kwargs)
    elif clf_name == "ridge":
        c = OneVsOneClassifier(RidgeClassifier(alpha=0.1, *params, **kwargs))
    elif clf_name == "sgd":
        c = SGDClassifier(loss="hinge", penalty="l2", n_iter=50, alpha=0.000001, fit_intercept=True, average=True)

    y_pred = np.zeros(y.shape)
    score_list = []
    for i, (train, test) in enumerate(cv):
        c.fit(X[train], y[train])
        if proba:
            y_pred[test] = c.predict_proba(X[test])
        else:
            y_pred[test] = c.predict(X[test])
        score_list.append(score(y[test], y_pred[test]))
        print(score_list[i])
    return y_pred


def cross_val_warm(clf_name, X, y, n_estimators_grid=range(10, 500, 50), *params, **kwargs):
    if "sklearn" in str(type(clf_name)):
        c = clf_name
    if clf_name == "random":
        c = RandomForestClassifier(warm_start=True, oob_score=True, *params, **kwargs)
    elif clf_name == "bag":
        c = BaggingClassifier(base_estimator=MultinomialNB(alpha=0.5, *params, **kwargs), n_estimators=100, n_jobs=-1, *params, **kwargs)
    if clf_name == "extra":
        c = ExtraTreesClassifier(*params, warm_start=True, **kwargs)
    for n_est in np.sort(n_estimators_grid):
        c.set_params(n_estimators=n_est)
        c.fit(X, y)
        print(str(n_est)+"\t"+str(c.oob_score_))
    return c
