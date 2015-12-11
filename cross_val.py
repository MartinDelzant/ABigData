# -*- coding: utf-8 -*-

import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from fonctions import *
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import accuracy_score

_, y = loadTrainSet()
cv = StratifiedKFold(y, n_folds=5, shuffle=True, random_state=41)


def cross_val(model_name, X, y, cv=cv, proba=False, score=accuracy_score):
    if clf_name == "extra":
        c = ExtraTreesClassifier(12, max_depth=23, max_features=10, n_jobs=-1)
    elif clf_name == "grad":
        c = GradientBoostingClassifier(n_estimators=40, learning_rate=0.1)
    elif clf_name == "cgrad":
        c = CalibratedClassifierCV(base_estimator=GradientBoostingClassifier(n_estimators = 20,learning_rate= 0.1), method='isotonic', cv=10) 
    elif clf_name == "cmulti":
        c = CalibratedClassifierCV(base_estimator=MultinomialNB(alpha = alpha_multi), method='isotonic', cv=10) 
    elif clf_name == "multi":
        c = MultinomialNB(alpha=alpha_multi)
    elif clf_name == "bag":
        c = BaggingClassifier(base_estimator=MultinomialNB(alpha = 0.008),n_estimators = 100,n_jobs = -1)
    elif clf_name == "bern":
        c = BernoulliNB(alpha=0.00000000001)
    elif clf_name == "gauss":
        c = GaussianNB()
    elif clf_name == "random":
        c = RandomForestClassifier(1200,max_depth= 23,max_features = 10,n_jobs = -1)
    elif clf_name == "lda":
        c = LDA()
    elif clf_name == "logistic":
        c = LogisticRegression(C = 1)
    elif clf_name == "svm":
        c = LinearSVC(C=100)
    elif clf_name == "knn":
        c = KNeighborsClassifier(n_neighbors=20)
    elif clf_name == "near":
        c = NearestCentroid()
    elif clf_name == "ridge":
        c = OneVsOneClassifier(RidgeClassifier(alpha = 0.1))
    elif clf_name == "sgd":
        c = SGDClassifier(loss="hinge", penalty="l2",n_iter=50,alpha=0.000001,fit_intercept=True,average=True)

    y_pred = np.zeros(y.shape)
    score_list = []
    for i, (train, test) in enumerate(cv):
        model.fit(X[train], y[train])
        if proba:
            y_pred[test] = model.predict_proba(X[test])
        else:
            y_pred[test] = model.predict(X[test])
        score_list.append(score(y[test], y_pred[test]))
        print(score_list[i])
    return y_pred
