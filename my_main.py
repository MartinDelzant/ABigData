# -*- coding: utf-8 -*-

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.cross_validation import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
# from sklearn.decomposition import NMF, LatentDirichletAllocation
from fonctions import *
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn import cross_validation

from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier

from time import time

from sklearn import metrics

from sklearn.grid_search import GridSearchCV


# load data
print("Loading training set")
data, y = loadTrainSet()
cv = StratifiedKFold(y, n_folds=5, shuffle=True, random_state=41)

# preprocess data
print("BeautifulSoup...")
data = use_beautifulsoup(data)
data = lemmatize(data)

# Different tokenizer
# Word_tokenizer
from nltk.tokenize import word_tokenize

# Tweet_tokenizer
## useful when terms like "baaaaad" or ".........."
tweet_to = TweetTokenizer(strip_handles=True,
	reduce_len=True).tokenize

# TF-IDF
# tweet_tokenizer
print("TF-IDF...")
tfidf_tweet = TfidfVectorizer(tokenizer = tweet_to,
	stop_words = 'english',
	ngram_range=(1,3),
	sublinear_tf=True)
X_tweet = tfidf_tweet.fit_transform(data) ## 553 081 features
# TO DO : count number of uppercase / words with uppercase

# word_tokenizer



# split train / test
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_tweet, y, test_size= 0.2 , random_state = 42)


# cross-validation
## MultinomialNB
print("MultinomialNB...")
model = MultinomialNB(alpha=0.5)
## return accuracy scores
scores = [model.fit(X_tweet[train], y[train]).score(X_tweet[test], y[test])
for train, test in cv] ## as a list
scores1 = cross_validation.cross_val_score(model, X_tweet, y, cv=cv) ## as an array
print("Accuracy: %0.2f (+/- %0.2f)" % (scores1.mean(), scores1.std() * 2))


# benchmarks
print("benchmarks...")
def benchmark(clf, X_tr, y_tr, X_ts, y_ts):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_tr, y_tr)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_ts)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(y_ts, pred)
    print("accuracy:   %0.3f" % score)

    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time



results1=[]
for clf, name in (
        #(RidgeClassifier(alpha = 0.1, tol=1e-1, solver="sag"), "Ridge Classifier"),
        #(Perceptron(n_iter=200, random_state = 42), "Perceptron"),
        (PassiveAggressiveClassifier(C=0.099, n_iter=200, loss='hinge',random_state = 42), "Passive-Aggressive"),
        #(KNeighborsClassifier(n_neighbors=10), "kNN"),
        #(RandomForestClassifier(n_estimators=500, n_jobs = 3), "Random forest"),
        ):
    print('=' * 80)
    print(name)
    results1.append(benchmark(clf))

## First classifiers
# ridge classifier : 0.904
# perceptron : 0.891
# passive-agressive : 0.906
# k-NN : 0.826
# random forest : 0.853

for penalty in ["l2", "l1"]:
    print('=' * 80)
    print("%s penalty" % penalty.upper())
    # Train Liblinear model
    results.append(benchmark(LinearSVC(loss='l2', penalty=penalty,
                                            dual=False, tol=1e-3)))

    # Train SGD model
results.append(benchmark(SGDClassifier(loss = 'log', alpha=.00001, l1_ratio=0.01, epsilon= 0.01, n_iter=100,
                                           penalty='l1')))
results.append(benchmark(LogisticRegression(C= 0.1, solver = 'lbfgs', penalty = 'l2', n_jobs = 3)))

from sklearn.linear_model import LogisticRegression


## L2 penalty
# LinearSVC : 0.905
# SGDClassifier : 0.899

## L1 penalty
# Linear SVC : 0.891
# SGDClassifier : 0.863


# Train SGD with Elastic Net penalty
print('=' * 80)
print("Elastic-Net penalty")
results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                       penalty="elasticnet")))

#Elastic Net penalty
# SGDClassifier : 0.885

# Train NearestCentroid without threshold
print('=' * 80)
print("NearestCentroid (aka Rocchio classifier)")
results.append(benchmark(NearestCentroid()))

# Train sparse Naive Bayes classifiers
print('=' * 80)
print("Naive Bayes")
results.append(benchmark(MultinomialNB(alpha=.01)))
results.append(benchmark(BernoulliNB(alpha=0.5)))

# Naive Bayes : 0.88....

# GridSearch
## RidgeClassifier

#ridge = RidgeClassifier(tol=1e-3, solver="lsqr") 
#alphas = np.logspace(-6, -1, 100)
#clf = GridSearchCV(estimator=ridge, param_grid=dict(alpha=alphas), n_jobs = 3)
#clf.fit(X_train, y_train)

# feature_selection
# selection from model
from sklearn.feature_selection import SelectFromModel

clf = PassiveAggressiveClassifier(C=0.099, n_iter=200, loss='hinge',random_state = 42)

sfm = SelectFromModel(clf, threshold = 0.001)

sfm.fit(X_train, y_train)

X_train_select = sfm.transform(X_train)
X_test_select = sfm.transform(X_test)

# test with new clf
clf1 = PassiveAggressiveClassifier(C=0.5, n_iter=200, loss='hinge',random_state = 42)

benchmark(clf1, X_train_select, y_train, X_test_select, y_test)

# GridSearch for C
# Set the parameters by cross-validation
tuned_parameters = [{'C': np.logspace(-6, 0, 1000)}]

score = 'accuracy'

print("# Tuning hyper-parameters for %s" % score)
print()

clf = GridSearchCV(clf1, tuned_parameters, cv=cv)
clf.fit(X_train, y_train)

print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
print("Grid scores on development set:")
print()
for params, mean_score, scores in clf.grid_scores_:
    print("%0.3f (+/-%0.03f) for %r"
          % (mean_score, scores.std() * 2, params))
print()