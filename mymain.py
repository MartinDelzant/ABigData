# -*- coding: utf-8 -*-

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.cross_validation import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.grid_search import GridSearchCV
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.svm import LinearSVC
# from sklearn.decomposition import NMF, LatentDirichletAllocation
from fonctions import *
from scipy import sparse
import nltk
from operator import itemgetter
from sklearn.linear_model import SGDClassifier

# Utility function to report best scores
def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.5f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

def xtrain_test():
	print("Loading training set")
	data, y = loadTrainSet()
	data_test = loadData(train=False)
	all_data = [*data, *data_test]
	print("Tfidf ...")
	# Stop words : Yes /No ?
	# regex : Yes / No ?
	# sublinear tf : Y/N ?
	# ...
	tfidfWord = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.90, tokenizer=nltk.word_tokenize)
	tfidfWord.fit(all_data)
	inv_voc = {v: k for k, v in tfidfWord.vocabulary_.items()}

	tfidfChar = TfidfVectorizer(ngram_range=(3, 5), min_df=2, max_df=0.90, analyzer='char')
	tfidfChar.fit(all_data)
	inv_vocChar = {v :k for k, v in tfidfChar.vocabulary_.items()}

	return y, sparse.hstack((tfidfWord.transform(data), tfidfChar.transform(data)), format="csr"), sparse.hstack((tfidfWord.transform(data_test), tfidfChar.transform(data_test)), format="csr")

#y, X_train, _ = xtrain_test()
print(X_train.shape)
cv = StratifiedKFold(y, n_folds=5, shuffle=True, random_state=41)
# LinearSVC for k in 1000:100000:10000 -> best score k ~ 80000 (pen l2) C ~ 1 -> 90.9 en crossval
for k in range(70000, 130001, 10000):
    for f in [chi2]:
        newX = SelectKBest(f, k).fit_transform(X_train,y)
        print("transformed")
        for alpha in [0,*np.logspace(-20,-9,12)]:
            print("alpha", alpha, "k", k)
            print(cross_val_score(MultinomialNB(alpha=alpha), newX, y, cv=cv).mean())
print("Done")
models = [
(Pipeline([('kbest', SelectKBest()), 
#('model', SGDClassifier(average=10))]),
('model', MultinomialNB())]),
 {'kbest__score_func':[chi2, f_classif], 'kbest__k':list(range(10000, 110001, 10000)),
 #"model__penalty":[ 'l2','l1','elasticnet'],"model__loss":['log'], "model__alpha":np.logspace(-7,2,10),"model__n_iter":[50]
"model__alpha":np.logspace(-5,4,10)})
]

for model, params in models:
	gdcv = GridSearchCV(model, params, cv=cv, verbose=3)
	gdcv.fit(X_train, y)
	report(gdcv.grid_scores_, n_top=10)

