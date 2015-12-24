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

# load data
print("Loading training set")
data, y = loadTrainSet()
cv = StratifiedKFold(y, n_folds=5, shuffle=True, random_state=41)

# preprocess data
print("BeautifulSoup...")
data = use_beautifulsoup(data)
data = lemmatize(data)

# HashingVectorizer
## with tweet tokenizer
tweet_to = TweetTokenizer()
vectorizer = HashingVectorizer(tokenizer = lambda doc: tweet_to.tokenize(doc),
	stop_words='english',
	non_negative=True,
	ngram_range=(1,3))
X_hash = vectorizer.fit_transform(data)

# Printing scores and roc curve :
# split train / test
from sklearn import cross_validation
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_hash, y, test_size= 0.2 , random_state = 42)
clf = MultinomialNB(alpha = 0.5).fit(X_train, y_train)
clf.score(X_test, y_test) # score = 0.857

# cross-validation
model = MultinomialNB(alpha=0.5)
[model.fit(X_hash[train], y[train]).score(X_hash[test], y[test])
for train, test in cv]
### TO DO : return best score / average score +- ecart type