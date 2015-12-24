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
tweet_to = TweetTokenizer(strip_handles=True, reduce_len=True).tokenize
## 553 081 features


# TF-IDF
# tweet_tokenizer
tfidf_tweet = TfidfVectorizer(tokenizer = tweet_to,
	stop_words = 'english',
	ngram_range=(1,3),
	min_df=2,
	max_df=0.95,
	sublinear_tf=True)
X_tweet = tfidf_tweet.fit_transform(data) ## 553 081 features


# word_tokenizer

# cross-validation
## MultinomialNB
model = MultinomialNB(alpha=0.5)
## return accuracy scores
scores = [model.fit(X_tweet[train], y[train]).score(X_tweet[test], y[test])
for train, test in cv] ## as a list
scores1 = cross_validation.cross_val_score(model, X_tweet, y, cv=cv) ## as an array
print("Accuracy: %0.2f (+/- %0.2f)" % (scores1.mean(), scores1.std() * 2))

