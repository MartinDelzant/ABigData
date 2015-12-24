


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


## Multinomial NB
# cross-validation
model = MultinomialNB(alpha=0.5)
## return accuracy scores
scores = [model.fit(X_hash[train], y[train]).score(X_hash[test], y[test])
for train, test in cv] ## as a list
scores1 = cross_validation.cross_val_score(model, X_hash, y, cv=cv) ## as an array
print("Accuracy: %0.2f (+/- %0.2f)" % (scores1.mean(), scores1.std() * 2))

# score = 0.86

## SVM
# GridSearchCV
from sklearn.grid_search import GridSearchCV
from sklearn import svm

svc = svm.SVC(C=1,kernel='linear')

Cs = np.logspace(-6, -1, 10)
clf = GridSearchCV(estimator=svc, param_grid=dict(C=Cs), n_jobs = 2)
clf.fit(X_train, y_train)

### ... TO FINISH

# SGDClassifier
n = X_train.shape[0]
model = SGDClassifier(penalty='elasticnet',n_iter = np.ceil(10**6 / n),shuffle=True)

print("CV starts.")
# run grid search
param_grid = [{'alpha' : np.logspace(-7, -1, 10),'l1_ratio': np.linspace(0, 0.99, 0.1)}]
gs = grid_search.GridSearchCV(model,param_grid,n_jobs=3,verbose=1)
gs.fit(X_train, y_train)

print("Scores for alphas:")
print(gs.grid_scores_)
print("Best estimator:")
print(gs.best_estimator_)
print("Best score:")
print(gs.best_score_)
print("Best parameters:")
print(gs.best_params_)

# RandomForestClassifier

# ExtraTreesClassifier

# GradientBoostingCLassifier

