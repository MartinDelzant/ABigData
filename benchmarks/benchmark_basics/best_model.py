import sys

sys.path.insert(0,"../../")

from fonctions import *
from sklearn.cross_validation import StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

data, y = loadTrainSet()

cv = StratifiedKFold(y, n_folds=10, shuffle=True, random_state=41)

from nltk.stem.wordnet import WordNetLemmatizer

lem = WordNetLemmatizer()
myFeat,data,_ = preprocess(data)

tfidfWord = TfidfVectorizer( ngram_range=(1,3), strip_accents = "ascii" , stop_words = "english", binary = False)

X = tfidfWord.fit_transform(data)

useful_cols = np.array( (X!=0).sum(axis=0) > 2).ravel()

X = X[:,useful_cols]

from sklearn.linear_model import SGDClassifier

model = SGDClassifier(alpha = 10**(-5),loss="log", penalty = "l2", n_iter = int(10**6/X.shape[0]))

scores_accuracy = cross_val_score(model, X, y, cv=cv, n_jobs=3)
print("Accuracy :\n", round(np.mean(scores_accuracy), 4), "+/-", round(2*np.std(scores_accuracy),4))

# 0.9024

