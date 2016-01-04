
###
# Python 3.5 compatible
###
import nltk
from ntlk.stem import WordNetLemmatizer
from fonctions import loadData, loadTrainSet
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import cross_val_score
from scipy import sparse
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.linear_model import PassiveAggressiveClassifier

data, y = loadData()
data_test = loadTrainSet()

# Remove html
data = [re.sub(r"<.*?>", " ", sentence) for sentence in data]
data_test = [re.sub(r"<.*?>", " ", sentence) for sentence in data_test]

all_data = [*data, *data_test]

wnl = WordNetLemmatizer()
tokLemm = lambda sentence: [wnl.lemmatize(x, pos='v') 
                            for x in nltk.word_tokenize(sentence)]

tfidf = TfidfVectorizer(ngram_range=(1, 3), min_df=2,
                        max_df=0.9, tokenizer=tokLemm,
                        stop_words='english')
tfidf.fit(all_data)

tfidfChar = TfidfVectorizer(ngram_range=ngram_range_char,
                            min_df=2, max_df=0.9,
                            analyzer="char")
tfidfChar.fit(all_data)

X_train = sparse.hstack(
    (tfidfWord.transform(data),
     tfidfChar.transform(data)))

X_test = sparse.hstack(
    (tfidfWord.transform(data_test),
     tfidfChar.transform(data_test)))

model = make_pipeline(SelectKBest(chi2, 810000),
                      PassiveAggressiveClassifier(n_iter=40,
                                                  C=0.01,
                                                  loss='squared_hinge'))

# Change the scoring param to 'roc_auc' to see another metric on the model
cvs = cross_val_score(model, X_train, y, cv=cv,
                      n_jobs=1, scoring='accuracy').mean()
print('Cross validation score', cvs)

model.fit(X_train, y)
y_pred = model.predict(X_test)
