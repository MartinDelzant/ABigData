#
#import sys
#
#sys.path.insert(0,"../../")
#
#from fonctions import *
#
#from sklearn.cross_validation import StratifiedKFold, cross_val_score
#from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
#
#data, y = loadTrainSet()
#
#cv = StratifiedKFold(y, n_folds=10, shuffle=True, random_state=41)
#
#from nltk.stem.wordnet import WordNetLemmatizer
#
#lem = WordNetLemmatizer()
#myFeat,data,_ = preprocess(data)
#
#tfidfWord = TfidfVectorizer( ngram_range=(1,3), strip_accents = "ascii" , stop_words = "english", binary = False)
#
#
#X = tfidfWord.fit_transform(data)
#
#useful_cols = np.array( (X!=0).sum(axis=0) > 2).ravel()
#
#X = X[:,useful_cols]
#
#from sklearn.linear_model import SGDClassifier
#
#model = SGDClassifier(alpha = 10**(-5),loss="log", penalty = "l2", n_iter = int(10**6/X.shape[0]),n_jobs = -1 )
#
#from sklearn.cross_validation import train_test_split
#
#X_train,X_test, y_train, y_test = train_test_split(X,y,test_size = 0.33, random_state = 42)
#data_train,data_test,y_train,y_test = train_test_split(data,y,test_size = 0.33,random_state = 42)
#
#model.fit(X_train,y_train)
#
#y_test_pred = model.predict(X_test)
#y_test_pred_proba = [ a[1] for a in model.predict_proba(X_test)]
#
#y_res1 = (np.array(list(map(lambda x : x[1] , model.predict_proba(X_test)))) >= 0.5).astype(int)
#y_res2 = (np.array(list(map(lambda x : x[1] ,model.predict_proba(X_test)))) >= 0.505).astype(int)
#
#print("Precision du modèle normal :" + str((y_res1 == y_test).sum()/len(y_res1)) )
#print("Precision du modèle pessimiste :" + str((y_res2 == y_test).sum()/len(y_res2)))
#
#proba_juste = list()
#
#for i in range(0,len(y_test_pred)) :
#    if y_test_pred[i] == 1 :
#        proba_juste.append(y_test_pred_proba[i])
#    else :
#        proba_juste.append(1 - y_test_pred_proba[i])
#
#data_misclassified_triple = [ (data[i],y_test[i],proba_juste[i]) for i in range(0,len(y_test)) if (y_test_pred != y_test)[i] ]
##last element : proba d'avoir le bon truc
#
#y_train_pred = model.predict(X_train)
#proba_juste_train = []
#
#y_train_pred_proba = [a[1] for a in model.predict_proba(X_train)]
#
#for i in range(0,len(y_train_pred)) :
#    if y_train_pred[i] == 1 :
#        proba_juste_train.append(y_train_pred_proba[i])
#    else :
#        proba_juste_train.append(1-y_train_pred_proba[i])
#
#proba_juste_train = np.array(proba_juste_train)
#
#filter2 = np.logical_not(np.logical_and(proba_juste_train > 0.7, model.predict(X_train) != y_train))
#
#X_train2 = X_train[filter2]
#y_train2 = y_train[filter2]
#
#model2 = SGDClassifier(alpha = 10**(-5),loss="log", penalty = "l2", n_iter = int(10**6/X.shape[0]),n_jobs = -1 )
#
#model2.fit(X_train2,y_train2)
#
y_res_biz = model2.predict(X_test)

print("Précision du modèle sans valeurs bizzares :" + str((y_res_biz == y_test).sum()/len(y_res_biz)))



