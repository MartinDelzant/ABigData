# -*- coding: utf-8 -*-
import pandas as pd 
import numpy as np
import os
import re
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import nltk
import string
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def loadData(train=True, verbose=False):
    """
    loadData() for the training set
    loadData(False) for the testing set
    """
    def loadTemp(path, verbose=False):
        data = []
        if verbose:
            i=0
        for _, _, files in os.walk(path):
            for file in files:
                if verbose and i%100 == 0:
                    print(i, file)
                with open(path+"/"+file, 'r') as content_file:
                    content = content_file.read() #assume that there are NO "new line characters"
                    data.append(content)
        return data
    data = []
    if train:
        data.extend(loadTemp('./data/train/pos', verbose=False))
        data.extend(loadTemp('./data/train/neg', verbose=False))
    else:
        data.extend(loadTemp('data/test'))
    return np.array(data)
    
def loadTrainSet(shuffle=False, dataFrame=False, verbose=False):
    data = loadData(verbose=verbose)
    label = np.array([1]*12500 + [0]*12500)
    
    if shuffle:
        pass #TODO maybe ?
        
    if dataFrame:
        return pd.DataFrame({'data':data, 'label':label})
        
    return data, label

def myFeatures(string):
    return [
    len(string),
    string.count('.'),
    string.count('!'),
    string.count('?'),
    len(re.findall(r'\W',string)),
    len(re.findall(r'10', string)),
    len(re.findall(r'[0-9]', string)),
    string.count('<')
    ]

def lemmatize(data):
	data2=[None]*len(data)
	wordnet_lemmatizer = WordNetLemmatizer()

	for doc_id, text in enumerate(data):
		# Tokenization
		tokens=nltk.word_tokenize(text.decode("utf-8"))
		
		# Lemmatize each text
		doc = ' '.join([wordnet_lemmatizer.lemmatize(w,pos='v') for w in tokens])
		data2[doc_id] = doc
	return data2
		
def preprocess(data):
    data = [ re.sub(r"<.*?>"," ",text) for text in data ] # Non-greedy regex !! 
    punctuation = set(string.punctuation)
    stemmer = PorterStemmer()
    #data2 = [ [ stemmer.stem(m.lower() for m in re.sub(text,regex_punctuation," ").split() if m.lower() not in punctuation ]  for text in data2]
    data = [ " ".join([ stemmer.stem(m) for m in nltk.word_tokenize(text.decode("utf-8")) if m not in punctuation ]) for text in data]   
    # re.sub(r"<.*>","",review2)
    #TODO features a la mano !!
    #TODO : Count break ...
	return data
    
def plot_roc_curve(y_true, probas, fig_args = dict(), **kwargs):
    """
    probas : Probability of having class 1
    """
    fpr, tpr, thres = roc_curve(y_true, probas)
    myauc = auc(fpr,tpr)
    plt.figure(**fig_args)
    plt.plot(fpr, tpr, label="AUC: %0.3f"%(myauc), **kwargs)
    plt.plot([0,1], [0,1], '--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()
