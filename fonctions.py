# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import re
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import nltk
import string
from sys import getfilesystemencoding as myencoding
#import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def loadData(train=True, verbose=False):
    """
    loadData() for the training set
    loadData(False) for the testing set
    """
    def loadTemp(path, verbose=False):
        data = []
        if verbose:
            i = 0
        for _, _, files in os.walk(path):
            for file in files:
                if verbose and i % 100 == 0:
                    print(i, file)
                with open(path+"/"+file, 'r', encoding=myencoding()) as content_file:
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
    all_notes = re.findall(r'[0-9]0? *?/ *?10', string)
    if len(all_notes)>1:
        print(all_notes)
    return [
        len(string),
        string.count('.'),
        string.count('!'),
        string.count('?'),
        len(re.findall(r'[^0-9a-zA-Z_ ]', string)),  # Non aplha numeric
        len(re.findall(r'10', string)),
        len(re.findall(r'[0-9]', string)),
        string.count('<'),
        len(re.findall(r'star(s)?', string)),
        np.mean([int(x.split('/')[0].strip()) for x in all_notes]) if all_notes else -1,
        len(re.findall(r'[A-Z]', string))
        ]


def lemmatize(data):
	wordnet_lemmatizer = WordNetLemmatizer()
	return [' '.join([
        wordnet_lemmatizer.lemmatize(w,pos='v') for w in nltk.word_tokenize(text)
        ]) for text in data]


def pos_tag(tokenized_sentence):
    return [nltk.pos_tag(token) for token in tokenized_sentence]


def preprocess(data, lemmatizer=None, stemmer=None):

    def preprocess_string(sentence, lemmatizer=lemmatizer, stemmer=stemmer):
        myfeat = myFeatures(sentence)
        tokenized_sentence = nltk.word_tokenize(re.sub(r"<.*?>", " ", sentence))
        # POS tagging :
        # postag = '##'.join(list(zip(*nltk.pos_tag(tokenized_sentence)))[1])
        if lemmatizer is not None:  # by default lemmatize. Else stem...
            tokenized_sentence = [lemmatizer.lemmatize(word, pos='v')
                                  for word in tokenized_sentence]
        if stemmer is not None:
            tokenized_sentence = [stemmer.stem(word)
                                  for word in tokenized_sentence]
        tokenized_sentence = " ".join(tokenized_sentence)
        return myfeat, tokenized_sentence

    myfeat, tokenized_data = list(zip(*map(preprocess_string, data)))
    return myfeat, tokenized_data, loadPostag()

def plot_roc_curve(y_true, probas, fig_args=dict(), **kwargs):
    """
    probas : Probability of having class 1
    """
    fpr, tpr, thres = roc_curve(y_true, probas)
    myauc = auc(fpr, tpr)
    plt.figure(**fig_args)
    plt.plot(fpr, tpr, label="AUC: %0.3f" % (myauc), **kwargs)
    plt.plot([0, 1], [0, 1], '--')
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


def loadPostag():
    postag = []
    for i in range(25000):
        with open("postag/postag_%u.txt" % (i), "r") as myfile:
            postag.append(myfile.readline())
    return postag
