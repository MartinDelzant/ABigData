import networkx as nx
import string
import pandas as pd
import numpy as np
import time
import re
import os.path
import math
from sklearn.feature_extraction.text import TfidfVectorizer
#num_documents: number of documents
#clean_train_documents: the collection
#unique_words: list of all the words we found 
#sliding_window: window size
#train_par: if true we are in the training documents
#idf_learned


def createGraphFeatures(clean_train_documents,sliding_window):
    num_documents = len(clean_train_documents)

    term_num_docs = {} #dictionay of each word with a count of that word through out the collections
    idf_col = {}#dictionay of each word with the idf of that word
   
    print("Creating unique word, starting idf") 
       
    tfidf = TfidfVectorizer(analyzer = "word",lowercase = True,norm = None)
    tfidf.fit(clean_train_documents)
    idf_col = { word : tfidf.idf_[k] for word,k in tfidf.vocabulary_.items()}
    
    unique_words = set(tfidf.vocabulary_.keys())
    unique_words_len = len(unique_words)
       
    print("Creating the graph of words for each document...")
    totalNodes = 0
    totalEdges = 0
    features = np.zeros((num_documents,len(unique_words)))#where we are going to put the features

    #go over all documents
    for i in range( 0,num_documents ):
        wordList1 = clean_train_documents[i].split(None)
        wordList2 = [string.rstrip(x.lower(), ',.!?;') for x in wordList1]
        docLen = len(wordList2)
        #the graph
        dG = nx.Graph()

        if len(wordList2)>1:
            populateGraph(wordList2,dG,sliding_window)
            dG.remove_edges_from(dG.selfloop_edges())
            centrality = nx.degree_centrality(dG) #dictionary of centralities (node:degree)

            totalNodes += dG.number_of_nodes()
            totalEdges += dG.number_of_edges()
            
            for k,node_term in enumerate(dG.nodes()):
                if node_term in idf_col :
                    features[i,unique_words.index(node_term)] = centrality[node_term]*idf_col[node_term]
    nodes_ret=term_num_docs.keys()
    #return 1: features, 2: idf values (for the test data), 3: the list of terms 
    return features, idf_col, nodes_ret
    
    
def populateGraph(wordList,dG,sliding_window):
    for k,word in enumerate(wordList):
        if not dG.has_node(word):
            dG.add_node(word)
        tempW = sliding_window
        if k +sliding_window > len(wordList):
            tempW = len(wordList)-k
        for j in range(1,tempW):
            next_word = wordList[k+j]
            dG.add_edge(word,next_word)
            
            
def countWords(wordList,term_num_docs):
    found=set()    
    for k,word in enumerate(wordList):
        if word not in found :
            found.add(word)
            if word in term_num_docs:
                term_num_docs[word]+=1
            else :
                term_num_docs[word]=1
