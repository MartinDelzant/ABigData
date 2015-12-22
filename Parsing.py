# -*- coding: utf-8 -*-
import os
from nltk.parse import stanford
os.environ['STANFORD_PARSER'] = 'C:/Users/Guillaume/Documents/Scolarite/Master Data Sciences/Projets/stanford-parser-full-2014-08-27'
os.environ['STANFORD_MODELS'] = 'C:/Users/Guillaume/Documents/Scolarite/Master Data Sciences/Projets/stanford-parser-full-2014-08-27'
os.environ['JAVAHOME'] = 'C:/Program Files (x86)/Java/jdk1.8.0_60/bin'
dep_parser=stanford.StanfordDependencyParser(model_path="C:/Users/Guillaume/Documents/Scolarite/Master Data Sciences/Projets/englishPCFG.ser.gz")
import nltk.data
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

### Load data
from fonctions.py import *
data, y = loadTrainSet()

## Clean data
## use BeautifulSoup to avoid encoding problems...
from bs4 import BeautifulSoup
for i, item in enumerate(data):
	data[i] = BeautifulSoup(item).get_text()

## erase all accents
import re
for i, item in enumerate(data):
	data[i] = re.sub("[^0-9a-zA-Z.,;!?():-@%/''""&#=+]", " ", item)

## transform '!!!' into '!' to avoid tokenizing problems
for i, item in enumerate(data):
	data[i] = item.replace('!!!','! ')
	data[i] = data[i].replace('...','. ')
	data[i] = data[i].replace(';','. ') ## this is to avoid too long sentences
									   ## parser memory can't handle it
	data[i] = data[i].replace('(',' ') ## again, problem of tokenizing
	data[i] = data[i].replace(')',' ') ## again, problem of tokenizing

## TO DO : use it to count length sentences ! data[257]
## TO DO : draw a graph a maximum length sentence per document
## TO DO : number of sentences !

## Get all dependencies in a list
d=[]

for i,j in enumerate(data[317:25000]):
	print("YO", i+317)
	e=[]
	ex1=tokenizer.tokenize(j)
	ex2=[]
	ex3=[]
	for i, item in enumerate(ex1):
		if (len(item)>3) and (len(item)<=500):				## to avoid tokenizing problems like sentences '..' or '.'
			ex2.append(item)
		if len(item)> 500:
			ex3 = item.replace(',','.')
			ex2.extend(tokenizer.tokenize(ex3))
	for i, item in enumerate(ex2):
		ex2[i] = item.replace('.','')
	rel1=[]
	for i in ex2:
		rel1=[list(parse.triples()) for parse in dep_parser.raw_parse(i)]
		e.extend([' '.join([a[1],b, c[1]]) for a,b,c in rel1[0]])


	d.append(e)
## Last error : problem of memory for parsing.... ABANDON

## Apply CountVectorizer to the list of dependencies
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(tokenizer=lambda doc: doc,lowercase=False)

X_train_counts = count_vect.fit_transform(d)
X_train_counts.shape


## TO DO: return the length of each sentence
## TO DO: implement Beautiful Soup
## TO DO: implement stopwords in TFidf
## TO DO: implement log in TFIDF
## TO DO: add list of negative words count & weigth in myFeatures



#################################################### 
## Other functions, not to be used

#parser=stanford.StanfordParser(model_path="C:/Users/Guillaume/Documents/Scolarite/Master Data Sciences/Projets/englishPCFG.ser.gz")
#sentences = parser.raw_parse_sents(("Hello, my name is Melroy","What is your name?"))

#print(sentences)

#draw the tree
#for line in sentences:
#	for sentence in line:
#		sentence.draw()

#print([parse.tree() for parse in dep_parser.raw_parse("The quick brown fox jumps over the lazy dog.")])

## split paragraph into sentences
#print((tokenizer.tokenize(data[0])))

# display attributes in a sentence as a list
#list(parser.raw_parse("the quick brown fox jumps over the lazy dog")) # return attribute of each word!
#[list(parse.triples()) for parse in dep_parser.raw_parse("The quick brown fox jumps over the lazy dog.")]

#[ ' '.join([a[1],b, c[1]]) for a,b,c in rel1[0]] #return VBZ nsubj NN

# count all the 2-grams relations
#rel1[0].count((('jumps', 'VBZ'), 'nsubj', ('fox', 'NN'))) ## return 1
#from collections import Counter
#Counter(rel1[0]) ## return each item and its number of occurence
# actually, first I want to just count the different relations in a sentence
#for a in rel1[0]:
#	print(a[1]) ## print nsubj, det, amod...

#GET ENCODING OF STRING
#for c in ex1[1]:
#    print('%s,%d' %(c , ex1[1].index(c)))
