# -*- coding: utf-8 -*-
import os
from nltk.parse import stanford
os.environ['STANFORD_PARSER'] = 'C:/Users/Guillaume/Documents/Scolarite/Master Data Sciences/Projets/stanford-parser-full-2014-08-27'
os.environ['STANFORD_MODELS'] = 'C:/Users/Guillaume/Documents/Scolarite/Master Data Sciences/Projets/stanford-parser-full-2014-08-27'
os.environ['JAVAHOME'] = 'C:/Program Files (x86)/Java/jdk1.8.0_60/bin'

parser=stanford.StanfordParser(model_path="C:/Users/Guillaume/Documents/Scolarite/Master Data Sciences/Projets/englishPCFG.ser.gz")
sentences = parser.raw_parse_sents(("Hello, my name is Melroy","What is your name?"))

print(sentences)

#draw the tree
for line in sentences:
	for sentence in line:
		sentence.draw()

dep_parser=stanford.StanfordDependencyParser(model_path="C:/Users/Guillaume/Documents/Scolarite/Master Data Sciences/Projets/englishPCFG.ser.gz")
print([parse.tree() for parse in dep_parser.raw_parse("The quick brown fox jumps over the lazy dog.")])

## split paragraph into sentences
import nltk.data
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
print((tokenizer.tokenize(data[0])))

# display attributes in a sentence as a list
list(parser.raw_parse("the quick brown fox jumps over the lazy dog")) # return attribute of each word!
[list(parse.triples()) for parse in dep_parser.raw_parse("The quick brown fox jumps over the lazy dog.")]

[ ' '.join([a[1],b, c[1]]) for a,b,c in rel1[0]] #return VBZ nsubj NN


## use BeautfiulSoup to avoid encoding problems...
from bs4 import BeautifulSoup 
example1 = BeautifulSoup(ex1[1])
example1.get_text()
### TO FINISH

## Apply lemmatizer to data before
d=[]
for i,j in enumerate(data):
	print("YO", i)
	e=[]
	ex1=tokenizer.tokenize(j)
	rel1=[]


	for i in ex1:
		rel1=[list(parse.triples()) for parse in dep_parser.raw_parse(i)]
		e.extend([' '.join([a[1],b, c[1]]) for a,b,c in rel1[0]])


	d.append(e)


# count all the 2-grams relations
rel1[0].count((('jumps', 'VBZ'), 'nsubj', ('fox', 'NN'))) ## return 1
from collections import Counter
Counter(rel1[0]) ## return each item and its number of occurence
# actually, first I want to just count the different relations in a sentence
for a in rel1[0]:
	print(a[1]) ## print nsubj, det, amod...


## TO DO: return the length of each sentence
## TO DO: implement Beautiful Soup
## TO DO: implement stopwords in TFidf
## TO DO: implement log in TFIDF
## TO DO: add list of negative words count & weigth in myFeatures

for c in ex1[1]:
    print('%s,%d' %(c , ex1[1].index(c)))