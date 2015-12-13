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