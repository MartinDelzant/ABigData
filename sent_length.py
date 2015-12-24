

# Sentence length tokenizer
# First, train sentence tokenizer with punktuation problems
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
punkt_param = PunktParameters()
punkt_param.abbrev_types = set(['dr', 'vs', 'mr', 'mrs', 'prof', 'inc'])
sentence_splitter = PunktSentenceTokenizer(punkt_param)
text = data[559]
sentences = sentence_splitter.tokenize(text)

import nltk.data
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')