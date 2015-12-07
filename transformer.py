# -*- coding: utf-8 -*-

from nltk import word_tokenize
import string
from nltk.stem import LancasterStemmer, PorterStemmer, SnowballStemmer
from nltk.stem import WordNetLemmatizer


class MyTextPreprocessor:

    """A preprocessor for text :
    - Stemmer
    - Lemmatizer
    ...
    """

    def __init__(self, name="porter", tokenizer=word_tokenize,
                 join_str=" ", punctuation=set(string.punctuation),
                 output_an_array=False, *args, **kwargs):
        """
        name : string. Can be 'porter', 'lancaster' or 'snowball'

        tokenizer : callable or None.
        If None, the sentences are supposed to be already tokenized !

        args : argument for the transformer

        kwargs : key words args for the transformer
        """
        self.name = name
        self.tokenizer = tokenizer
        self.join_str = join_str
        self.punctuation = punctuation
        self.output_an_array = output_an_array
        self.args = args
        self.kwargs = kwargs

    def fit(self, X, y=None):
        """
        X is expected to be a 1d array of text
        """
        if self.name == "porter":
            self.transformer = PorterStemmer(*self.args, **self.kwargs)
            self.transform_method = self.transformer.stem
        elif self.name == "lancaster":
            self.transformer = LancasterStemmer(*self.args, **self.kwargs)
            self.transform_method = self.transformer.stem
        elif self.name == "snowball":
            self.transformer = SnowballStemmer(*self.args, **self.kwargs)
            self.transform_method = self.transformer.stem
        elif self.name == "wordnet":
            self.transformer = WordNetLemmatizer(*self.args, **self.kwargs)
            self.transform_method = self.transformer.lemmatize
        else:
            raise NameError("""the transformer's name can be 'porter',
                'lancaster', 'snowball' or 'wordnet'""")

    def transform(self, X, y=None):
        if self.tokenizer is None:
            result = [[self.transform_method(word)
                       for word in list_words]
                      for list_words in X]
        else:
            result = [[self.transform_method(word)
                       for word in self.tokenizer(sentence)
                       if word not in self.punctuation]
                      for sentence in X]
        if self.output_an_array:
            return result
        else:
            return list(map(self.join_str.join, result))

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)
