from fonctions import *
from GoW import *
from nltk.stem.wordnet import WordNetLemmatizer
from load_unload_sparse_matrix import *


lem =  WordNetLemmatizer()

data,y = loadTrainSet()

myFeat,data,postag = preprocess(data,lemmatizer = lem)


features, idf_col , nodes_ret  = createGraphFeatures(data,2)

features = features.to_csr()

save_sparse_csr("Intermediate_data/TrainSet/GoW_features/data.txt",features)

