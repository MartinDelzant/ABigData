import numpy as np
import pandas as pd
from fonctions import *
from nltk.parse.malt import MaltParser

print("Loading training set")
data, y = loadTrainSet()

malt.MaltParser(working_dir="/home/rohith/malt-1.7.2",
	mco="engmalt.linear-1.7",
	additional_java_args=['-Xmx512m'])
