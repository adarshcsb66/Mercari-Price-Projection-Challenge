import math
from time import time
import tensorflow as tf
import pandas as pd
import numpy as np
import csv
import gensim
from nltk.corpus import stopwords
import gzip

def rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(np.log1p(y) - np.log1p(y0), 2)))
#create trial document
model = gensim.models.Word2Vec (documents, size=50, window=10, min_count=2)
model.train(documents,total_examples=len(documents),epochs=10)
