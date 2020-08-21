import math
from time import time
import tensorflow as tf
import pandas as pd
import numpy as np
import csv
from nltk.corpus import stopwords

def rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(np.log1p(y) - np.log1p(y0), 2)))
