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
models=[0]*1
for j in range(len(models)):
  models[j]=Sequential()
  models[j].add(Conv2D(32,(3,3),strides=(1,1),name="first",activation='relu',input_shape=(50,70,1)))
  models[j].add(BatchNormalization())
  models[j].add(AveragePooling2D((2,2),strides=(2,2)))
  models[j].add(Conv2D(64,(5,5),strides=(1,1),activation='relu'))
  models[j].add(BatchNormalization())
  models[j].add(AveragePooling2D((2,2),strides=(2,2)))
  models[j].add(Flatten())
  models[j].add(Dense(units=512,activation='relu'))
  models[j].add(Dense(units=256,activation='relu'))
  models[j].add(Dense(units=10,activation='relu'))
  models[j].compile(optimizer="adam",loss='categorical_crossentropy',metrics=['accuracy'])
