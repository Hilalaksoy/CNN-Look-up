from keras.models import load_model
from preprocessing.resizing import resize_area
from utility.data_generator import DataGenerator
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import keras
sess = tf.InteractiveSession()

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 16:08:23 2019

@author: Hilal
"""

def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    '''
    Compute the Hamming score (a.k.a. label-based accuracy) for the multi-label case
    http://stackoverflow.com/q/32239577/395857
    '''
    acc_list = []
    for i in range(y_true.shape[0]):
        y_pred[y_pred>=0.1] = 1
        y_pred[y_pred<0.1] = 0
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        #print('\nset_true: {0}'.format(set_true))
        #print('set_pred: {0}'.format(set_pred))
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/\
                    float( len(set_true.union(set_pred)) )
        #print('tmp_a: {0}'.format(tmp_a))
        acc_list.append(tmp_a)
    return np.mean(acc_list)


model= load_model('CNN2.model')

VAL_DB_NAME = './data/val_info.db'
VAL_DATA_PATH = './data/val2017'
TRAIN_DB_NAME = './data/train_info.db'
TRAIN_DATA_PATH = './data/train2017'

IMAGE_SIZE = 128
BATCH_SIZE = 256

val_data_gen = DataGenerator(VAL_DB_NAME, VAL_DATA_PATH,table_name='Images',image_size=IMAGE_SIZE, resizing_method='area', batch=BATCH_SIZE)
train_data_gen = DataGenerator(TRAIN_DB_NAME, TRAIN_DATA_PATH,table_name='TrainImages', image_size=IMAGE_SIZE, resizing_method='area', batch=BATCH_SIZE)
test_data_gen= DataGenerator(TRAIN_DB_NAME, TRAIN_DATA_PATH,table_name='TestImages', image_size=IMAGE_SIZE, resizing_method='area', batch=BATCH_SIZE)


a = test_data_gen.flow()
batch = next(a)

i = 10
for batch in a:
	y_pred = model.predict(batch[0])

	acc = keras.metrics.top_k_categorical_accuracy(batch[1], y_pred)

	print('keras :', acc.eval())
	print('hamming :', hamming_score(batch[1], y_pred))
	i -= 1
	if i == 0:
		break

	
sess.close()