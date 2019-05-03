# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-

from utility.data_generator import DataGenerator
import tensorflow as tf
import matplotlib.pyplot as plt
import keras.backend as K
import numpy as np

VAL_DATA_PATH = './data/val2017'
DB_NAME = './data/train_info.db'
TRAIN_DATA_PATH = './data/resized/train/'
TEST_DATA_PATH = './data/resized/test/'
VAL_DATA_PATH = './data/resized/val/'

IMAGE_SIZE = 128
BATCH_SIZE = 128
NUM_CLASSES = 80
TRAIN_BATCH_NO = 640
TEST_BATCH_NO = 274
VAL_BATCH_NO = 40


from keras import models
from keras.layers import Conv2D, Activation, MaxPooling2D, Dropout, Flatten, Dense


model = models.Sequential()
model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(NUM_CLASSES, activation='sigmoid'))

print(model.summary())

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['binary_accuracy', 'top_k_categorical_accuracy'])


x_train = np.array([], dtype=np.int64).reshape(0, IMAGE_SIZE, IMAGE_SIZE, 3)
y_train = np.array([], dtype=np.int64).reshape(0, NUM_CLASSES)
for i in range(TRAIN_BATCH_NO):
	x = np.load(TRAIN_DATA_PATH + 'x' + str(i) + '.npy')
	y = np.load(TRAIN_DATA_PATH + 'y' + str(i) + '.npy')
	x_train = np.concatenate((x_train, x), axis=0)
	y_train = np.concatenate((y_train, y), axis=0)

	
x_val = np.array([], dtype=np.int64).reshape(0, IMAGE_SIZE, IMAGE_SIZE, 3)
y_val = np.array([], dtype=np.int64).reshape(0, NUM_CLASSES)
for i in range(VAL_BATCH_NO):
	x = np.load(VAL_DATA_PATH + 'x' + str(i) + '.npy')
	y = np.load(VAL_DATA_PATH + 'y' + str(i) + '.npy')
	x_val = np.concatenate((x_val, x), axis=0)
	y_val = np.concatenate((y_val, y), axis=0)


#from keras.callbacks import ModelCheckpoint
#filepath="weights.best.hdf5"
#checkpoint = ModelCheckpoint(filepath, monitor='binary_crossentropy', save_best_only=True,  verbose=1, mode='min')
#
#callbacks_list = [checkpoint]

history = model.fit(
          x_train, y_train,
		  batch_size=BATCH_SIZE,
          validation_data=(x_val, y_val),
          epochs=10)

model.save("CNN6.model")

import json

with open('history6.json', 'w') as w:
	w.write(json.dumps(history.history))



