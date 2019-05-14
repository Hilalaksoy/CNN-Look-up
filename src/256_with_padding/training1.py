from utility.data_generator import DataGenerator
import tensorflow as tf
import matplotlib.pyplot as plt
import keras.backend as K
import numpy as np
import keras
from keras import models
from keras.layers import Conv2D, Activation, MaxPooling2D, Dropout, Flatten, Dense ,SeparableConv2D

VAL_DATA_PATH = './data/val2017'
DB_NAME = './data/train_info.db'
TRAIN_DATA_PATH = './data/resized_128_with_padding/train/'
TEST_DATA_PATH = './data/resized_128_with_padding/test/'
VAL_DATA_PATH = './data/resized_128_with_padding/val/'
NUM_CLASSES = 80

# 256
# TRAIN_BATCH_NO = 1280
# TEST_BATCH_NO = 545
# VAL_BATCH_NO = 80
# IMAGE_SIZE = 256
# BATCH_SIZE = 64

# 128
TRAIN_BATCH_NO = 640
TEST_BATCH_NO = 274
VAL_BATCH_NO = 40
IMAGE_SIZE = 128
BATCH_SIZE = 128

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


model = models.Sequential()
model.add(Conv2D(16, (5, 5), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Conv2D(32, (4, 4), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Conv2D(128, (2, 2), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (1, 1), padding='same'))
model.add(Activation('relu'))

model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(NUM_CLASSES, activation='sigmoid'))

print(model.summary())

opt = keras.optimizers.rmsprop(lr=0.00005, decay=1e-6)

model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['binary_accuracy', 'top_k_categorical_accuracy'])

def gen(batch_no, path):
	while True:
		for i in range(batch_no):
			x = np.load(path + 'x' + str(i) + '.npy')
			y = np.load(path + 'y' + str(i) + '.npy')
			yield (x, y)


history = model.fit_generator(
          gen(TRAIN_BATCH_NO, TRAIN_DATA_PATH),
          validation_data=gen(TEST_BATCH_NO, TEST_DATA_PATH),
          steps_per_epoch=TRAIN_BATCH_NO,
          validation_steps=VAL_BATCH_NO,
          epochs=25)

model.save("256_with_padding/CNN1.model")

import json

with open('256_with_padding/history1.json', 'w') as w:
	w.write(json.dumps(history.history))



