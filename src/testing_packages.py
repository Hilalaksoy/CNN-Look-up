# -*- coding: utf-8 -*-

from utility.data_generator import DataGenerator
import tensorflow as tf
import matplotlib.pyplot as plt

VAL_DB_NAME = './data/val_info.db'
VAL_DATA_PATH = './data/val2017'
TRAIN_DB_NAME = './data/train_info.db'
TRAIN_DATA_PATH = './data/train2017'

IMAGE_SIZE = 128
BATCH_SIZE = 128

val_data_gen = DataGenerator(VAL_DB_NAME, VAL_DATA_PATH,table_name='Images',image_size=IMAGE_SIZE, resizing_method='area', batch=BATCH_SIZE)
train_data_gen = DataGenerator(TRAIN_DB_NAME, TRAIN_DATA_PATH,table_name='TrainImages', image_size=IMAGE_SIZE, resizing_method='area', batch=BATCH_SIZE)
test_data_gen= DataGenerator(TRAIN_DB_NAME, TRAIN_DATA_PATH,table_name='TestImages', image_size=IMAGE_SIZE, resizing_method='area', batch=BATCH_SIZE)


from keras import layers
from keras import models
from keras import optimizers

model = models.Sequential()
model.add(layers.Conv2D(32, (5, 5), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (5, 5), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (5, 5), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(92, activation='sigmoid'))

print(model.summary())


model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

with tf.device("/gpu:0"):
	history = model.fit_generator(
      train_data_gen.flow(),
      validation_data=val_data_gen.flow(),
      steps_per_epoch=10000//BATCH_SIZE,
      validation_steps=5000//BATCH_SIZE,
      epochs=10)
	model.save("training_model_1.model")
