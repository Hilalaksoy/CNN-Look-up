# -*- coding: utf-8 -*-

from preprocessing.read_data import *
from preprocessing.resizing import *
import cv2
import matplotlib.pyplot as plt


DB_NAME = './data/val_info.db'
DATA_PATH = './data/val2017'
IMAGE_SIZE = 64
VALIDATION_SPLIT = 0.2

#i = 0
#for resim in db_iterator(db_name, data_path):
#    
#    plt.imshow(resize_area(resim.get_pixel_matrix(), 256))
#    plt.show()
#    if i > 6:
#        break
#    i += 1
    

def data_gen(batch=32):
    while True:
        for chunk in get_chunks(db_iterator(DB_NAME, DATA_PATH),batch):
            x = []
            y = []
            for resim in chunk:
                x.append(resize_area(resim.get_pixel_matrix(), IMAGE_SIZE))
                y.append(resim.labels)
            x = np.array(x)
            y = np.array(y)            
            yield (x, y)
    return

        
from keras import layers
from keras import models
from keras import optimizers

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(92, activation='sigmoid'))

print(model.summary())



model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])



history = model.fit_generator(
      data_gen(50),
      steps_per_epoch=100,
      epochs=5)