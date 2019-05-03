# -*- coding: utf-8 -*-

from keras.models import load_model
from preprocessing.resizing import resize_area
import cv2
import json
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 16:08:23 2019

@author: Hilal
"""

import numpy as np

model= load_model('256_area/CNN2.model')

etiket = None
images = ['cat.jpg', 'cow.jpg', 'pc.jpg','redapple.jpg','zebra.jpg', 'dog.jpg', 'dining_table.jpg' ,'apple.jpg', 'banana.jpg', 'bus.jpg',
		   'cup_knife.jpg', 'person_with_umbrella.jpg', 'red_truck.jpg', 'sheeps.jpg']

with open('etiket.json', 'r') as fi:
    etiket = json.loads(fi.read())

resimler = []
for f in images:
	resimler.append(cv2.imread('./images/' + f))

for i, img in enumerate(resimler):
    resimler[i] = resize_area(img, 256)

arr = np.array(resimler)

vals = model.predict(arr)

first_5 = []
for val in vals:
    first_5.append(np.argsort(val)[::-1][:10])


for i, labels in enumerate(first_5):
    print(i, '.', images[i])
    for label in labels:
        print(etiket[str(label)])
