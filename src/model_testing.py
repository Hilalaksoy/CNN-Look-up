# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 16:08:23 2019

@author: Hilal
"""

import numpy as np
from keras.models import load_model

model=load_model('C:/Users/Hilal/Documents/GitHub/CNN-Look-up/src/training_model_1.model')

import json
with open('C:/Users/Hilal/Documents/GitHub/CNN-Look-up/src/history_1.json', 'r') as fi:
    etiket = json.loads(fi.read())
#    
#import cv2
#    
#def resize_area(image, size):
#    """resize with OpenCv ."""
#    dim = (size, size)
#    return cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
#
#resim1 = cv2.imread('C:/Users/Hilal/Desktop/000000000019.jpg')
#
#resimler=[]
#
#resimler.append(resize_area(resim1,128))
#
#arr = np.array(resimler)
#
#vals = model.predict(arr)
#
#resim1_predict = vals[0].argmax(axis=0)
#
#print(vals[0])
	
	
