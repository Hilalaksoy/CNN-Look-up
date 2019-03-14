# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 23:28:33 2019

@author: hilal
"""

import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator 




def ResizeImageData(validation_data_dir,img_width,img_height):
    val_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
    
    val_generator = val_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    class_mode='input')
    
    inputs, targets = next(val_generator)
    print(inputs)

def resizeDeneme(img):
    width, height = img.shape[0], img.shape[1]
    img = image.array_to_img(img, scale=False)

    # Crop 48x48px
    desired_width, desired_height = 64, 64

    if width < desired_width:
        desired_width = width
    start_x = np.maximum(0, int((width-desired_width)/2))

    img = img.crop((start_x, np.maximum(0, height-desired_height), start_x+desired_width, height))
    img = img.resize((64, 64))

    img = image.img_to_array(img)
    return img / 255.



if __name__=="__main__":
    
    img_width, img_height = 256, 256 
    
    validation_data_dir = './val2017img/val2017'
    nb_validation_samples = 5000
    
    ResizeImageData(validation_data_dir,img_width,img_height)
    
    
   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    