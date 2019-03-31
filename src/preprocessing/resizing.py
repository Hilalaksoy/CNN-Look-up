# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 21:32:32 2019

@author: hilal aksoy
"""
import cv2


def resize_area(image, size):
    """resize with OpenCv ."""
    dim = (size, size)
    return cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

def resize_preserve_aspect(image, max_size):
    height, width, channel = image.shape
    
    if width > height:
        ratio = width / max_size
        new_width = max_size
        new_height = int(height // ratio)
    else:
        ratio = height / max_size
        new_width = int(width // ratio)
        new_height = max_size
    
    return cv2.resize(image, (new_width, new_height), interpolation = cv2.INTER_AREA) 

def resize_with_padding(image, max_size):
    
    image = resize_preserve_aspect(image, max_size)
    height, width, channel = image.shape
    
    delta_w = max_size - width
    delta_h = max_size - height
        
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    
    color = [0, 0, 0]
    
    return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

#def resize_with_nearest_neighbor(image, max_size):
    


if __name__=="__main__":
    """resize with OpenCv ."""
    #img = cv2.imread('./val2017img/val2017/000000000139.jpg', cv2.IMREAD_UNCHANGED)
    #print(img)
    #print('Original Dimensions : ',img.shape)
    #resize_with_matrix(img,64)


