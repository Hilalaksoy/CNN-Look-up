# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 21:32:32 2019

@author: hilal
"""
import cv2
import tensorflow as tf
from matplotlib import pyplot as plt


def resize_with_matrix(matrix,size):
    """resize with OpenCv ."""
    dim=(size,size)
    resized = cv2.resize(matrix, dim, interpolation = cv2.INTER_AREA)
    print('Resized Dimensions : ',resized.shape)
    cv2.imshow("Resized image", resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def resize_with_padding(matrix,size): 
    
    resize_nearest_neighbor = tf.image.resize_images(matrix, size=[size,size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return resize_nearest_neighbor


if __name__=="__main__":
    """resize with OpenCv ."""
    #img = cv2.imread('./val2017img/val2017/000000000139.jpg', cv2.IMREAD_UNCHANGED)
    #print(img)
    #print('Original Dimensions : ',img.shape)
    #resize_with_matrix(img,64)
    
    """resize with Padding"""
    filename=tf.placeholder(tf.string,name="inputFile")
    fileContent= tf.read_file(filename,name="loadFile")
    image=tf.image.decode_jpeg(fileContent,name="DecodeJpeg")
    
    
    #Tensorflow işlemlerini çalıştırmak için sınıf-Session
    sess=tf.Session()
    feed_dict={filename:"./val2017img/val2017/000000000139.jpg"}
    
    #as_default varsayılan olarak bu nesneyi döndürme
    with sess.as_default():
         actualImage=image.eval(feed_dict)  #Resmin matris hali.        
         plt.imshow(actualImage)
         plt.title("original image")
         plt.show()
         
         new_image=resize_with_padding(actualImage,64).eval(feed_dict)
         print(new_image)
         plt.imshow(new_image)
         plt.title("resize_image_with_pad")
         plt.show()
       
