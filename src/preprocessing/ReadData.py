# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 15:55:18 2019

@author: hilal
"""
import sqlite3
import cv2

con=sqlite3.connect("../val_info.db")
cursor=con.cursor()

class Image():
    def __init__(self,id,fileName,height,width,matrix):
        self.id=id
        self.fileName=fileName
        self.height=height
        self.width=width
        #self.annotation=annotation
        self.matrix=matrix

#Belli aralıktaki resimlerin özelliklerini alma.    
def GetBatch(_from,_to):
    ret_list=[]
    cursor.execute("Select id,file_name,height,width From Images")
    lis=cursor.fetchall()  #istenilen satırdaki veri
    list1=lis[_from:_to]
    for i in list1:
        id_value=i[0]
        fileName_value=i[1]
        height_value=i[2]
        width_value=i[3]          
        print(fileName_value)    
        matrix_value=ToPixels(fileName_value)
        image=Image(id_value,fileName_value,height_value,width_value,matrix_value)
        ret_list.append(image)
        
    return ret_list
    
#Resmin piksellerini matrise çevirme.
def ToPixels(filename):
    imgPath= "./val2017img/val2017/"+filename
    pixel=cv2.imread(imgPath)   
    return pixel





