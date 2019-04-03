# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 14:06:15 2019

@author: Hilal
"""

import json
import sqlite3

file=open('../instances_train2017.json','r')
strr=file.read()
jsObject=json.loads(strr)
jsObject.keys()
print(len(jsObject['images']))

con=sqlite3.connect("../train_info.db")
cursor=con.cursor()

def tablo_olustur():
    cursor.execute("CREATE TABLE IF NOT EXISTS  Licenses(id INT PRIMARY KEY, name TEXT,url TEXT)")
    cursor.execute("CREATE TABLE IF NOT EXISTS  Categories(supercategoryId INT,supercategory TEXT, id INT PRIMARY KEY,name TEXT, FOREIGN KEY (supercategoryId) REFERENCES Categories (id))")
    cursor.execute("CREATE TABLE IF NOT EXISTS  Images(id INT PRIMARY KEY,file_name TEXT,height INT,width INT,coco_url TEXT,date_captured TEXT,flickr_url TEXT,license INT, FOREIGN KEY (license) REFERENCES Licenses (id))")  
    cursor.execute("CREATE TABLE IF NOT EXISTS  Annotations(id INT PRIMARY KEY,area REAL,category_id INT,image_id INT,iscrowd INT, FOREIGN KEY (category_id) REFERENCES Categories (id),FOREIGN KEY (image_id) REFERENCES Images (id) )")
    cursor.execute("CREATE TABLE IF NOT EXISTS  Info(contributor TEXT,date_created TEXT,description TEXT,url TEXT,version TEXT,year INT)")
    con.commit()


def veri_ekleme_licenses():
    for i in range(len(jsObject['licenses'])):
        lisencesId=jsObject['licenses'][i]["id"]
        lisencesName=jsObject['licenses'][i]["name"]
        lisencesUrl=jsObject['licenses'][i]["url"]
        degerler3=(lisencesId,lisencesName,lisencesUrl)
        cursor.execute("INSERT INTO  Licenses(id,name,url) VALUES(?,?,?)",degerler3)
    con.commit()

#Categories tablosuna veri ekleme
def veri_ekleme_cat():
    supercategories = set()
    
    for cat in jsObject['categories']:
        supercategories.add(cat['supercategory'])
    supercategories_dict = {}    
    
    for i,cat in enumerate(supercategories):
        supercategories_dict[cat] = 100 + i
        cursor.execute("INSERT INTO Categories(supercategory,id,name) VALUES (?,?,?)",( '', 100 + i, cat))
        
    for category in jsObject['categories']:        
        kategori=category["supercategory"]       
        sayi=category["id"]
        isim=category["name"]
        supercategoryId = supercategories_dict[category["supercategory"]]
        degerler=(supercategoryId,kategori,sayi,isim)       
        cursor.execute("INSERT INTO Categories(supercategoryId,supercategory,id,name) VALUES (?,?,?,?)",degerler)
    con.commit()


def veri_ekleme_image():
    for image in jsObject['images']:
       degerler=(image['id'],image['file_name'], image['height'],
                 image['width'], image['coco_url'], image['date_captured'],
                 image['flickr_url'], image['license'])
       cursor.execute("INSERT INTO Images(id,file_name,height,width,coco_url,date_captured,flickr_url,license) VALUES (?,?,?,?,?,?,?,?)",degerler)
    con.commit()


def veri_ekleme_annotation():
    for annot in jsObject['annotations']:
       degerler=(annot['id'],annot['area'], annot['category_id'],
                 annot['image_id'], annot['iscrowd'])
       cursor.execute("INSERT INTO Annotations(id, area, category_id, image_id, iscrowd) VALUES (?,?,?,?,?)",degerler)
    con.commit()


def veri_ekleme_info():    
      contributor=jsObject['info']["contributor"]
      date_created=jsObject['info']["date_created"]
      description=jsObject['info']["description"]
      url=jsObject['info']["url"]
      version=jsObject['info']["version"]
      year=jsObject['info']["year"]
      degerler2=(contributor,date_created,description,url,version,year)
      cursor.execute("INSERT INTO Info(contributor,date_created,description,url,version,year) VALUES (?,?,?,?,?,?)",degerler2)
      con.commit()




#tablo_olustur()
#veri_ekleme_cat()
#veri_ekleme_info()
#veri_ekleme_image()
#veri_ekleme_licenses()
#veri_ekleme_annotation()
#con.close()
