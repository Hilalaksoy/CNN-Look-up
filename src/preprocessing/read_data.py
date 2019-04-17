"""
Created on Sat Feb 16 15:55:18 2019

@author: hilal aksoy
"""
from itertools import islice
import sqlite3
import numpy as np
import os
from preprocessing.image import Image

def get_chunks(iterable, max_size):
    """Yields chunks from iterable with given max_size"""
    sourceiter = iter(iterable)
    while True:
        batchiter = islice(sourceiter, max_size)
        chunk = [x for x in batchiter]
        if len(chunk) != 0:
            yield chunk
        else:
            return

def db_iterator(db_name, data_path,table_name):
    """db_name veritabanindaki resimleri al"""
    db_connection = sqlite3.connect(db_name)
    cursor = db_connection.cursor()

    values = None
    categories = None
    annotations = None
    try:
        values = cursor.execute("Select id,file_name,height,width From"+table_name).fetchall()
        categories = cursor.execute("Select supercategoryId,supercategory,id,name From Categories").fetchall()
        annotations = cursor.execute("Select id, area, category_id, image_id, iscrowd From Annotations").fetchall()
    except sqlite3.ProgrammingError as e:
        print('db_iterator error:' + e)
        return

    categories_index = {}
    super_categories = {}
    index = 0
    for cat in categories:
        categories_index[cat[2]] = index
        super_categories[cat[2]] = cat[0]
        index += 1

    if not os.path.isdir(data_path):
        raise NotADirectoryError('db_iterator error: given parameter [' + data_path + ']is not a path.')

    try:
        for i in values:
            labels = np.zeros(len(categories))

            for label in annotations:
                if label[3] == i[0]:
                    labels[categories_index[label[2]]] = 1
                    labels[categories_index[super_categories[label[2]]]] = 1

            image = Image(
                i[0],
                os.path.abspath(os.path.join(data_path, i[1])),
                i[2],
                i[3],
                labels
                )

            yield image
    except Exception as e:
        raise e
    return

if __name__ == "__main__":
    db_name = "../val_info.db"
    
