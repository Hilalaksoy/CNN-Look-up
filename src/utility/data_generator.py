from preprocessing.read_data import *
from preprocessing.resizing import *
import numpy as np

class DataGenerator(object):
    """
    DataGenerator, keras modellerinin kullanabilecegi sekilde tasarlanan bir data iteratoru
    resizing method: 
        ('area', resize_area)
    """
    def __init__(self, db_name, data_path,table_name,image_size=64, resizing_method='area', batch=128, grayscale=False):
        super(DataGenerator, self).__init__()
        self.batch = batch
        self.db_name = db_name
        self.data_path = data_path
        self.table_name = table_name
        self.image_size = image_size
        self.resizing_method = resizing_method
        self.grayscale = grayscale
    
    def flow(self):
        while True:
            for chunk in get_chunks(db_iterator(self.db_name, self.data_path,self.table_name, grayscale=self.grayscale), self.batch):
                x = []
                y = []
                for resim in chunk:
                    if self.resizing_method == 'area':
                        x.append(resize_area(resim.get_pixel_matrix(), self.image_size))
                    elif self.resizing_method == 'with_padding':
                        x.append(resize_with_padding(resim.get_pixel_matrix(), self.image_size))
                    elif self.resizing_method == 'no_resizing':
                        x.append(resim.get_pixel_matrix())
                    else:
                        print('Hata DataGenerator: resizing_method ("area", "with_padding") degerlerinden birisi olmalidir.')
                    y.append(resim.labels)
                x = np.array(x)
                y = np.array(y)            
                yield (x, y)
        return
    
    
        
