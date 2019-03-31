import numpy as np
import cv2

class Image(object):
    """Image object holds some attr's about every image"""
    def __init__(self, _id, full_path, height_value, width_value, labels_value, read_image=True ):
        self.full_path = full_path
        self.height = height_value
        self.width = width_value
        self.labels = labels_value
        self.id = _id

        if read_image:
            try:
                self.matrix = cv2.imread(self.full_path)
            except IOException as e:
                raise e
        else:
            self._matrix = None

    def get_pixel_matrix(self):
        if self.matrix is None:
            try:
                self.matrix = cv2.imread(self.full_path)
            except IOException as e:
                raise e
        return self.matrix
