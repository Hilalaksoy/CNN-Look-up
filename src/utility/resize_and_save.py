# -*- coding: utf-8 -*-
import os
import numpy
from utility.data_generator import DataGenerator       

VAL_DATA_PATH = './data/val2017'
DB_NAME = './data/train_info.db'
TRAIN_DATA_PATH = './data/train2017'

IMAGE_SIZE = 128
BATCH_SIZE = 128
NUM_CLASSES = 80
TRAIN_BATCH_NO = 640
TEST_BATCH_NO = 274
VAL_BATCH_NO = 40

data_gen = DataGenerator(DB_NAME, TRAIN_DATA_PATH, table_name='TrainImages',image_size=IMAGE_SIZE, resizing_method='with_padding', batch=BATCH_SIZE)
SAVE_PATH = './data/resized_128_with_padding/train/'

index = 0
for (x,y) in data_gen.flow():
        numpy.save(SAVE_PATH + 'x' + str(index), x)
        numpy.save(SAVE_PATH + 'y' + str(index), y)
        print('Saving batch ', index)
        index += 1
        if index == TRAIN_BATCH_NO:
                break

data_gen = DataGenerator(DB_NAME, TRAIN_DATA_PATH, table_name='TestImages',image_size=IMAGE_SIZE, resizing_method='with_padding', batch=BATCH_SIZE)
SAVE_PATH = './data/resized_128_with_padding/test/'

index = 0
for (x,y) in data_gen.flow():
        numpy.save(SAVE_PATH + 'x' + str(index), x)
        numpy.save(SAVE_PATH + 'y' + str(index), y)
        print('Saving batch ', index)
        index += 1
        if index == TEST_BATCH_NO:
                break
        

data_gen = DataGenerator(DB_NAME, VAL_DATA_PATH, table_name='ValidationImages',image_size=IMAGE_SIZE, resizing_method='with_padding', batch=BATCH_SIZE)
SAVE_PATH = './data/resized_128_with_padding/val/'

index = 0
for (x,y) in data_gen.flow():
        numpy.save(SAVE_PATH + 'x' + str(index), x)
        numpy.save(SAVE_PATH + 'y' + str(index), y)
        print('Saving batch ', index)
        index += 1
        if index == VAL_BATCH_NO:
                break
