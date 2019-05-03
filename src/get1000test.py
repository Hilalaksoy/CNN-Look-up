# -*- coding: utf-8 -*-


from preprocessing.read_data import db_iterator

import json
import numpy as np

DATA_PATH = './data/val2017'
DB_NAME = './data/train_info.db'

images = []

for x in db_iterator(DB_NAME, DATA_PATH, 'ValidationImages', read_images=False):
	images.append(x)


with open('vals.json', 'w', encoding='utf8') as fi:
	for x in images:
		wr = {'image': x.full_path.split('\\')[-1], 'labels':[int(x) for x in np.where(x.labels == 1)[0].astype(np.int32) ]}
		fi.write(json.dumps(wr) + '\n')