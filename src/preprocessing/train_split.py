# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 23:16:14 2019

@author: Hilal
"""

import sqlite3
import numpy as np



 db_connection = sqlite3.connect(db_name)
 cursor = db_connection.cursor()
 
    values = None
    categories = None
    annotations = None