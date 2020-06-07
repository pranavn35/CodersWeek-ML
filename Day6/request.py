# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 12:52:54 2020

@author: ADMIN
"""

import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'Weight':420.309, 'Length1':25.4, 'Length2':24.9,'Length3':30.4,'Height':4.9,'Width':3.08})

print(r.json())