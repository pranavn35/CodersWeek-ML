# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 12:52:54 2020

@author: ADMIN
"""

import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'Age':27, 'Estimated Salary':33000})

print(r.json())