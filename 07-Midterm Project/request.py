#!/usr/bin/env python
# coding: utf-8

import requests

url = 'http://localhost:9090/predict'

pizza_id = 'project-pizza'
pizza = {'restaurant': 'B',
 'extra_cheeze': 'no',
 'extra_mushroom': 'yes',
 'size_by_inch': 15,
 'extra_spicy': 'yes'}

response = requests.post(url, json=pizza).json()
print('Pizza predicted Price : %f' % response['price'])

