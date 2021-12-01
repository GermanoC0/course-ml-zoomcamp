#!/usr/bin/env python
# coding: utf-8

import tflite_runtime.interpreter as tflite
import numpy as np

from io import BytesIO
from urllib import request
from PIL import Image

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img
    
def preprocess_input(x):
    x /= 255
    x -= 0.2
    return x



classes = [
    'dogs',
    'cats'
]

interpreter = tflite.Interpreter(model_path='../dogs_cats_10_0.687.tflite')
interpreter.allocate_tensors() # Weights

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

#url = 'wget https://upload.wikimedia.org/wikipedia/commons/9/9a/Pug_600.jpg -O Pug_600.jpg'

def predict(url):
	img = download_image(url)
	image = prepare_image(img, (150, 150))
	x = np.array(image, dtype='float32')
	X = np.array([x])
	X = preprocess_input(X)
	
	interpreter.set_tensor(input_index, X)
	interpreter.invoke()
	preds = interpreter.get_tensor(output_index)
	
	float_predictions = preds[0].tolist()

	return dict(zip(classes, float_predictions))


def lambda_handler(event, context):
	url = event['url']
	result = predict(url)
	
	return result

