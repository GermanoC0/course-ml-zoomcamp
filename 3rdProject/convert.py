#!/usr/bin/env python
# coding: utf-8

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.applications.xception import decode_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tflite_runtime.interpreter as tflite
from keras_image_helper import create_preprocessor



model = keras.models.load_model('./xception_v2_06_0.821.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('garbage-classification-model.tflite', 'wb') as f_out:
    f_out.write(tflite_model)
    
