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


def make_model(learning_rate=0.01, size_inner=100, droprate=0.5):
    base_model = Xception(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
    base_model.trainable = False
    
    ##############################################
    
    inputs = keras.Input(shape=(150, 150, 3))
    base = base_model(inputs, training=False)
    vectors = keras.layers.GlobalAveragePooling2D()(base)
    
    inner = keras.layers.Dense(size_inner, activation='relu')(vectors)
    
    drop = keras.layers.Dropout(droprate)(inner)
    
    outputs = keras.layers.Dense(6)(drop)
    model = keras.Model(inputs, outputs)
    
    ##############################################
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    loss = keras.losses.CategoricalCrossentropy(from_logits=True)
    
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=['accuracy']
    )
    
    return model



train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_ds = train_gen.flow_from_directory('./Garbage classification/train/',target_size=(150, 150), batch_size=32)

val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

val_ds = val_gen.flow_from_directory('./Garbage classification/validation',target_size=(150, 150), batch_size=32, shuffle=False)




checkpoint = keras.callbacks.ModelCheckpoint(
    'xception_v2_{epoch:02d}_{val_accuracy:.3f}.h5',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max'
)




learning_rate = 0.001
size = 100
droprate = 0.2


model = make_model(
    learning_rate=learning_rate,
    size_inner=size,
    droprate=droprate
)


history = model.fit(train_ds, epochs=50, validation_data=val_ds, callbacks=[checkpoint])

