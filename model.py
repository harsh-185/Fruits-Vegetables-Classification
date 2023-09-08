import os
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras.layers import GlobalAveragePooling2D, Dense
from keras.models import Model
from keras.applications import ResNet101
import tensorflow as tf
import pandas as pd 
import numpy as np

data_gen = ImageDataGenerator(
    preprocessing_function = keras.applications.vgg16.preprocess_input,
    rotation_range=30,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
)
train_images = data_gen.flow_from_directory(
    directory= "train",
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=True,
    seed=0,)

valid_images = data_gen.flow_from_directory(
    directory= "validation",
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=True,
    seed=0,)

test_gen = ImageDataGenerator(
    preprocessing_function = keras.applications.vgg16.preprocess_input
)

test_images = data_gen.flow_from_directory(
    directory= "test",
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=False,
    seed=0,)

base_model = ResNet101(
    include_top=False,
    input_shape=(224,224,3),
    pooling="avg"
)

base_model.trainable = False

inputs = base_model.input

x = Dense(128, activation='relu')(base_model.output) 
x = Dense(256, activation='relu')(x) 
outputs = Dense(36, activation='softmax')(x)
model = Model(inputs=inputs, outputs=outputs)

model.compile(                                           
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(                    
    train_images,
    validation_data=valid_images,
    batch_size = 32,
    epochs=8)

loss, accuracy = model.evaluate(test_images)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

model.save("Fruits_Vegetables.h5")

#93 accuracy
