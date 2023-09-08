from keras.preprocessing.image import ImageDataGenerator
import keras
from keras.applications import vgg16
from keras.models import load_model
import numpy as np

items = {0: 'apple', 1: 'banana', 2: 'beetroot', 3: 'bell pepper', 4: 'cabbage', 5: 'capsicum', 6: 'carrot', 7: 'cauliflower', 8: 'chilli pepper', 9: 'corn', 10: 'cucumber', 11: 'eggplant', 12: 'garlic', 13: 'ginger', 14: 'grapes', 15: 'jalepeno', 16: 'kiwi', 17: 'lemon', 18: 'lettuce', 19: 'mango', 20: 'onion', 21: 'orange', 22: 'paprika', 23: 'pear', 24: 'peas', 25: 'pineapple', 26: 'pomegranate', 27: 'potato', 28: 'raddish', 29: 'soy beans', 30: 'spinach', 31: 'sweetcorn', 32: 'sweetpotato', 33: 'tomato', 34: 'turnip', 35: 'watermelon'}

test_gen = ImageDataGenerator(
    preprocessing_function = vgg16.preprocess_input
)

model = load_model('Fruits_Vegetables.h5')

image_path = 'predict'

img = test_gen.flow_from_directory(directory = image_path, target_size=(224, 224))

prediction = model.predict(img)

print(prediction)
print(items[np.argmax(prediction)])