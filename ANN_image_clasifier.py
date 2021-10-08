import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, Bidirectional, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
import glob
import matplotlib.pyplot as plt
import warnings
from dataset.graphx import plotImages

warnings.simplefilter(action='ignore', category=FutureWarning)

# organize data into train, valid and test.
os.chdir('dogs-vs-cats')
if os.path.isdir('train/dog') is False:
    os.makedirs('train/dog')
    os.makedirs('train/cat')
    os.makedirs('valid/dog')
    os.makedirs('valid/cat')
    os.makedirs('test/dog')
    os.makedirs('test/cat')

    for c in random.sample(glob.glob('cat*'), 500):
        shutil.move(c, 'train/cat')
    for c in random.sample(glob.glob('dog*'), 500):
        shutil.move(c, 'train/dog')
    for c in random.sample(glob.glob('cat*'), 100):
        shutil.move(c, 'valid/cat')
    for c in random.sample(glob.glob('dog*'), 100):
        shutil.move(c, 'valid/dog')
    for c in random.sample(glob.glob('cat*'), 50):
        shutil.move(c, 'test/cat')
    for c in random.sample(glob.glob('dog*'), 50):
        shutil.move(c, 'test/dog')

os.chdir('../')
train_path = 'dogs-vs-cats/train'
test_path = 'dogs-vs-cats/test'
valid_path = 'dogs-vs-cats/valid'

# put the data into the right format => keras generator vgg16

train_batches = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=train_path,
                                                                                             target_size=(244, 244),
                                                                                             classes=['cat', 'dog'],
                                                                                             batch_size=10)
valid_batches = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=valid_path,
                                                                                             target_size=(244, 244),
                                                                                             classes=['cat', 'dog'],
                                                                                             batch_size=10)
test_batches = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=test_path,
                                                                                             target_size=(244, 244),
                                                                                             classes=['cat', 'dog'],
                                                                                             batch_size=10,
                                                                                             shuffle=False)

assert train_batches.n == 1000
assert valid_batches.n == 200
assert test_batches.n == 100

imgs, labels = next(train_batches)

plotImages(imgs)
print(labels)
plt.show()


# Build and train CNN image data kernelsize 3 by 3, maxpool cut image dimension in half.

model = Sequential([
    Conv2D(filters=32,kernel_size=(3,3),activation='relu',padding='same', input_shape=(244,244,3)),
    MaxPool2D(pool_size=(2,2),strides=2),
    Conv2D(filters=64,kernel_size=(3,3),activation='relu',padding='same'),
    MaxPool2D(pool_size=(2,2),strides=2),
    Flatten(),
    Dense(units=2,activation='softmax')
])