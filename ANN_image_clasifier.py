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
from dataset.graphx import plot_confusion_matrix

warnings.simplefilter(action='ignore', category=FutureWarning)

physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs available:", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)

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
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(244, 244, 3)),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Flatten(),
    Dense(units=2, activation='softmax')
])

model.summary()
model.compile(optimizer=Adam(learning_rate=0.0001),loss='categorical_crossentropy',metrics=['accuracy']) #binary cross entropy also possible => sigmoid activation.
model.fit(x=train_batches,validation_data=valid_batches,epochs=10,verbose=2) #data is stored as a generator so we dont have to specify y

#CNN predictions

test_imgs,test_labels=next(test_batches)
plotImages(test_imgs)
plt.show()
print(test_labels)

test_batches.classes

predictions1= model.predict(x=test_batches,verbose=0)
print(np.round(predictions1)) #each one of the arrays is one prediction=> prediction for the first sample =>

cm=confusion_matrix(y_true=test_batches.classes,y_pred=np.argmax(predictions1,axis=1))
print(test_batches.classes)
cm_plot_labels=['cat','dog']
plot_confusion_matrix(cm=cm,classes=cm_plot_labels,title='Confusion matrix cnn',cmap=plt.cm.Blues)
plt.show() #overfitting

#Build a fine tunned VGG16 model

vgg16model = tf.keras.applications.vgg16.VGG16()
vgg16model.summary()

model = Sequential()
for layer in vgg16model.layers[:-1]:  #we look in every layer o
    model.add(layer)

for layer in model.layers:
    layer.trainable=False  #iterar por las layers del nuevo model y vamos a setearlas como no entrenables => freeze the weights and biases from al the liars, not retrained => vgg16 already trained.

model.add(Dense(units=2,activation='softmax'))

model.summary()  #added last dense layer with two output classes => dog,cat, trainable parameters => dense layer, the rest not trainable


model.compile (optimizer=Adam(learning_rate=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(x=train_batches,validation_data=valid_batches,epochs=5,verbose=2)

#Predictions usin fine tuned

predictions2 = model.predict(x=test_batches,verbose=0)

print(np.round(predictions2)) #each one of the arrays is one prediction=> prediction for the first sample =>

cm=confusion_matrix(y_true=test_batches.classes,y_pred=np.argmax(predictions2,axis=1))
print(test_batches.classes)
cm_plot_labels=['cat','dog']
plot_confusion_matrix(cm=cm,classes=cm_plot_labels,title='Confusion matrix VGG16',cmap=plt.cm.Blues)
plt.show() #overfitting