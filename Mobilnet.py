import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense,Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.applications import imagenet_utils
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
import matplotlib.pyplot as plt

physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs available:", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)

mobile = tf.keras.applications.mobilenet.MobileNet()

def prepare_image(file):
    img_path= 'data/MobilNet-samples/'
    img = image.load_img(img_path+file,target_size=(224,224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array,axis=0)
    return tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

from IPython.display import Image

Image(filename='data/MobilNet-samples/1.PNG',width=300,height=200)

preprocessed_image=prepare_image('1.PNG')
predictions= mobile.predict (preprocessed_image)
results = imagenet_utils.decode_predictions(predictions) #return the top 5 predictions that mobilenet have
print (results) #40% frilled lizard preditcion 25% green lizard

Image(filename='data/MobilNet-samples/2.PNG',width=300,height=300)

preprocessed_image=prepare_image('2.PNG')
predictions= mobile.predict (preprocessed_image)
results = imagenet_utils.decode_predictions(predictions) #return the top 5 predictions that mobilenet have
print (results) #88% frilled lizard preditcion 9% taza

#Fine tune MobileNet

#Downloaded https://github.com/ardamavi/Sign-Language-Digits-Dataset with images an saved  into data/Sign-Language
#organizer folders and dataset:

os.chdir('data/Sign-Language')
if os.path.isdir('train/0/') is False:
    os.mkdir('train')
    os.mkdir('test')
    os.mkdir('valid')

    for i in range(0,10):
        shutil.move(f'{i}','train')
        os.mkdir(f'valid/{i}')
        os.mkdir(f'test/{i}')

        valid_samples = random.sample (os.listdir(f'train/{i}'),30)
        for j in valid_samples:
            shutil.move(f'train/{i}/{j}',f'valid/{i}')

        test_samples = random.sample(os.listdir(f'train/{i}'), 5)
        for j in test_samples:
            shutil.move(f'train/{i}/{j}', f'test/{i}')

os.chdir('../..')

train_path='data/Sign-Language/train'
test_path = 'data/Sign-Language/test'
valid_path  = 'data/Sign-Language/valid'

train_batches = ImageDataGenerator(preprocessing_function=tf.keras.application.mobilenet.preprocess_input).flow_from_directory(directory=train_path,target_size=(244,244),batch_size=10)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.application.mobilenet.preprocess_input).flow_from_directory(directory=test_path,target_size=(244,244),batch_size=10)
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.application.mobilenet.preprocess_input).flow_from_directory(directory=valid_path,target_size=(244,244),batch_size=10,shuffle=False)


mobile = tf.keras.applications
