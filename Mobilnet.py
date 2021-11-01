import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation
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
    img_path = 'data/MobilNet-samples/'
    img = image.load_img(img_path + file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)


from IPython.display import Image

Image(filename='data/MobilNet-samples/1.PNG', width=300, height=200)

preprocessed_image = prepare_image('1.PNG')
predictions = mobile.predict(preprocessed_image)
results = imagenet_utils.decode_predictions(predictions)  # return the top 5 predictions that mobilenet have
print(results)  # 40% frilled lizard preditcion 25% green lizard

Image(filename='data/MobilNet-samples/2.PNG', width=300, height=300)

preprocessed_image = prepare_image('2.PNG')
predictions = mobile.predict(preprocessed_image)
results = imagenet_utils.decode_predictions(predictions)  # return the top 5 predictions that mobilenet have
print(results)  # 88% frilled lizard preditcion 9% taza

# Fine tune MobileNet

# Downloaded https://github.com/ardamavi/Sign-Language-Digits-Dataset with images an saved  into data/Sign-Language
# organizer folders and dataset:

os.chdir('data/Sign-Language')
if os.path.isdir('train/0/') is False:
    os.mkdir('train')
    os.mkdir('test')
    os.mkdir('valid')

    for i in range(0, 10):
        shutil.move(f'{i}', 'train')
        os.mkdir(f'valid/{i}')
        os.mkdir(f'test/{i}')

        valid_samples = random.sample(os.listdir(f'train/{i}'), 30)
        for j in valid_samples:
            shutil.move(f'train/{i}/{j}', f'valid/{i}')

        test_samples = random.sample(os.listdir(f'train/{i}'), 5)
        for j in test_samples:
            shutil.move(f'train/{i}/{j}', f'test/{i}')

os.chdir('../..')

train_path = 'data/Sign-Language/train'
test_path = 'data/Sign-Language/test'
valid_path = 'data/Sign-Language/valid'

train_batches = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(directory=train_path,
                                                                                                 target_size=(244, 244),
                                                                                                 batch_size=10)
test_batches = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(directory=test_path,
                                                                                                 target_size=(244, 244),
                                                                                                 batch_size=10)
valid_batches = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(directory=valid_path,
                                                                                                 target_size=(244, 244),
                                                                                                 batch_size=10,
                                                                                                 shuffle=False)

mobile.summary()

#  voy a quitar layers:
# global_average_pooling2d (Gl (None, 1024)              0
# _________________________________________________________________
# reshape_1 (Reshape)          (None, 1, 1, 1024)        0
# _________________________________________________________________
# dropout (Dropout)            (None, 1, 1, 1024)        0
# _________________________________________________________________
# conv_preds (Conv2D)          (None, 1, 1, 1000)        1025000
# _________________________________________________________________
# reshape_2 (Reshape)          (None, 1000)              0
# _________________________________________________________________
# predictions (Activation)     (None, 1000)              0
# =================================================================
# Total params: 4,253,864
# Trainable params: 4,231,976
# Non-trainable params: 21,888

x = mobile.layers[-6].output
output = Dense(units=10, activation='softmax')(x)

model = Model(inputs=mobile.input,
              outputs=output)  # we are adding to the model model all the layers from mobile but the last six and the ouptus layer DEnse

for layer in model.layers[:-23]:
    layer.trainable = False  # only the last 23 layers to be trainable. 88 layers on the original mobile training label.

model.summary()

# conv_dw_13_relu (ReLU)       (None, 7, 7, 1024)        0
# _________________________________________________________________
# conv_pw_13 (Conv2D)          (None, 7, 7, 1024)        1048576
# _________________________________________________________________
# conv_pw_13_bn (BatchNormaliz (None, 7, 7, 1024)        4096
# _________________________________________________________________
# conv_pw_13_relu (ReLU)       (None, 7, 7, 1024)        0
# _________________________________________________________________
# global_average_pooling2d (Gl (None, 1024)              0
# _________________________________________________________________
# dense (Dense)                (None, 10)                10250
# =================================================================
# Total params: 3,239,114
# Trainable params: 1,873,930
# Non-trainable params: 1,365,184
# ________________________________

model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy',metrics=['accuracy'])
# Run for more epoch to see better results

model.fit(x= train_batches, validation_data=valid_batches,epochs=10,verbose=2)

# Epoch 10/10
# 172/172 - 6s - loss: 0.0063 - accuracy: 0.9994 - val_loss: 0.0413 - val_accuracy: 0.9867

#100% accuracy on our training set and 98 on our validation set
# a little bit overfitted  validation accuracy is lower than our training accuracy.
# val los and val accuracy don't show a dow and up tendency => more epochs?


#Precit sign language digits

test_labels=test_batches.classes
predictions= model.predict(x=test_batches,verbose=0)

cm= confusion_matrix(y_true=test_labels,y_pred=predictions.argmax(axis=1))
test_batches.class_indices
cm_plot_labels = ['0','1','2','3','4','5','6','7','8','9']
from dataset.graphx import plot_confusion_matrix

plot_confusion_matrix(cm=cm,classes=cm_plot_labels,title='MobileNet confusion Matrix')
plt.show()