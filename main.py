# Autor: Guillermo
import numpy as np
from dataset.createDataset import createSet

train_labels = []
train_samples = []
samples_size=1000
split=0.05
scaled_train_samples,train_labels=createSet(train_samples,train_labels,samples_size,split)

#### create an ANN


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy

model = Sequential([
    Dense(units=16,input_shape=(1,),activation='relu'),
    Dense(units=32,activation='relu'),
    Dense(units=2,activation='softmax')  # dos clases, con sintomas o sin sintomas.
])
model.summary()  #shows the arquitecture of the model

#train the model

model.compile(optimizer=Adam(learning_rate=0.0001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
# model.fit(x=scaled_train_samples,y=train_labels,batch_size=10,epochs=30,shuffle=True, verbose=2)  #en lotes de diez en diez, treinta viajes, shuffel true, te manda mas ifo el verbose

#Result
# Epoch 28/30
# 210/210 - 0s - loss: 0.2619 - accuracy: 0.9333
# Epoch 29/30
# 210/210 - 0s - loss: 0.2603 - accuracy: 0.9314
# Epoch 30/30
# 210/210 - 0s - loss: 0.2585 - accuracy: 0.9333

#validation set

model.fit(x=scaled_train_samples,y=train_labels, validation_split=0.1,batch_size=10,epochs=30,shuffle=True, verbose=2)  #en lotes de diez en diez, treinta viajes, shuffel true, te manda mas ifo el verbose
#the validation date is removed from the training data samples
#only the training set is shuffled not the validation => the split goes first => take care if data is filtered or ordered

#Result
# Epoch 28/30
# 189/189 - 0s - loss: 0.2287 - accuracy: 0.9450 - val_loss: 0.2548 - val_accuracy: 0.9333
# Epoch 29/30
# 189/189 - 0s - loss: 0.2284 - accuracy: 0.9455 - val_loss: 0.2548 - val_accuracy: 0.9333
# Epoch 30/30
# 189/189 - 0s - loss: 0.2282 - accuracy: 0.9455 - val_loss: 0.2545 - val_accuracy: 0.9333

#Test dataset

test_samples=[]
test_labels= []
samples_size=200
split=0.05

scaled_test_samples,test_labels=createSet(test_samples, test_labels,samples_size,split)

#predict

precictions=model.predict(x=scaled_test_samples,batch_size=10,verbose=0)
print(precictions)
rounded_predictions=np.argmax(precictions,axis=-1) # te pilla de los porcentajes de prediciones el mas probable y redondea a 0 o 1

#confusion matrix => is my model doing ok

from sklearn.metrics import confusion_matrix
from dataset.graphx import plot_confusion_matrix
import matplotlib.pyplot as plt

cm = confusion_matrix(y_true=test_labels,y_pred=rounded_predictions)
cm_plot_labels=['no side effects','had side effects']
plot_confusion_matrix (cm,cm_plot_labels,title='Confusion matrix')
# plt.show()
import os.path

if os.path.isfile('models/medical_trial_model.h5') is False:
    model.save('models/medical_trial_model.h5')
    # save architecture of the mode
    # save the wiehgs
    # the trainging cofiguration
    # the state of the optimizer

from tensorflow.keras.models import load_model
new_model=load_model('models/medical_trial_model.h5')
#chek if it is the same
new_model.summary()
new_model.get_weights()