import numpy as np
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

def createSet(sample, label,size,split):

    split_sample=int(size*split)
    for i in range(split_sample):  # the 5% of the younguer with sidefects and the 5% of the older with no sidefects
        random_younguer = randint(13,
                                  64)  # drug tested on indivuduals from 13 to 100 years old this are the younger guys
        sample.append(random_younguer)
        label.append(1)
        random_older = randint(65, 100)  # this are the older ones
        sample.append(random_older)
        label.append(0)

    for i in range(size):  # the rest
        random_younguer = randint(13,
                                  64)  # drug tested on indivuduals from 13 to 100 years old this are the younger guys
        sample.append(random_younguer)
        label.append(0)
        random_older = randint(65, 100)  # this are the older ones
        sample.append(random_older)
        label.append(1)

    label=np.array(label)
    sample=np.array(sample)

    label,sample =shuffle(label,sample)
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_samples= scaler.fit_transform(sample.reshape(-1,1)) #fit transform does not accept 1D data so we reshape the scaled train samples to be 2D

    # print((scaled_samples))
    return scaled_samples,label