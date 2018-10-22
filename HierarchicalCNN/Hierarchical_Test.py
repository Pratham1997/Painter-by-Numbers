import os
import numpy as np
import cv2
import random
import pandas as pd
from keras import optimizers, initializers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import h5py
path = 'Dataset/Train1'
pathTest = 'TestingData'
files = os.listdir(path)
filesTest=os.listdir(pathTest)
art_count = 10
label=pd.read_csv('train_info.csv')
artistnum=label.artist.unique()
values={}
for i in range(len(artistnum)):
    values[artistnum[i]]=i
art={}
for i in range(len(label)):
    art[label.loc[i,'filename']]=label.loc[i,'artist']
hgt = 20
wdt = 20
images  = np.zeros((len(files),hgt,wdt,3))
s=0;
images_sep = []
img_order = {}
for i in range(10):
        images_sep.append([])
#Loading the training data since the dictionary for artists is based on the training data order.
artists = {}
count = 0
for i in files:
    images[s,:,:,:]=cv2.resize(cv2.imread(path + '/' + i),(wdt,hgt))
    if not art[i[2:]] in artists:
        artists[art[i[2:]]] = count
        count = count + 1
    images_sep[artists[art[i[2:]]]].append(s)
    s+=1
for i in range(10):
        img_order[min(images_sep[i])] = i
files=filesTest
imagesTest  = np.zeros((len(files),hgt,wdt,3))
images_sep_test = []
for i in range(10):
        images_sep_test.append([])
s=0;
artists_test = {}
artists_label = []
for i in files:
    imagesTest[s,:,:,:]=cv2.resize(cv2.imread(pathTest + '/' + i),(wdt,hgt))
    artists_label.append(artists[art[i]])
    images_sep_test[artists[art[i]]].append(s)
    s+=1;
images = images.astype('float32')
imagesTest = imagesTest.astype('float32')
#Standardizing the images.
imagesTest = (imagesTest - np.mean(images, 0)) / np.std(images, 0)
images = (images - np.mean(images, 0)) / np.std(images, 0)
#Generating the models and loading the weights.
main_model = Sequential()
main_model.add(Convolution2D(16, (3, 3),padding='same', activation = 'relu', input_shape = (hgt, wdt, 3)))
main_model.add(Convolution2D(32, (5, 5), activation = 'relu',padding='same'))
main_model.add(MaxPooling2D(2, 2))
main_model.add(Flatten())
main_model.add(Dense(128, activation = 'sigmoid'))
main_model.add(Dense(5, activation = 'softmax'))
main_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
main_model.load_weights('weights.hdf5')
model_num = main_model.predict(imagesTest)
smaller_models = []
for i in range(0, 5):
        smaller_models.append(Sequential())
        smaller_models[i].add(Convolution2D(16, (3, 3),padding='same', activation = 'relu', input_shape = (hgt, wdt, 3)))
        smaller_models[i].add(Convolution2D(32, (5, 5), activation = 'relu',padding='same'))
        smaller_models[i].add(MaxPooling2D(2, 2))
        smaller_models[i].add(Flatten())
        smaller_models[i].add(Dense(128, activation = 'sigmoid'))
        smaller_models[i].add(Dense(2, activation = 'softmax'))
        smaller_models[i].compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        smaller_models[i].load_weights('weights'+str(i)+'.hdf5')
pred_artists = []
correct = 0
for i in range(0, imagesTest.shape[0]):
        scrs = np.zeros((10))
        for j in range(0, 5):
                #Multiplying the scores from the two-level models.
                x = smaller_models[j].predict(np.expand_dims(imagesTest[i, :, :, :], 0))
                scrs[2*j] = x[0, 0]
                scrs[2*j+1] = x[0, 1]
                scrs[2*j] *= model_num[i, j]
                scrs[2*j+1] *= model_num[i, j]
        #Taking argmax for predicting model label.
        pred_artists.append(np.argmax(scrs))
        if pred_artists[i] == artists_label[i]:
                correct += 1
#Printing the accuracy.
print "Accuracy: "+str(correct*100.0/imagesTest.shape[0])+"%"
