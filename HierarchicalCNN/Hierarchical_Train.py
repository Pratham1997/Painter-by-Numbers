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
pathTest = 'Dataset/Validation1'
files = os.listdir(path)
filesTest=os.listdir(pathTest)
#Number of artists to be selected.
art_count = 10
label=pd.read_csv('train_info.csv')
#Finding the number of unique artists in the csv file.
artistnum=label.artist.unique()
values={}
for i in range(len(artistnum)):
    values[artistnum[i]]=i
#Creating a dictionary with keys as the filenames and values as the artist names.
art={}
for i in range(len(label)):
    art[label.loc[i,'filename']]=label.loc[i,'artist']
#Size of input
hgt = 20
wdt = 20
images  = np.zeros((len(files),hgt,wdt,3))
s=0;
images_sep = []
for i in range(art_count):
        images_sep.append([])
artists = {}
count = 0
for i in files:
    #Reading the images.
    images[s,:,:,:]=cv2.resize(cv2.imread(path + '/' + i),(wdt,hgt))
    #Assigning artist number in range 0 - 9.
    if not art[i[2:]] in artists:
        artists[art[i[2:]]] = count
        count = count + 1
    #Segregating based on artists and storing the index.
    images_sep[artists[art[i[2:]]]].append(s)
    s+=1
#Loading the validation data.
files=filesTest
imagesTest  = np.zeros((len(files),hgt,wdt,3))
images_sep_test = []
for i in range(art_count):
        images_sep_test.append([])
s=0;
artists_test = {}
for i in files:
    imagesTest[s,:,:,:]=cv2.resize(cv2.imread(pathTest + '/' + i),(wdt,hgt))
    images_sep_test[artists[art[i]]].append(s)
    s+=1;
#Standardizing the images.
images = images.astype('float32')
imagesTest = imagesTest.astype('float32')
imagesTest = (imagesTest - np.mean(images, 0)) / np.std(images, 0)
images = (images - np.mean(images, 0)) / np.std(images, 0)
#Generating the label vectors for the main model training images.
labels_main = np.zeros((images.shape[0], 5))
labels_main[images_sep[0]+images_sep[1], 0] = 1
labels_main[images_sep[2]+images_sep[3], 1] = 1
labels_main[images_sep[4]+images_sep[5], 2] = 1
labels_main[images_sep[6]+images_sep[7], 3] = 1
labels_main[images_sep[8]+images_sep[9], 4] = 1
#Generating the label vectors for the main model validation images.
labels_main_test = np.zeros((imagesTest.shape[0], 5))
labels_main_test[images_sep_test[0]+images_sep_test[1], 0] = 1
labels_main_test[images_sep_test[2]+images_sep_test[3], 1] = 1
labels_main_test[images_sep_test[4]+images_sep_test[5], 2] = 1
labels_main_test[images_sep_test[6]+images_sep_test[7], 3] = 1
labels_main_test[images_sep_test[8]+images_sep_test[9], 4] = 1
#Generating the smaller models (for 2 class classification)
smaller_models = []
for i in range(0, 5):
        #Defining the models.
        smaller_models.append(Sequential())
        smaller_models[i].add(Convolution2D(16, (3, 3),padding='same', activation = 'relu', input_shape = (hgt, wdt, 3)))
        smaller_models[i].add(Convolution2D(32, (5, 5), activation = 'relu',padding='same'))
        smaller_models[i].add(MaxPooling2D(2, 2))
        smaller_models[i].add(Flatten())
        smaller_models[i].add(Dense(128, activation = 'sigmoid'))
        smaller_models[i].add(Dense(2, activation = 'softmax'))
        smaller_models[i].compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        checkpointer = ModelCheckpoint(filepath="weights"+str(i)+".hdf5", verbose=1, save_best_only=True)
        #Generating label vectors for 2 class classifier training images.
        lsk = images_sep[2*i+0] + images_sep[2*i+1]
        img = np.zeros((len(lsk), hgt, wdt, 3))
        labels1 = np.zeros((len(lsk), 2))
        count = 0
        for j in range(0, len(images_sep[2*i+0])):
                img[count, :, :, :] = images[images_sep[2*i+0][j], :, :, :]
                labels1[count, 0] = 1
                count += 1
        for j in range(0, len(images_sep[2*i+1])):
                img[count, :, :, :] = images[images_sep[2*i+1][j], :, :, :]
                labels1[count, 1] = 1
                count += 1
        #Generating label vectors for 2 class classifier validation images.
        lsk1 = images_sep_test[2*i+0] + images_sep_test[2*i+1]
        img_test = np.zeros((len(lsk1), hgt, wdt, 3))
        labels2 = np.zeros((len(lsk1), 2))
        count = 0
        for j in range(0, len(images_sep_test[2*i+0])):
                img_test[count, :, :, :] = imagesTest[images_sep_test[2*i+0][j], :, :, :]
                labels2[count, 0] = 1
                count += 1
        for j in range(0, len(images_sep_test[2*i+1])):
                img_test[count, :, :, :] = imagesTest[images_sep_test[2*i+1][j], :, :, :]
                labels2[count, 1] = 1
                count += 1
        smaller_models[i].fit(img, labels1, shuffle = True, batch_size=1000,validation_data = (img_test, labels2), epochs=60, callbacks = [checkpointer])
#Defining the main model.
main_model = Sequential()
main_model.add(Convolution2D(16, (3, 3),padding='same', activation = 'relu', input_shape = (hgt, wdt, 3)))
main_model.add(Convolution2D(32, (5, 5), activation = 'relu',padding='same'))
main_model.add(MaxPooling2D(2, 2))
main_model.add(Flatten())
main_model.add(Dense(128, activation = 'sigmoid'))
main_model.add(Dense(5, activation = 'softmax'))
main_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath="weights.hdf5", verbose=1, save_best_only=True)
main_model.fit(images, labels_main, batch_size=100,validation_data = (imagesTest, labels_main_test), epochs=60, callbacks = [checkpointer])
