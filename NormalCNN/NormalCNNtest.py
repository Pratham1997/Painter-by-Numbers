

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
from keras.models import load_model
import h5py


#Read training and validation data
path = 'Dataset/Train1'
pathTest = 'TestingData'
files = os.listdir(path)
filesTest=os.listdir(pathTest)

label=pd.read_csv('train_info.csv')

artistnum=label.artist.unique()
values={}

for i in range(len(artistnum)):
    values[artistnum[i]]=i


art={}
for i in range(len(label)):
    art[label.loc[i,'filename']]=label.loc[i,'artist']


#Read input images

images  = np.zeros((len(files),20,20,3))
s=0;
artists = {}
output=np.zeros((len(files),10))
count = 0
for i in files:
    images[s,:,:,:]=cv2.imread(path + '/' + i)
    
    if not art[i[1:]] in artists:
        artists[art[i[1:]]] = count
        count = count + 1
    output[s,artists[art[i[1:]]]]=1
    s+=1;

#Read test images
files=filesTest
imagesTest  = np.zeros((len(files),20,20,3))

s=0;
outputTest=np.zeros((len(files),10))
count = 0
for i in files:
    if i[0]=='.':
        continue;
    imagesTest[s,:,:,:]=cv2.imread(pathTest + '/' + i)
    outputTest[s,artists[art[i]]]=1
    s+=1;

print ("ImagesTest Read")


imagesTest = imagesTest.astype('float32')
imagesTest = imagesTest/255

correct=0

model = load_model('NormalCNNweights.h5')

predicted=model.predict(imagesTest)
for i in range(len(imagesTest)):
    if np.argmax(predicted[i,:])==np.argmax(np.array(outputTest)[i,:]):
        correct+=1

print "Accuracy is :"
print correct*100.0/len(imagesTest)


