
# coding: utf-8


import os
import numpy as np
import cv2
import pandas as pd
from keras import optimizers, initializers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import h5py
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
get_ipython().run_line_magic('matplotlib', 'inline')


#Read training and validation data
path = 'Dataset/Train1'
pathTest = 'Dataset/Validation1'
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
    
print ("Images Read")

#Read validation images
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



images = images.reshape(images.shape[0], 20, 20, 3)
imagesTest = imagesTest.reshape(imagesTest.shape[0], 20, 20, 3)

images = images.astype('float32')
images=images/255;
imagesTest = imagesTest.astype('float32')
imagesTest=imagesTest/255;

#Create model for cnn
model = Sequential()

model.add(Convolution2D(16, (3,3),padding='same', activation = 'relu', input_shape = (20, 20, 3)))
model.add(Convolution2D(32, (3, 3), activation = 'relu',padding='same'))
model.add(MaxPooling2D(2, 2))
model.add(Flatten())
model.add(Dense(128, activation = 'sigmoid'))
model.add(Dense(10))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

#Train the cnn on input and validation data
checkpointer = ModelCheckpoint(filepath="weightNormal10.hdf5", verbose=1, save_best_only=True)
model.fit(images, output, batch_size=800,validation_data=(imagesTest,outputTest), epochs=80,callbacks = [checkpointer])



