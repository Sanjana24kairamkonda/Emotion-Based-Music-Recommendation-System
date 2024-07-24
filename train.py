from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

import pandas as pd
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array
from keras.layers import Dense, Flatten, Dropout,BatchNormalization ,Activation,Conv2D,MaxPool2D
from keras.models import Sequential
from keras import layers
import matplotlib.pyplot as plt
train_directory='C://Users//lenovo//Downloads//Emotion-Music-Recommendation-1//Emotion-Music-Recommendation//train.py'
test_directory='C://Users//lenovo//Downloads//Emotion-Music-Recommendation-1//Emotion-Music-Recommendation//train.py'
train_datagen = ImageDataGenerator(width_shift_range = 0.2,height_shift_range = 0.2,rescale = 1./255,validation_split = 0.2)
validation_datagen = ImageDataGenerator(rescale = 1./255,validation_split = 0.2)
train_generator = train_datagen.flow_from_directory(train_directory, target_size=(224,224), color_mode='rgb', batch_size=64, class_mode='categorical', subset='training',shuffle=True,seed=50)
test_generator = validation_datagen.flow_from_directory(test_directory, target_size=(224,224), color_mode='rgb', batch_size=64, class_mode='categorical', subset='validation',shuffle=True,seed=50)
pretrained_model= keras.applications.EfficientNetB7(include_top=False,input_shape=(224,224,3),pooling='avg',classes=7,weights='imagenet')
for layer in pretrained_model.layers:
        layer.trainable=False




face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
ds_factor=0.6

emotion_model = Sequential()
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))
emotion_model.load_weights('model.h5')

opt = keras.optimizers.SGD(learning_rate = 0.01,nesterov=True)
emotion_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.75,patience=5, min_lr=0.001)
m=emotion_model.fit(train_generator,validation_data=test_generator,epochs = 100,callbacks=[reduce_lr])
plt.plot(m.history['loss'], label='train loss')
plt.plot(m.history['val_loss'],label='val_loss')
plt.legend()
plt.show()

plt.plot(m.history['accuracy'], label='train accuracy')
plt.plot(m.history['val_accuracy'], label='val accuracy')
plt.legend()
plt.show()

emotion_model.save_weights('model.h5')


