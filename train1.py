import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# import utils
import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Input, Dropout,Flatten, Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
from IPython.display import SVG, Image
# from livelossplot import PlotLossesKeras
from livelossplot import PlotLossesKerasTF

import tensorflow as tf
print("Tensorflow version:", tf.__version__)

Data_dir_train = "H:/LEARNNNNNN/BACHELOR PROJECt/Dataset/2/train/"
Data_dir_test = "H:/LEARNNNNNN/BACHELOR PROJECt/Dataset/2/test/"
Data_dir_save = "H:/LEARNNNNNN/BACHELOR PROJECt/Dataset/2/"


Classes  = ['angry','disgust','fear','happy','neutral','sad','surprise']

# for expression in os.listdir(Data_dir_train):
# print(str(len(os.listdir(Data_dir_train+ expression)))+ " " + expression + " images")
#############################################################################
img_size = 48
batch_size = 64
datagen_train = ImageDataGenerator(horizontal_flip=True)
train_generator = datagen_train.flow_from_directory(Data_dir_train,
                                                    target_size=(img_size, img_size),
                                                    color_mode='grayscale',
                                                    batch_size=batch_size,
                                                    class_mode='categorical', shuffle=True)

datagen_validation = ImageDataGenerator(horizontal_flip=True)
validation_generator = datagen_validation.flow_from_directory(Data_dir_test,
                                                              target_size=(img_size, img_size),
                                                              color_mode='grayscale',
                                                              batch_size=batch_size, class_mode='categorical',
                                                              shuffle=True)


model= Sequential()
#1 Conv
model.add(Conv2D(64,(3,3),padding='same',input_shape=(48,48,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
# 2 conv
model.add(Conv2D(128,(5,5), padding = 'same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
# 3 conv
model.add(Conv2D(512,(3,3), padding = 'same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
# 4 conv
model.add(Conv2D(512,(3,3), padding = 'same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(7,activation='softmax'))
opt=Adam(lr=0.0005)
model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
epochs = 15

steps_per_epoch = train_generator.n//train_generator.batch_size
validation_steps= validation_generator.n//validation_generator.batch_size
checkpoint= ModelCheckpoint(Data_dir_save+"model_weights.h5",monitor='val_accuracy',
                           save_weights_only= True,
                mode='max',verbose=1)

reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=2,min_lr=0.00001,model='auto')

callbacks=[PlotLossesKerasTF(),checkpoint, reduce_lr]
history = model.fit(
          x=train_generator,
          steps_per_epoch=steps_per_epoch,
          epochs =epochs,
          validation_data=validation_generator,
          validation_steps=validation_steps,
          callbacks = callbacks
)
model.save(Data_dir_save+"model.h5")
