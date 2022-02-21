import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.helpers import Helpers

class CNN(Helpers):
    def __init__(self, *args, **kwargs):
        super(CNN, self).__init__(*args, **kwargs)
        self.path = '/home/lizano/Documents/SAC/data/raw/cnn'
        self.k = len(os.listdir(os.path.join(self.path,'train')))
        self.IMG_H=212
        self.IMG_W=212
        self.chan=1
    
    def createCNN(self,summary=True):
        '''
        function that creates and compile the CNN
        args:
        	-IMG_H:Images height (def 212px)
        	-IMH_W:Images width (def 212px)
            -chan: color channels (def 1 i.e. grayscale)
        '''
		
        # Model Creation
        self.model = Sequential()
        self.model.add(Conv2D(16, 3, padding='same', activation='relu',
			input_shape=(self.IMG_H, self.IMG_W, self.chan)))
        self.model.add(MaxPooling2D())
        self.model.add(Dropout(0.2))
        self.model.add(Conv2D(32, 3, padding='same', activation='relu'))
        self.model.add(MaxPooling2D())
        self.model.add(Conv2D(64, 3, padding='same', activation='relu'))
        self.model.add(MaxPooling2D())
        self.model.add(Dropout(0.2))
        self.model.add(Conv2D(128, 3, padding='same', activation='relu'))
        self.model.add(MaxPooling2D())
        self.model.add(Dropout(0.2))
        self.model.add(Conv2D(256, 3, padding='same', activation='relu'))
        self.model.add(MaxPooling2D())
        self.model.add(Dropout(0.2))
        self.model.add(Flatten())
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dense(self.k, activation='softmax'))

		#Compile the model
        self.model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
              
        if summary:
            self.model.summary()
            tf.keras.utils.plot_model(
				model = self.model,
				rankdir="TB",
				dpi=72,
				show_shapes=True
				)
    
    def trainCNN(self,batch=32,epochs=8): 
        '''
        A function that trains a CNN given the model
        and the PATH of the data set.
        '''
        train_dir = os.path.join(self.path,'train')
        test_dir = os.path.join(self.path,'test')
        train_crystal_dir = os.path.join(train_dir,'0')
        train_fluid_dir = os.path.join(train_dir,'1')
        train_defective_dir = os.path.join(train_dir,'2')

        test_crystal_dir = os.path.join(test_dir,'0')
        test_fluid_dir = os.path.join(test_dir,'1')
        test_defective_dir = os.path.join(test_dir,'2')

        #Process the Data
        image_gen = ImageDataGenerator(rescale=1./255)
        train_data_gen = image_gen.flow_from_directory(
                                    #batch_size=batch,
                                    directory=train_dir,
                                    color_mode='grayscale',
                                    shuffle=True,
                                    target_size=(self.IMG_H, self.IMG_W),
                                    class_mode='categorical')
        #print(train_data_gen.shape)

        test_data_gen = image_gen.flow_from_directory(
                                    #batch_size=batch,
                                    directory=test_dir,
                                    color_mode='grayscale',
                                    target_size=(self.IMG_H, self.IMG_W),
                                    class_mode='categorical')
        
        num_crystal_train = len(os.listdir(train_crystal_dir))
        num_fluid_train = len(os.listdir(train_fluid_dir))
        num_defective_train = len(os.listdir(train_defective_dir))

        num_crystal_test = len(os.listdir(test_crystal_dir))
        num_fluid_test = len(os.listdir(test_fluid_dir))
        num_defective_test = len(os.listdir(test_defective_dir))

        total_train = num_crystal_train + num_fluid_train + num_defective_train
        total_test = num_crystal_test + num_fluid_test + num_defective_test

        history = self.model.fit(
                            train_data_gen,
                            steps_per_epoch=total_train // batch,
                            epochs=epochs,
                            validation_data=test_data_gen,
                            validation_steps=total_test // batch,
                            callbacks = [tf.keras.callbacks.EarlyStopping(
                            monitor='val_loss',
                            min_delta=0.01,
                            patience=7)])
        
        #Plot Accuracy, change this to matplotlib
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=history.epoch,
                            y=history.history['accuracy'],
                            mode='lines+markers',
                            name='Training accuracy'))
        fig.add_trace(go.Scatter(x=history.epoch,
                            y=history.history['val_accuracy'],
                            mode='lines+markers',
                            name='Validation accuracy'))
        fig.update_layout(title='Accuracy',
                    xaxis=dict(title='Epoch'),
                    yaxis=dict(title='Percentage'))
        fig.show()
