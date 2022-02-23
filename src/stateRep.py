import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.helpers import CNN_Helpers

class CNN(CNN_Helpers):
    def __init__(self, *args, **kwargs):
        super(CNN, self).__init__(*args, **kwargs)
        self.ds_path = './data/raw/cnn'
        self.weights_path = './models/cnn/CNN.h5'
        self.k = len(os.listdir(os.path.join(self.ds_path,'train')))
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
        train_dir = os.path.join(self.ds_path,'train')
        test_dir = os.path.join(self.ds_path,'test')
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

    def loadWeights(self,path):
        '''
        Functions that loads weight for the model
        args:
			-path: path from which to load weights
		'''
        if path == None:
            path = self.weights_path
        #Load model wieghts
        self.model.load_weights(path)
        print("Loaded model from disk")

    def runCNN(self,img_path):
        '''
        Method that receives image and returns label of image.
        args:
            -img_path: path to image to be classified. 
        returns:
            -label of the classified image.
            -probabilities of the classified image.
        '''
        img_batch = self.preProcessImg(img_path)
        prediction = self.model.predict(img_batch)
        #cat_index = np.argmax(prediction[0])
        
        return np.argmax(prediction[0]), prediction

class CNN_Testing(CNN):
    def __init__(self, *args, **kwargs):
        super(CNN_Testing, self).__init__(*args, **kwargs)

    def testCNN(self,path):
        '''
        Function that test CNN performance by callculating the 
        confusion matrix using the data in the specified path.
        args:
            -path: the path to the data sst to test. Set path = None
            to test using training/testing data set. 
        returns;
            -confution matrix: confusion matirx for the selected data.
            -missclassified data: csv containing the missclassified images.  
        '''
        if path == None:
            path = self.ds_path

        results_path = './results/cnn'
        Conf_Mat = np.zeros([self.k,self.k])
        data = pd.DataFrame()
        
        #CHECK REFACTOR FOR THIS LOOPS
        for dir_name in os.listdir(path):
            new_path = os.path.join(path,dir_name)
            if os.path.isdir(new_path):
                for tag in os.listdir(new_path):
                    real_state = int(tag)
                    new_path = os.path.join(os.path.join(path,dir_name),tag)
                    print(new_path)
                    for filename in os.listdir(new_path):
                        if filename.endswith(".png"):
                            img = os.path.join(new_path,filename)
                            s_cnn, _ = self.runCNN(img)
                            Conf_Mat[int(real_state),int(s_cnn)] += 1
                            if int(real_state) != int(s_cnn):
                                entry = {}
                                entry['Path'] = os.path.join(new_path,filename)
                                entry['CNN'] = s_cnn
                                entry['Real'] = real_state
                                data = data.append(entry,ignore_index=True)
        print(Conf_Mat)
        np.save(os.path.join(results_path,'ConfMat.npy'),Conf_Mat)
        print('Number of point in data set: ',Conf_Mat.sum())
        data.to_csv(os.path.join(results_path,'CNNerror_log.csv'),index=False)

        score = Conf_Mat.trace()/Conf_Mat.sum()
        print(score)
