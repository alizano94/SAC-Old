import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Input, Lambda, LSTM
import tensorflow_probability as tfp


from src.stateRep import SNN_Asistance

class SNN(SNN_Asistance):
    def __init__(self,w,m,*args,**kwargs):
        super(SNN,self).__init__(*args,**kwargs)
        self.m = m
        self.w = w

        weight_file = 'SNN-W'+str(self.w)+'-M'+str(self.m)+'.h5'
        self.snn_weights_path = os.path.join(self.snn_ds_path, weight_file)

    def createSNNDS(self):
        '''
        Function that creates the csv files that 
        serve as DS for the CNN
        Arguments:
            -path: path to SNN DS dir
            -model: CNN model to predict
        '''

        data = pd.DataFrame()

        sep = '","'
        ds_name = 'Balanced-W'+str(self.w)+'-M'+str(self.m)+'.csv'
        for v_dir in os.listdir(self.snn_ds_path):
            v_path = os.path.join(self.snn_ds_path,v_dir)
            if os.path.isdir(v_path):
                V = v_dir.replace('V','')
                for sampling_dir in os.listdir(v_path):
                    sts_path = os.path.join(v_path,sampling_dir)
                    if os.path.isdir(sts_path):
                        sts_step = sampling_dir.replace('s','')
                        for traj_dir in os.listdir(sts_path):
                            traj_path = os.path.join(sts_path,traj_dir)
                            T = traj_dir.replace('T','')
                            if os.path.isdir(traj_path):
                                csv_name = 'V'+str(V)+'-'+str(sts_step)+'s-T'+str(T)+'.csv'
                                csv_path = os.path.join(traj_path,csv_name)
                                if os.path.exists(csv_path):
                                    if V != 'R':
                                        csv_df = pd.read_csv(csv_path)
                                        csv_df = self.windowResampling(csv_df,
                                            int(sts_step),self.w,self.m)
                                        data = data.append(csv_df)
                                    elif V == 'R' and int(sts_step) == self.w:
                                        csv_df = pd.read_csv(csv_path)
                                        csv_df = self.windowResampling(csv_df,
                                            int(sts_step),int(sts_step),self.m)
                                        data = data.append(csv_df)
                                    else:
                                        pass

        data.reset_index(inplace=True)						
        #data = self.helpers.DropBiasData(data)
        #data.drop(columns=['level_0'],inplace=True)
        print('Saving DS of size: '+str(len(data)))
        print(ds_name)
        ds_file = os.path.join(self.preprocess_data_path,'snn')
        ds_file = os.path.join(ds_file,ds_name)
        data.to_csv(ds_file,index=False)

    def createRNN(self,summary=False,keep_v=False):
        '''
        Function that creates and compile the SNN
        args:
            -step: time memory size
        returns:
            -model: stochastic/recurrent model.
        ''' 

        FEATURE_NAMES = []

        for i in range(self.m):
            name = 'S'+str(i-self.m)
            FEATURE_NAMES += [name]

        if keep_v:
            for i in range(self.m):
                name = 'V'+str(i-self.m)
        else:
            FEATURE_NAMES += ['V']


        inputs = {}
        for name in FEATURE_NAMES:
            inputs[name] = tf.keras.Input(shape=(1,), name=name)
        print(inputs)

        if keep_v:
            memory = inputs.copy()
            for i in range(self.m):
                name = 'V'+str(i-self.m)
                memory.pop(name)
        else:
            memory = inputs.copy()
            memory.pop('V')

        features = keras.layers.concatenate(list(memory.values()))
        features = layers.BatchNormalization()(features)
        features = tf.keras.layers.Reshape((self.m,1), input_shape=(2,))(features)
        for units in [16,32]:
            features = LSTM(units,
                activation="tanh",
                recurrent_activation="sigmoid",
                dropout=0.2,
                recurrent_dropout=0.2)(features)
            features = tf.keras.layers.Reshape((units,1))(features)

        features = keras.layers.Flatten()(features)
        features = keras.layers.concatenate([features,inputs['V']])
        features = layers.BatchNormalization()(features)

        # Create hidden layers with weight uncertainty 
        #using the DenseVariational layer.
        for units in [64,128]:
            features = layers.Dense(units=units,activation="sigmoid")(features)
            features = layers.Dropout(0.3)(features)

        # The output is deterministic: a single point estimate.
        outputs = layers.Dense(self.k, activation='softmax')(features)

        self.snn_model = keras.Model(inputs=inputs, outputs=outputs)

        #opt = keras.optimizers.Adam(learning_rate=0.01)


        self.snn_model.compile(
            loss = 'categorical_crossentropy',
            optimizer='adam'
            )
        if summary:
            self.snn_model.summary()
            tf.keras.utils.plot_model(
                model = self.snn_model,
                rankdir="TB",
                dpi=72,
                show_shapes=True
                )
        
    def trainSNN(self,epochs=10,batch=5,plot=False):
        '''
        A function that trains a SNN given the model
        and the PATH of the data set.
        '''

        data = pd.read_csv(self.snn_ds_path)

        
        train_features = self.df2dict(data)
        train_labels = self.onehotencoded(data)

        print(train_features)
        print(train_labels)

        history = self.snn_model.fit(train_features,
            train_labels,
            epochs=epochs,
            batch_size=batch,
            validation_split=0.2,
            verbose=2
            )
        
        self.snn_model.save_weights(self.snn_weights_path)

        if plot:
            #Plot Accuracy, change this to matplotlib
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=history.epoch,
                                y=history.history['loss'],
                                mode='lines+markers',
                                name='Training accuracy'))
            fig.add_trace(go.Scatter(x=history.epoch,
                                y=history.history['val_loss'],
                                mode='lines+markers',
                                name='Validation accuracy'))
            fig.update_layout(title='Loss',
                        xaxis=dict(title='Epoch'),
                        yaxis=dict(title='Loss'))
            fig.show()

    def loadWeights(self,path):
        '''
        Functions that loads weight for the model
        args:
			-path: path from which to load weights
		'''
        if path == None:
            path = self.snn_weights_path
        #Load model wieghts
        self.snn_model.load_weights(path)
        print("Loaded model from disk")

