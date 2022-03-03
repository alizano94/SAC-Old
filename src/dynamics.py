import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt

SMALL_SIZE = 24
MEDIUM_SIZE = 32
BIGGER_SIZE = 64

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Input, Lambda, LSTM
import tensorflow_probability as tfp


from src.stateRep import SNN_Asistance

class SNN(SNN_Asistance):
    def __init__(self,a,w,m,*args,**kwargs):
        super(SNN,self).__init__(*args,**kwargs)
        self.m = m
        self.w = w
        self.a = a

        weight_file = 'SNN-W'+str(self.w)+'-M'+str(self.m)+'.h5'
        ds_file = 'Balanced-W'+str(self.w)+'-M'+str(self.m)+'.csv'
        self.snn_preprocess_data_path = os.path.join(self.snn_preprocess_data_path,ds_file)
        self.snn_weights_path = os.path.join(self.snn_weights_path, weight_file)

    def createSNNDS(self):
        '''
        Function that creates the csv files that 
        serve as DS for the CNN
        '''

        data = pd.DataFrame()


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
        data = self.balanceData(data,method='expand')
        #data.drop(columns=['level_0'],inplace=True)
        print('Saving DS of size: '+str(len(data)))
        data.to_csv(self.snn_preprocess_data_path,index=False)

    def createSNN(self,summary=False):
        '''
        Function that creates and compile the SNN
        args:
            -summary: False by default, set to True
                        to get sumary of the model. 
        ''' 

        FEATURE_NAMES = []

        for i in range(self.m):
            name = 'S'+str(i-self.m)
            FEATURE_NAMES += [name]

        FEATURE_NAMES += ['V']


        inputs = {}
        for name in FEATURE_NAMES:
            inputs[name] = tf.keras.Input(shape=(1,), name=name)
        
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
        
    def trainSNN(self,epochs=100,batch=10,plot=False):
        '''
        A function that trains a SNN given the model
        and the PATH of the data set.
        '''

        data = pd.read_csv(self.snn_preprocess_data_path)

        
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

    def loadSNN(self,path):
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

    def runSNN(self,voltage_level,states):
        '''
		Function that runs SNN.
		Args:
			-V: voltage level applied to the transition
            -states: list containing the previuos necesary states.
		Returns:
			-out: label of the state after transition. 
		'''
        if len(states) < self.m:
            while len(states) < self.m:
                states.insert(0,states[0])
        elif len(states) > self.m:
            while len(states) > self.m:
                states.pop(0)
        else:
            pass
        input_feat = {'V':np.array([float(voltage_level)])}
        for i in range(self.m):
            name = 'S'+str(i-self.m)
            input_feat[name] = np.array([float(states[i])])

        probs = self.snn_model.predict(input_feat)
        cat_dist = tfp.distributions.Categorical(probs=probs[0])		
        out = cat_dist.sample(1)[0]
        return out

class SNN_Testing(SNN):
    def __init__(self,*args,**kwargs):
        super(SNN_Testing,self).__init__(*args,**kwargs)
        self.snn = SNN(a=self.a,w=self.w,m=self.m)
        self.snn.createSNN()
        self.snn.loadSNN(None)

    def getTranitionTensorDS(self):
        '''
        Calculates and plots transition tensor for the data set.
        '''
        tensor_file_name = 'DS_TransitionTensor-W'+str(self.w)+'-M'+str(self.m)+'.npy'
        tensor_plot_file_name = 'DS_TransitionTensor-W'+str(self.w)+'-M'+str(self.m)+'.png'
        tensor_save_path = os.path.join(self.snn_results_path,tensor_file_name)
        DS_df = pd.read_csv(self.snn_preprocess_data_path)
        trans_matrix = np.zeros((4,self.k,self.k))

        for _, rows in DS_df.iterrows():
            trans_matrix[int(rows['V'])-1,int(rows['S-1']),int(rows['S0'])] += 1

        np.save(tensor_save_path,trans_matrix)

        x=np.arange(self.k)
        
        for i in range(4):
            for j in range(3):
                plt.subplot(4,self.k,self.k*i+j+1)
                height=trans_matrix[i,j,:]/sum(trans_matrix[i,j,:])
                plt.bar(x,height=height,color='red')
                plt.ylim([0.0,1.0])
                if 3*i+j+1 > 9:
                    plt.xlabel('Tansition State')
                if 3*i+j+1 < 4:
                    plt.title('Initial State: '+str(x[j]))
                if j == 0:
                    plt.ylabel('Applied Voltage: V'+str(i+1))

        figure = plt.gcf()
        figure.set_size_inches(32,16)
        plt.savefig(os.path.join(self.snn_results_path,tensor_plot_file_name),dpi=100)
        plt.clf()


    def testSNN(self,replicate_DS=False):
        '''
        Calculates and plots transition tensor from pretrained model.
        args:
            -replicate_DS: if True takes inputs form DS and use it to 
                            calculate the tensor.
                            if False calculates the tensor using all 
                            the posible input states.
        '''
        
        tensor_file_name = 'SNN_TransitionTensor-W'+str(self.w)+'-M'+str(self.m)+'.npy'
        tensor_plot_file_name = 'SNN_TransitionTensor-W'+str(self.w)+'-M'+str(self.m)+'.png'
        tensor_save_path = os.path.join(self.snn_results_path,tensor_file_name)
        DS_df = pd.read_csv(self.snn_preprocess_data_path)
        trans_matrix = np.zeros((4,self.k,self.k))

        if replicate_DS:
            for _, rows in DS_df.iterrows():
                V = float(rows['V'])
                states = []
                for i in range(self.m):
                    name = 'S'+str(i-self.m)
                    states.insert(-1,int(rows[name]))
                out = self.snn.runSNN(V,states)
                trans_matrix[int(rows['V'])-1,int(rows['S-1']),int(out)] += 1
        else:
            for V in range(4):
                for initial_state in range(3):
                    states = [initial_state]
                    for i in range(1000):		
                        out = self.snn.runSNN(V+1,states)
                        trans_matrix[V,initial_state,int(out)] += 1

        np.save(tensor_save_path,trans_matrix)

        x=np.arange(self.k)

        for i in range(4):
            for j in range(3):
                plt.subplot(4,self.k,self.k*i+j+1)
                height=trans_matrix[i,j,:]/sum(trans_matrix[i,j,:])
                plt.bar(x,height=height,color='black')
                plt.ylim([0.0,1.0])
                plt.xticks(x,['Fluid','Defective','Crystal'])
                if 3*i+j+1 > 9:
                    plt.xlabel('Tansition State')
                if 3*i+j+1 < 4:
                    plt.title('Initial State: '+str(x[j]))
                if j == 0:
                    plt.ylabel('Applied Voltage: V'+str(i+1))

        figure = plt.gcf()
        figure.set_size_inches(32,16)
        plt.savefig(os.path.join(self.snn_results_path,tensor_plot_file_name),dpi=100)
        plt.clf()

    def getTrajectoryHistogran(self,initial_image,N,l):
        '''
        Plots a time dependent histogram.
        args:
            -initial_image: path to initial state image.
            -N: Nunmber of simulations.
            -l: lenght of the trajectory for each simulation.
        '''

        for v in range(self.a):
            trajectory_hist = []
            for i in range(l*(self.k+1)):
                trajectory_hist += [0]
            x = np.arange(len(trajectory_hist))
            state, _ = self.runCNN(initial_image)
            for n in range(N):
                for i in range(l):
                    trajectory_hist[4*i+state] += 1
                    state = self.runSNN(v+1,[state])
            for i in range(len(trajectory_hist)):
                trajectory_hist[i] = trajectory_hist[i]/N
            plt.subplot(2,2,v+1)
            plt.bar(x,height=trajectory_hist,color='black')
            xticks = []
            for i in range(self.k):
                xticks += [str(i)]
            xticks += ['']
            xticks = xticks*l
            plt.xticks(x,xticks)
            x_minor = []
            x_minor_lbs = []
            for i in range(l):
                label = str(l+1)+'step'
                x_minor += [(self.k-1)/2+l*(self.k+1)]
                x_minor_lbs += [label]
            #plt.xticks(x_minor,x_minor_lbs,minor=True)
            plt.ylim([0.0,1.2])
            plt.yticks([0.0,0.5,1.0],['0.0','0.5','1.0'])
            plt.ylabel('Frecuecy')
            plt.title('Voltage level '+str(v+1))

        figure = plt.gcf()
        figure.set_size_inches(32,18)    
        plt.savefig(os.path.join(self.snn_results_path,'Trajectories.png'),dpi=100)
        plt.clf()
    
    def getTrajectories(self,initial_image,l):
        '''
        Get single trajectory plots for each voletage leve.
        args:
            -
        '''
        for v in range(self.a):
            state, _ = self.runCNN(initial_image)
            trajectory_hist = [state]
            for i in range(l):
                state = self.runSNN(v+1,[state])
                trajectory_hist += [state]
            x = np.arange(len(trajectory_hist))
            plt.subplot(2,2,v+1)
            plt.plot(x,trajectory_hist,color='black', linewidth=3)
            plt.ylim([-0.5,2.5])
            ylabels = ['Fluid','Deffective','Crystal']
            plt.yticks(np.arange(3),ylabels)
            plt.xlabel('Time step.')
            plt.title('Voltage level '+str(v+1))

        figure = plt.gcf()
        figure.set_size_inches(32,18)    
        plt.savefig(os.path.join(self.snn_results_path,'Trajectory.png'),dpi=100)
        plt.clf()


class Control_Asistance(SNN):
    def __init__(self,*args, **kwargs):
        super(Control_Asistance,self).__init__(*args,**kwargs)
        