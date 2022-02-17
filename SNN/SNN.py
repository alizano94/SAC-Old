import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Input, Lambda, LSTM
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from CNN.CNN import *

from Utils.Helpers import *
from sklearn.preprocessing import MinMaxScaler


class SNN():
	def __init__(self):
		self.helpers = Helpers()
		self.cnn = CNN()

	def createDS(self,path,window,memory):
		'''
		Function that creates the csv files that 
		serve as DS for the CNN
		Arguments:
			-path: path to SNN DS dir
			-model: CNN model to predict
		'''

		data = pd.DataFrame()

		sep = '","'
		ds_name = path+'/'+'Balanced-W'+str(window)+'-M'+str(memory)+'.csv'
		for v_dir in os.listdir(path):
			v_path = os.path.join(path,v_dir)
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
										csv_df = self.helpers.windowResampling(csv_df,
											int(sts_step),window,memory)
										data = data.append(csv_df)
									elif V == 'R' and int(sts_step) == window:
										csv_df = pd.read_csv(csv_path)
										csv_df = self.helpers.windowResampling(csv_df,
											int(sts_step),int(sts_step),memory)
										data = data.append(csv_df)
									else:
										pass

		data.reset_index(inplace=True)						
		data = self.helpers.DropBiasData(data)
		#data.drop(columns=['level_0'],inplace=True)
		print('Saving DS of size: '+str(len(data)))
		print(ds_name)
		data.to_csv(ds_name,index=False)
											            


	def prior(self,kernel_size, bias_size, dtype=None):
		n = kernel_size + bias_size
		prior_model = keras.Sequential(
	        [
	            tfp.layers.DistributionLambda(
	                lambda t: tfp.distributions.MultivariateNormalDiag(
	                    loc=tf.zeros(n), scale_diag=tf.ones(n)
	                )
	            )
	        ]
	    )
		return prior_model


	def posterior(self,kernel_size, bias_size, dtype=None):
		n = kernel_size + bias_size
		posterior_model = keras.Sequential(
	        [
	            tfp.layers.VariableLayer(
	                tfp.layers.MultivariateNormalTriL.params_size(n), dtype=dtype
	            ),
	            tfp.layers.MultivariateNormalTriL(n),
	        ]
	    )
		return posterior_model



	def createCSNN(self,step,summary=False):
		'''
		Function that creates and compile the SNN
		args:
			-step: time memory size
		returns:
			-model: stochastic/recurrent model.
		''' 

		FEATURE_NAMES = [
			'Si',
			'V']

		inputs = {}
		for name in FEATURE_NAMES:
			inputs[name] = tf.keras.Input(shape=(1,), name=name)
		
		features = keras.layers.concatenate(list(inputs.values()))
		features = layers.BatchNormalization()(features)

		# Create hidden layers with weight uncertainty 
		#using the DenseVariational layer.
		for units in [16,32]:
			features = tfp.layers.DenseVariational(
				units=units,
				make_prior_fn=self.prior,
				make_posterior_fn=self.posterior,
				activation="sigmoid",
				)(features)
			featrues = layers.Dropout(0.2)

		# The output is deterministic: a single point estimate.
		outputs = layers.Dense(3, activation='softmax')(features)

		self.model = keras.Model(inputs=inputs, outputs=outputs)

		self.model.compile(
			loss = 'categorical_crossentropy',
			optimizer='adam'
			)
		if summary:
			self.model.summary()
			tf.keras.utils.plot_model(
				model = self.model,
				rankdir="TB",
				dpi=72,
				show_shapes=True
				)

		return self.model

	def createDNN(self,step,summary=False):
		'''
		Function that creates and compile the SNN
		args:
			-step: time memory size
		returns:
			-model: stochastic/recurrent model.
		''' 

		FEATURE_NAMES = [
			'Si',
			'V',
			't']

		inputs = {}
		for name in FEATURE_NAMES:
			inputs[name] = tf.keras.Input(shape=(1,), name=name)
		
		features = keras.layers.concatenate(list(inputs.values()))
		features = layers.BatchNormalization()(features)

		# Create hidden layers with weight uncertainty 
		#using the DenseVariational layer.
		for units in [16,32,64]:
			features = layers.Dense(units=units,activation="sigmoid")(features)
			features = layers.Dropout(0.2)(features)

		# The output is deterministic: a single point estimate.
		outputs = layers.Dense(3, activation='softmax')(features)

		self.model = keras.Model(inputs=inputs, outputs=outputs)

		self.model.compile(
			loss = 'categorical_crossentropy',
			optimizer='adam'
			)
		if summary:
			self.model.summary()
			tf.keras.utils.plot_model(
				model = self.model,
				rankdir="TB",
				dpi=72,
				show_shapes=True
				)

		return self.model

	def createRNN(self,step,summary=False,keep_v=False):
		'''
		Function that creates and compile the SNN
		args:
			-step: time memory size
		returns:
			-model: stochastic/recurrent model.
		''' 

		FEATURE_NAMES = []

		for i in range(step):
			name = 'S'+str(i-step)
			FEATURE_NAMES += [name]

		if keep_v:
			for i in range(step):
				name = 'V'+str(i-step)
		else:
			FEATURE_NAMES += ['V']


		inputs = {}
		for name in FEATURE_NAMES:
			inputs[name] = tf.keras.Input(shape=(1,), name=name)
		print(inputs)

		if keep_v:
			memory = inputs.copy()
			for i in range(step):
				name = 'V'+str(i-step)
				memory.pop(name)
		else:
			memory = inputs.copy()
			memory.pop('V')

		features = keras.layers.concatenate(list(memory.values()))
		features = layers.BatchNormalization()(features)
		features = tf.keras.layers.Reshape((step,1), input_shape=(2,))(features)
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
			features = layers.Dropout(0.2)(features)

		# The output is deterministic: a single point estimate.
		outputs = layers.Dense(3, activation='softmax')(features)

		self.model = keras.Model(inputs=inputs, outputs=outputs)

		#opt = keras.optimizers.Adam(learning_rate=0.01)


		self.model.compile(
			loss = 'categorical_crossentropy',
			optimizer='adam'
			)
		if summary:
			self.model.summary()
			tf.keras.utils.plot_model(
				model = self.model,
				rankdir="TB",
				dpi=72,
				show_shapes=True
				)

		return self.model

	def createDSNN(self,step,summary=False):
		'''
		Function that creates and compile the SNN
		args:
			-step: time memory size
		returns:
			-model: stochastic/recurrent model.
		''' 

		FEATURE_NAMES = [
			'Si',
			'V']

		inputs = {}
		for name in FEATURE_NAMES:
			inputs[name] = tf.keras.Input(shape=(1,), name=name)

		featrues_Si = layers.BatchNormalization()(inputs['Si'])
		featrues_V = layers.BatchNormalization()(inputs['V'])
		
		for units in [8,16]:
			featrues_Si = tfp.layers.DenseVariational(
				units = units,
				make_prior_fn=self.prior,
				make_posterior_fn=self.posterior,
				activation="softmax",
				)(featrues_Si)
			featrues_Si = layers.Dropout(0.2)(featrues_Si)

		for units in [8,16]:
			featrues_V = layers.Dense(units=units,activation="sigmoid")(featrues_V)
			featrues_V = layers.Dropout(0.2)(featrues_V)

		features = keras.layers.concatenate([featrues_Si,featrues_V])

		# The output is deterministic: a single point estimate.
		outputs = layers.Dense(3, activation='softmax')(features)

		self.model = keras.Model(inputs=inputs, outputs=outputs)

		self.model.compile(
			loss = 'categorical_crossentropy',
			optimizer='adam'
			)
		if summary:
			self.model.summary()
			tf.keras.utils.plot_model(
				model = self.model,
				rankdir="TB",
				dpi=72,
				show_shapes=True
				)

		return self.model

	def trainModel(self,path,epochs=10,batch=5,plot=True):
		'''
		A function that trains a SNN given the model
		and the PATH of the data set.
		'''

		data = pd.read_csv(path)

		
		train_features = self.helpers.df2dict(data)
		train_labels = self.helpers.onehotencoded(data)

		print(train_features)
		print(train_labels)

		history = self.model.fit(train_features,
			train_labels,
			epochs=epochs,
			batch_size=batch,
			validation_split=0.2,
			verbose=2
			)

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

	def runSNN(self,inp):
		'''
		Function that runs SNN.
		Args:
			-model: SNN model object
			-inp: input state
		Returns:
			-out
		'''

		out = self.model.predict(inp)
		return out

	def runSNN2(self,model,inp):
		'''
		Function that runs SNN.
		Args:
			-model: SNN model object
			-inp: input state
		Returns:
			-out
		'''
		probs = model.predict(inp)
		cat_dist = tfp.distributions.Categorical(probs=probs[0])		
		out = cat_dist.sample(1)[0]
		return out

	def trajectory(self,step,init,length):
		'''
		Function that runs SNN.
		Args:
			-model: model to obtain probabilities from
			-inp: dict containing input features
		Returns:
			-trajectory: list with predicted trajectories.
		'''

		trajectory = [init['S-1']]
		v_traj = [init['V']]
		for i in range(length):
			#print(init)
			v_traj.append(init['V'])
			probs = self.runSNN(self.model,init)
			cat_dist = tfp.distributions.Categorical(probs=probs[0])		
			So = cat_dist.sample(1)[0]
			
			trajectory.append(int(So))
			init['S-1'] = np.array([So])

			for i in range(step-1):
				name = 'S'+str(i-step)
				past_state = 'S'+str(i-step+1)
				init[name] = init[past_state]
		return trajectory, v_traj

	def testSNN(self,W,M,k):
		'''
		'''
		test_all_states = False
		replicate_DS = True

		DS_path = '/home/lizano/Documents/SAC/SNN/DS'
		csv_DS_name = 'Balanced-W'+str(W)+'-M'+str(M)+'.csv'
		csv_DS_path = os.path.join(DS_path,csv_DS_name)
		tensor_save_path = os.path.join(DS_path,'SNN_TransitionTensor.npy')
		DS_df = pd.read_csv(csv_DS_path)
		trans_matrix = np.zeros((4,3,3))

		if test_all_states:
			for v in range(4):
				for encode_state in range(k**M):
					decode_state = self.helpers.stateDecoder(k,encode_state,M)
					inp_feat = {'V':np.array([float(v+1)])}
					for j in range(M):
						name = 'S'+str(j-M)
						state_value = decode_state[j]
						inp_feat[name] = np.array([float(state_value)])
					print(inp_feat)
					for i in range(100):
						hist = self.model.predict(inp_feat)
						cat_dist = tfp.distributions.Categorical(probs=hist[0])		
						out = cat_dist.sample(1)[0]
						trans_matrix[int(inp_feat['V'])-1,int(inp_feat['S-1']),int(out)] += 1
		elif replicate_DS:
			for _, rows in DS_df.iterrows():
				inp_feat = {'V':np.array([float(rows['V'])])}
				for i in range(M):
					name = 'S'+str(i-M)
					inp_feat[name] = np.array([float(rows[name])])
				print(inp_feat)
				hist = self.model.predict(inp_feat)
				cat_dist = tfp.distributions.Categorical(probs=hist[0])		
				out = cat_dist.sample(1)[0]
				trans_matrix[int(rows['V'])-1,int(rows['S-1']),int(out)] += 1
		else:
			for v in range(4):
				for initial_state in range(3):
					inp_feat = {'V':np.array([float(v+1)])}
					for j in range(M):
						name = 'S'+str(j-M)
						inp_feat[name] = np.array([float(initial_state)])
					print(inp_feat)
					hist = self.model.predict(inp_feat)
					print(hist)
					cat_dist = tfp.distributions.Categorical(probs=hist[0])
					for i in range(10000):		
						out = cat_dist.sample(1)[0]
						trans_matrix[int(inp_feat['V'])-1,int(inp_feat['S-1']),int(out)] += 1
		
		np.save(tensor_save_path,trans_matrix)

		x=np.arange(3)
		titles = ['Fluid','Defective','Crystal']

		for i in range(4):
			for j in range(3):
				plt.subplot(4,3,3*i+j+1)
				height=trans_matrix[i,j,:]/sum(trans_matrix[i,j,:])
				plt.bar(x,height=height,color='black')
				plt.ylim([0.0,1.0])
				plt.xticks(x,['Fluid','Defective','Crystal'])
				if 3*i+j+1 > 9:
					plt.xlabel('Tansition State')
				if 3*i+j+1 < 4:
					plt.title('Initial State: '+titles[j])
				if j == 0:
					plt.ylabel('Applied Voltage: V'+str(i+1))

		plt.show()