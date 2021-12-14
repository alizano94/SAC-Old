import os

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from IPython.display import clear_output


from Utils.Helpers import *
from CNN.CNN import *
from SNN.SNN import *
from RL.RL import *

#load classes
Helpers = Helpers()
CNN = CNN()
SNN = SNN()
RL = RL()


#Define Variables
#FLAGS
cnn_train = False
snn_train = False
rl_train = True
preprocess_snnDS = False
Test_CNN = False
Test_SNN = False
Test_RL = False



#Parameters
k = 3
memory = 4
window = 100
V_levels = 4

#paths
cnn_ds_dir = './CNN/DS'
snn_ds_dir = '/home/lizano/Documents/SAC/SNN/DS'
rl_ds_dir = './RL/Qtables/'
csv_snnDS_path = snn_ds_dir+'/Balanced-W'+str(window)+'-M'+str(memory)+'.csv'
weights_dir = './SavedModels/'
cnn_weights = weights_dir+'CNN.h5'
snn_weights = weights_dir+'SNN'+str(window)+'-M'+str(memory)+'.h5'
q_table_file = rl_ds_dir+str(k**(memory))+'X'+str(V_levels)+'Q_table'+str(memory)+'M.npy'

#CNN
#Create CNN model
cnn_model = CNN.createCNN(summary=False)
#Tain the model or load learning
if cnn_train:
	if os.path.isfile(cnn_weights):
		os.remove(cnn_weights)
	CNN.trainCNN(cnn_ds_dir,cnn_model,epochs=100)
	Helpers.saveWeights(cnn_model,cnn_weights)
else:
	print('Loading CNN model...')
	Helpers.loadWeights(cnn_weights,cnn_model)


#SNN
#Create SNN model
snn_model = SNN.createRNN(memory,summary=False)
#Data Set Hadelling
if os.path.isfile(csv_snnDS_path):
	pass
else:
	if preprocess_snnDS:
		Helpers.preProcessSNNDS(snn_ds_dir,cnn_model)
	SNN.createDS(snn_ds_dir,window,memory)
#Tain the model or load learning
if snn_train:
	if os.path.isfile(snn_weights):
		os.remove(snn_weights)
	SNN.trainModel(csv_snnDS_path,snn_model,epochs=100,batch=6)
	Helpers.saveWeights(snn_model,snn_weights)
else:
	print('Loading SNN model...')
	Helpers.loadWeights(snn_weights,snn_model)


#RL
if rl_train:
	if os.path.isfile(q_table_file):
		os.remove(q_table_file)
	q_table = RL.get_Q_table(snn_model,memory,k,a_size=V_levels)
else:
	pass
	print('Loading Q table...')
	q_table = np.load(q_table_file)

#Test Data 
#Test CNN accuracy.
if Test_CNN:
	score = CNN.testCNN(snn_ds_dir,cnn_model)
	print('Prediction accuracy for CNN :'+str(score))

if Test_SNN:
	Helpers.DataTrasnProbPlot(window,memory,k)
	SNN.testSNN(snn_model,window,memory,k)

if Test_RL:
	#Add RL heat map method
	pass
