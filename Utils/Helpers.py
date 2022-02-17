import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from random import seed, randint

from CNN.CNN import *

class Helpers():
	'''
	Helper functions for the loop
	'''
	def __init__(self):
		self.cnn = CNN()
		
	
	def convertToMatrix(self,data,step):
		'''
		A function that takes a series of data 
		and returns the X and Y tensors that serves
		as DS for the SNN
		args:
			-data: Numpy type array containing 
				   the data
			-step: The ammount of previous steps
				   to store in memory.
		'''
		X, Y =[], []
		for i in range(len(data[0,:])-step):
			d=i+step  
			X.append(data[:,i:d])
			Y.append(data[0,d])
		return np.array(X), np.array(Y)

	def plotImages(images_arr):
		'''
		A function that plots images in the form of 
		a grid with 1 row and 3 columns where images 
		are placed in each column.

		args: 
			-images_arr: array of images to plot
		'''
		fig, axes = plt.subplots(1, 3, figsize=(20,20))
		axes = axes.flatten()
		for img, ax in zip( images_arr, axes):
			ax.imshow(img)
			ax.axis('off')
		plt.tight_layout()
		plt.show()

	def preProcessImg(self,img_path,IMG_H=212,IMG_W=212):
		'''
		A function that preprocess an image to fit 
		the CNN input.
		args:
			-img_path: path to get image
			-IMG_H: image height
			-IMG_W: image width
		Returns:
			-numpy object containing:
				(dum,img_H,img_W,Chanell)
		'''
		#Load image as GS with st size
		img = image.load_img(img_path,color_mode='grayscale',target_size=(IMG_H, IMG_W))
		#save image to array (H,W,C)
		img_array = image.img_to_array(img)

		#Create a batch of images
		img_batch = np.expand_dims(img_array, axis=0)
		return img_batch

	def preProcessSNNDS(self,path,model):
		'''
		Method that takes op1.txt and plots and creates 
		csv containing information about time, order params,
		real state, cnnn state and voltage.
		'''

		sep = '","'
		for v_dir in os.listdir(path):
			v_path = os.path.join(path,v_dir)
			if os.path.isdir(v_path):
				for step_dir in os.listdir(v_path):
					step_path = os.path.join(v_path,step_dir)
					if os.path.isdir(step_path):
						for t_dir in os.listdir(step_path):
							t_path = os.path.join(step_path,t_dir)
							if os.path.isdir(t_path):
								op_path = os.path.join(t_path,'op1.txt')
								if os.path.exists(op_path):
									os.chdir(t_path)
									csv_name = v_dir+'-'+step_dir+'-'+t_dir+'.csv'
									os.system("awk '{print $1,"
										+sep+",$2,"
										+sep+",$3,"
										+sep+",$4,"
										+sep+",$5,"
										+sep+",$6,"
										+sep+",$7}' op1.txt > test.txt")
									data = pd.read_csv('test.txt', header=None)
									data.columns = ['Time','C6_avg','rgmean','psi6','RC','V','lambda']
									data = data.drop(labels=['RC','lambda','rgmean'],axis=1)
									states = pd.DataFrame(columns = ['S_cnn', 'S_param'])
									for i in range(0,len(data.index)):
										file_name = t_path+'/plots/'+v_dir+'-'+t_dir+'-'+str(i)+'step'+step_dir+'.png'
										img_batch = self.preProcessImg(file_name)
										s_cnn, _ = self.cnn.runCNN(model,img_batch)
										c6 = data.iloc[i]['C6_avg']
										psi6 = data.iloc[i]['psi6']
										if c6 <= 4.0:
											s_real = 0
										elif c6 > 4.0 and psi6 < 0.99:
											s_real = 1
										else:
											s_real = 2
										states_dict = {'S_cnn':s_cnn,'S_param':s_real}
										states = states.append(states_dict,ignore_index=True)
									data = pd.concat([data,states], axis=1)
									print(data)
									data.to_csv(csv_name,index=False)
									os.system('rm -rf test.txt')
									os.chdir(path)

	def windowResampling(slef,data,sampling_ts,window,memory):
		'''
		Receives data in a dataframe and returns data frame 
		with resampled data using slinding window method
		'''
		standard = ['Time','C6_avg','psi6','V']
		columns = []+standard
		for i in range(memory+1):
			name = 'S'+str(-memory+i)
			columns += [name]

		out_df = pd.DataFrame(columns=columns)
		new_size = int(len(data) - memory*window/sampling_ts)
		#refactor this to use all the data set competting with augmented data

		for index, rows in data.iterrows():
			row = {}
			if index < new_size:
				for name in standard:
					i = int(index+(memory-1)*window/sampling_ts)
					row[name] = data.at[i,name]
				for m in range(memory+1):
					name = 'S'+str(-memory+m)
					i = int(index+m*window/sampling_ts)
					row[name] = data.at[i,'S_param']
				#print(row)
				out_df = out_df.append(row,ignore_index=True)
		

		return out_df

	def createhist(self,data):
		'''
		create histogram from labels data
		'''
		hist = [0,0,0]

		for index, rows in data.iterrows():
			hist[int(rows['S0'])] += 1
		print(hist)

		return hist

	def DropBiasData(self,data):
		'''
		Resamples the data to ensure theres no BIAS on 
		ouput state dsitribution.
		'''
		seed(1)

		hist = self.createhist(data)

		min_hist = min(hist)
		arg_min = np.argmin(hist)

		while max(hist) != min_hist:
			index = randint(0,len(data)-1)
			hist_index = int(data['S0'][index])
			if hist[hist_index] > min_hist:
				data.drop(index=index, inplace=True)
			hist = self.createhist(data)
			data.reset_index(inplace=True)
			data.drop(columns=['index'],inplace=True)

		return data

	def df2dict(self,df,dtype=float):
		'''
		Takes a df and returns a dict of tensors
		'''
		out_dict = {name: np.array(value,dtype=dtype)
					for name, value in df.items()}

		return out_dict

	def onehotencoded(self,df,dtype=float):
		'''
		Transforms array with out state into one hot encoded vector
		'''
		array = np.array(df['S0'],dtype=dtype)
		onehotencoded_array = np.zeros((len(array),3),dtype=int)
		for i in range(len(array)):
			index = int(array[i])
			onehotencoded_array[i][index] = 1

		return onehotencoded_array


	def preProcessTens(self,csv,print_tensors=False):
		'''
		Function that takes data form csv 
		and creates a pd Data Frame to use as 
		'''
		dataset = pd.read_csv(csv)
		train_features = dataset.copy()
		train_features.drop(train_features.tail(1).index,inplace=True)
		train_features.drop(columns=['cat'],inplace=True)
		#train_features.drop(columns=['cat','#time'],inplace=True)

		train_labels = dataset.copy()
		train_labels.drop(columns=['cat','V_level','#time'],inplace=True)
		train_labels.drop(train_labels.head(1).index,inplace=True)		

		return train_features, train_labels

	


	def Data2df(self,PATH,step,window):
		'''
		Takes directory with csv and loads them into a DF
		'''
		train_features = pd.DataFrame()
		train_labels = pd.DataFrame()
		
		for file in os.listdir(PATH):
			if file.endswith('.csv'):
				train_csv = PATH+'/'+file
				X, Y = self.preProcessTens(train_csv)
				tmp_X = pd.DataFrame()#columns=['V','S1','S2'])
				tmp_Y = pd.DataFrame(columns=['S0'])
				for index, rows in X.iterrows():
					new_size = len(X) - window
					if index < new_size and index > step-1:
						x_dict = {}
						for i in range(step):
							name = 'S'+str(-i-1)
							x_dict[name] = X.at[index-i,'cat_index']
						x_dict['V'] = rows['V_level']
						tmp_X = tmp_X.append(x_dict,ignore_index=True)
						tmp_Y = tmp_Y.append(
							{'S0':Y.loc[index+window,'cat_index']
							},ignore_index=True)


				train_features = train_features.append(tmp_X)
				train_labels = train_labels.append(tmp_Y)
		train_features.reset_index(inplace=True)
		train_labels.reset_index(inplace=True)


		return train_features, train_labels


	def DataTrasnProbPlot(self,W,M,k,Balanced=True):
		'''
		Plot transition probabilities from Dataset
		'''
		size = k**M
		hist = []
		bars = []
		volt_lvl = [1,2,3,4]
		for i in range(k):
			state = 'S'+str(i)
			hist += [0]
			bars += [state]

		x_pos = np.arange(len(bars))
		plt.yticks(color='black')
		
		if Balanced:
			csv_path = '/home/lizano/Documents/CSA-Loop/SNN/DS/Balanced-W'+str(W)+'-M'+str(M)+'.csv'
		else:
			csv_path = '/home/lizano/Documents/CSA-Loop/SNN/DS/UnBalanced-W'+str(W)+'-M'+str(M)+'.csv'

		save_path = '/home/lizano/Documents/CSA-Loop/Results/DataSet/plots/'

		data = pd.read_csv(csv_path)

		col_list = ['V']
		for i in range(M):
			name = 'S'+str(i-M)
			col_list += [name]

		for V in volt_lvl:
			for encoded in range(size):
				state = self.stateDecoder(k,encoded,M)
				print(state)
				for i in range(k):
					hist[i] = 0
				search_dict = {'V':V}
				for i in range(M):
					name = 'S'+str(i-M)
					search_dict[name] = state[i]
				print(search_dict)

				for index, row in data.iterrows():
					row_dict = row[col_list].to_dict()
					s0 = int(row['S0'])
					if row_dict == search_dict:
						hist[s0] += 1

				if sum(hist) != 0:
					hist[:] = [x/sum(hist) for x in hist]
				print(hist)

				fig_name = save_path + 'V'+str(V)
				for i in range(M):
					name = 'S'+str(i-M)
					fig_name += '-'+name+str(int(search_dict[name]))
				fig_name += '.png'
				plt.xticks(x_pos, bars, color='black')
				plt.bar(x_pos,hist,color='red')
				plt.savefig(fig_name)
				plt.clf()



	def saveWeights(self,model,save_path):
		model.save_weights(save_path, save_path)

	def loadWeights(self,load_path,model):
		'''
		Functions that loads weight for the model
		args:
			-load_path: path from which to load weights
			-model: keras model
		'''
		#Load model wieghts
		model.load_weights(load_path)
		print("Loaded model from disk")

	def csv2df(self,csvname):
		'''
		Function that extracts info from csv 
		and creates array.
		'''
		results = []
		with open(csvname) as csvfile:
			reader = csv.reader(csvfile) # change contents to floats
			headers = next(reader) 
			for row in reader:
				results.append(row)

		x = []
		for i in range(0,len(results)):
			x.append(float(results[i][1]))

		df = pd.DataFrame(x)

		return df

	def plotList(self,list,x_lab,y_lab,save_path='./out.png'):
		'''
		Function that plots elements on a list
		'''
		data = np.zeros((2,len(list)))
		for i in range(len(list)):
			data[0][i] = float(i)
			data[1][i] = list[i]
		plt.figure()
		plt.figure()
		plt.yticks([0, 1, 2], ['Fluid', 'Deffective', 'Crystal']
			,rotation=45)
		plt.ylim(-.5,2.5)
		plt.xlabel(x_lab)
		#plt.ylabel(y_lab)
		plt.ylim(-.5,2.5)
		plt.scatter(data[:][0],data[:][1],s=0.5)
		plt.savefig(save_path)

	def stateEncoder(self,k,state):
		'''
		Method that encondes a given state into 
		a number
		'''
		s = 0
		m = len(state)
		for i in range(m):
			j = m - i -1
			s += state[j]*k**i
		return s

	def stateDecoder(self,k,state,m):
		'''
		Method that decodes stae from number to 
		input vector.
		'''
		done = False
		out = []
		q,r = 0,0
		q = state
		while not done:
			new_q = q // k
			#print(new_q)
			r = q % k
			q = new_q
			out.append(r)
			if new_q == 0:
				done = True
		while len(out) < m:
			out.append(0)

		#print(out)
		out = out[::-1]
		return out



