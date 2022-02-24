import os
import pandas as pd
import numpy as np
from random import seed, randint


from tensorflow.keras.preprocessing import image

class Helpers():
    def __init__(self):

        self.cnn_results_path = '/home/lizano/Documents/SAC/results/cnn'
        self.cnn_ds_path = '/home/lizano/Documents/SAC/data/raw/cnn'
        self.cnn_weights_path = '/home/lizano/Documents/SAC/models/cnn/CNN.h5'

        self.snn_results_path = '/home/lizano/Documents/SAC/results/snn'
        self.snn_ds_path = '/home/lizano/Documents/SAC/data/raw/snn'
        self.snn_weights_path = '/home/lizano/Documents/SAC/models/snn'
        self.snn_preprocess_data_path = '/home/lizano/Documents/SAC/data/preprocessed/snn'

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

    def dropBiasData(self,data):
        '''
        Resamples the data to ensure theres no BIAS on 
        ouput state dsitribution.
        '''
        seed(1)

        hist = self.getHist(data)

        min_hist = min(hist)

        while max(hist) != min_hist:
            index = randint(0,len(data)-1)
            hist_index = int(data['S0'][index])
            if hist[hist_index] > min_hist:
                data.drop(index=index, inplace=True)
            hist = self.getHist(data)
            data.reset_index(inplace=True)
            data.drop(columns=['index'],inplace=True)

        return data

    def getHist(self,data):
        '''
        create histogram from labels data
        '''
        hist = [0,0,0]

        for _, rows in data.iterrows():
            hist[int(rows['S0'])] += 1
        print(hist)

        return hist