import os
import pandas as pd
import numpy as np


from tensorflow.keras.preprocessing import image

class Helpers():
    def __init__(self):
        self.cnn_ds_path = '/home/lizano/Documents/SAC/data/raw/cnn'
        self.cnn_weights_path = '/home/lizano/Documents/SAC/models/cnn/CNN.h5'

        self.snn_ds_path = '/home/lizano/Documents/SAC/data/raw/snn'
        self.snn_weights_path = '/home/lizano/Documents/SAC/models/snn'
        self.preprocess_data_path = '/home/lizano/Documents/SAC/data/prepocessed'

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