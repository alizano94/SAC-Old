import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DS_path = '/home/lizano/Documents/SAC/SNN/DS'
csv_DS_path = os.path.join(DS_path,'Balanced-W100-M1.csv')
tensor_save_path = os.path.join(DS_path,'DS_TransitionTensor.npy')

DS_df = pd.read_csv(csv_DS_path)
trans_matrix = np.zeros((4,3,3))

for index, rows in DS_df.iterrows():
    trans_matrix[int(rows['V'])-1,int(rows['S-1']),int(rows['S0'])] += 1

np.save(tensor_save_path,trans_matrix)

x=np.arange(3)
titles = ['Fluid','Defective','Crystal']

for i in range(4):
    for j in range(3):
        plt.subplot(4,3,3*i+j+1)
        height=trans_matrix[i,j,:]/sum(trans_matrix[i,j,:])
        plt.bar(x,height=height,color='red')
        plt.ylim([0.0,1.0])
        plt.xticks(x,['Fluid','Defective','Crystal'])
        if 3*i+j+1 > 9:
            plt.xlabel('Tansition State')
        if 3*i+j+1 < 4:
            plt.title('Initial State: '+titles[j])
        if j == 0:
            plt.ylabel('Applied Voltage: V'+str(i+1))

plt.show()