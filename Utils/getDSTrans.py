import os
import pandas as pd
import numpy as np

DS_path = '/home/lizano/Documents/SAC/SNN/DS'
csv_DS_path = os.path.join(DS_path,'Balanced-W100-M4.csv')

DS_df = pd.read_csv(csv_DS_path)
trans_matrix = np.zeros((4,3,3))

for index, rows in DS_df.iterrows():
    trans_matrix[int(rows['V'])-1,int(rows['S-1']),int(rows['S0'])] += 1

print(trans_matrix)

