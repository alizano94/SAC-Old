import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.helpers import SNN_Helpers
from src.stateRep import CNN

class SNN(SNN_Helpers, CNN):
    def __init__(self,w,m,*args,**kwargs):
        super(SNN,self).__init__(*args,**kwargs)
        self.cnn = CNN()
        self.cnn.createCNN(summary=False)
        self.cnn.loadWeights(None)

        self.ds_path = './data/raw/snn'
        self.weights_path = './models/snn'
        self.m = m
        self.w = w

    def preProcessSNNDS(self):
        '''
        Method that takes op1.txt and plots and creates 
        csv containing information about time, order params,
        real state, cnnn state and voltage.
        '''

        sep = '","'
        for v_dir in os.listdir(self.ds_path):
            v_path = os.path.join(self.ds_path,v_dir)
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
                                        s_cnn, _ = self.runCNN(file_name)
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
                                    os.chdir(self.ds_path)
        
        