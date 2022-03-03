import os
import numpy as np
from IPython.display import clear_output

from src.dynamics import Control_Asistance

class RL(Control_Asistance):
    def __init__(self,*args,**kwargs):
        super(RL,self).__init__(*args,**kwargs)

        self.s_size = self.k**self.m
        self.out_file = str(self.s_size)+'X'+str(self.a)+'Q_table'+str(self.m)+'M.npy'
        self.out_file = os.path.join(self.contorl_policies,self.out_file)
        self.q_table = np.load(self.out_file)


    def getQTable(self,alpha=0.1,gamma=0.95,epsilon=0.1,eps=10000):
        '''
        Learns the q table 
        '''
        self.q_table = np.random.rand(self.s_size,self.a)
        goal = int(self.k-1)
        actions = np.arange(self.a)

        for i in range(0, eps):
            state = np.random.choice(range(self.s_size))
            v_state = self.stateDecoder(self.k,state,self.m)

            epochs, penalties, reward, = 0, 0, 0
            done = False

            if v_state[-1] == goal:
                    done = True

            while not done:

                if np.random.uniform(0, 1) < epsilon:
                    action =  np.random.choice(actions)# Explore action space
                else:
                    action = np.argmax(self.q_table[state]) # Exploit learned values

                next_state = int(self.runSNN(action,v_state))
                reward = next_state - goal #we dessign it  

                old_value = self.q_table[state, action-1]
                next_max = np.max(self.q_table[next_state])

                new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
                self.q_table[state, action-1] = new_value
                
                v_state += [next_state]
                v_state.pop(0)
                state = self.stateEncoder(self.k,v_state)
                epochs += 1

                if v_state[-1] == goal:
                    done = True

            if i % 0.001*eps == 0:
                clear_output(wait=True)
                print(f"Episode: {i}")

        print("Training finished.\n")
        np.save(self.out_file,self.q_table)

    def QControl(self,states):
        '''
        Method that handles the control
        '''
        if len(states) < self.m:
            while len(states) < self.m:
                states.insert(0,states[0])
        elif len(states) > self.m:
            while len(states) > self.m:
                states.pop(0)
        else:
            pass

        state = self.stateEncoder(self.k,states)
        action = np.argmax(self.q_table[state]) +1

        return action


