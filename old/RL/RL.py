import os
import numpy as np
from IPython.display import clear_output


from Utils.Helpers import *
from CNN.CNN import *
from SNN.SNN import *


class RL():

	def __init__(self):
		self.snn = SNN()
		self.helpers = Helpers()

	def get_Q_table(self,model,memory,k,a_size=4,episodes=1000):
		'''
		Learns the q table 
		'''

		s_size = k**memory
		rl_ds_dir = '/home/lizano/Documents/SAC/RL/Qtables/'
		out_file = str(s_size)+'X'+str(a_size)+'Q_table'+str(memory)+'M.npy'
		out_file = os.path.join(rl_ds_dir,out_file)
		q_table = np.random.rand(s_size,a_size)

		# Hyperparameters
		alpha = 0.1
		gamma = 0.95
		epsilon = 0.1

		goal = int(k-1)

		for i in range(0, episodes):
			state = np.random.choice(range(s_size))
			v_state = self.helpers.stateDecoder(k,state,memory)

			epochs, penalties, reward, = 0, 0, 0
			done = False

			if v_state[-1] == goal:
					done = True

			while not done:

				if np.random.uniform(0, 1) < epsilon:
					action =  np.random.choice([1,2,3,4])# Explore action space
				else:
					action = np.argmax(q_table[state]) # Exploit learned values

				inp_feat = {'V':np.array([float(action)])}
				for j in range(memory):
					name = 'S'+str(j-memory)
					state_value = v_state[j]
					inp_feat[name] = np.array([float(state_value)])

				next_state = int(self.snn.runSNN2(model,inp_feat))
				reward = next_state - goal#we dessign it  

				old_value = q_table[state, action-1]
				next_max = np.max(q_table[next_state])

				new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
				q_table[state, action-1] = new_value
				
				v_state += [next_state]
				v_state.pop(0)
				state = self.helpers.stateEncoder(k,v_state)
				epochs += 1

				if v_state[-1] == goal:
					done = True

			if i % 100 == 0:
				clear_output(wait=True)
				print(f"Episode: {i}")

		print("Training finished.\n")
		np.save(out_file,q_table)

		return q_table

	def QControl(self,q_table,state):
		'''
		Method that handles the control
		'''
		action = np.argmax(q_table[state]) +1

		return action
