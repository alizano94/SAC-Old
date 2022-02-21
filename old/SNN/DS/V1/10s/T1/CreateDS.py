import os
import pandas as pd

sep = '","'
os.system("awk '{print $1,"
	+sep+",$2,"
	+sep+",$3,"
	+sep+",$4,"
	+sep+",$5,"
	+sep+",$6,"
	+sep+",$7}' op1.txt >test.txt")
data = pd.read_csv('test.txt', header=None)
data.columns = ['Time','C6_avg','rgmean','psi6','RC','V','lambda']
data = data.drop(labels=['RC','lambda','rgmean'],axis=1)
states = pd.DataFrame(columns = ['S_cnn', 'S_param'])
for i in range(0,len(data.index)):
	s_cnn = 1
	c6 = data.iloc[i]['C6_avg']
	psi6 = data.iloc[i]['psi6']
	if c6 <= 4.0:
		s_real = 0
	elif c6 > 4.0 and psi6 < 0.9:
		s_real = 1
	else:
		s_real = 2
	states_dict = {'S_cnn':s_cnn,'S_param':s_real}
	states = states.append(states_dict,ignore_index=True)
data = pd.concat([data,states], axis=1)
data.to_csv('./test.csv',index=False)
os.system('rm -rf test.txt')
data = pd.read_csv('./test.csv')
print(data)