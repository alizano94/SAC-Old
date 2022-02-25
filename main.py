from src.control import RL


control = RL(w=100,m=1,number_actions=4)
control.createCNN()
control.createSNNDS()
control.createSNN()
control.trainSNN(plot=True)

from unit_testing.unit_test_model import test_SNN
test_SNN()