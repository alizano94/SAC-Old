from src.stateRep import CNN
from src.dynamics import SNN

init_state_img = './data/initialstates/Crsytal_test.png'


snn = SNN(w=100,m=1)
snn.createRNN(summary=True)
snn.trainSNN()



