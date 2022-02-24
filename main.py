from src.stateRep import CNN
from src.dynamics import SNN

init_state_img = './data/initialstates/Crsytal_test.png'

cnn = CNN()
cnn.createCNN()
cnn.loadWeights(None)

snn = SNN(w=100,m=1)
snn.createSNN()
snn.loadWeights(None)

initial_state, _ = cnn.runCNN(init_state_img)
print(initial_state)
print(int(snn.runSNN(4,[initial_state])))