from src.stateRep import CNN
from src.dynamics import SNN

init_state_img = './data/initialstates/Crsytal_test.png'

cnn = CNN()
snn = SNN()

cnn.createCNN(summary=False)
cnn.loadWeights(None)



