from src.stateRep import CNN

init_state_img = '/home/lizano/Documents/SAC/data/initialstates/Crsytal_test.png'
cnn_weights_path = '/home/lizano/Documents/SAC/models/cnn/CNN.h5'

cnn = CNN()
cnn.createCNN(summary=False)
cnn.loadWeights(None)
print(cnn.runCNN(init_state_img))
