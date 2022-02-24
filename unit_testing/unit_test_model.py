from src.stateRep import CNN_Testing
from src.dynamics import SNN_Testing

def test_CNN():
    test = CNN_Testing()
    test.createCNN()
    test.loadWeights(None)
    test.testCNN(None)

    return None

def test_SNN():
    test = SNN_Testing(w=100,m=1)
    test.createSNN()
    test.loadWeights(None)
    test.getTranitionTensorDS()
    test.testSNN()

    return None