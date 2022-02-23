from src.stateRep import CNN_Testing

def test_CNN():
    test = CNN_Testing()
    test.createCNN()
    test.loadWeights(None)
    test.testCNN(None)

    return None