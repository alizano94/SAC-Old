from tkinter import NONE
from src.stateRep import CNN_Testing
from src.dynamics import SNN_Testing

def test_CNN():
    test = CNN_Testing()
    test.createCNN()
    test.loadCNN(None)
    test.testCNN(None)

    return None

def test_SNN():
    test = SNN_Testing(a=4,w=100,m=1)
    test.createSNN()
    test.loadSNN(None)
    test.getTranitionTensorDS()
    test.testSNN()

    return None

def test_SNNTrajectories(initial_image,N,l):
    test = SNN_Testing(a=4,w=100,m=1)
    test.createCNN()
    test.loadCNN(None)
    test.createSNN()
    test.loadSNN(None)
    test.getTrajectoryHistogran(initial_image,N,l)
    test.getTrajectories(initial_image,l)

    return None
