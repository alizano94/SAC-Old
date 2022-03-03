from unit_testing.unit_test_model import test_CNN, test_SNN, test_SNNTrajectories


initial_image = '/home/lizano/Documents/SAC/data/initialstates/Fluid_test.png'
#
#test_CNN()
test_SNNTrajectories(initial_image,N=10,l=10)