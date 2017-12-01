from numpy import random, array
import nn
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

TRAIN_SET_SIZE = 100
TEST_SET_SIZE = 40
EPOCHS = 100000
LEARNING_RATE = 0.01

if __name__ == "__main__":

    train_accu = []
    test_accu = []

    # Define the layers of the neural network
    for n_h in range(1,11):
        
        # Initialize random seed with a number to imporve reproducability
        random.seed(1)
        
        layer1 = nn.NeuronLayer(n_h, 2)
        layer2 = nn.NeuronLayer(1, n_h)

        # Combine the layers to create a neural network
        neural_network = nn.NeuralNetwork(layer1, layer2)

        training_data, training_labels = nn.get_dataset(TRAIN_SET_SIZE)
        neural_network.train(array(training_data), array(training_labels), EPOCHS, LEARNING_RATE)
        testing_data, testing_labels = nn.get_dataset(TEST_SET_SIZE)

        train_accu.append( nn.get_accu(neural_network, training_data, training_labels) )
        test_accu.append( nn.get_accu(neural_network, testing_data, testing_labels) )

    plt.plot([x for x in range(1,11)], train_accu)
    plt.plot([x for x in range(1,11)], test_accu, '--')
    plt.ylabel('Accuracy')
    plt.savefig('train_acc.png')
