from numpy import exp, array, random, dot
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

class NeuronLayer():
    def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
        self.weights = 2 * random.random((number_of_inputs_per_neuron, number_of_neurons)) - 1

class NeuralNetwork():
    def __init__(self, layer1, layer2):
        self.layer1 = layer1
        self.layer2 = layer2

    def sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations, learning_rate):
        for iteration in xrange(number_of_training_iterations):
            # Pass the training set through our neural network
            output_from_layer_1, output_from_layer_2 = self.forward(training_set_inputs)

            # Calculate loss
            layer2_error = training_set_outputs - output_from_layer_2
            layer2_delta = layer2_error * self.sigmoid_derivative(output_from_layer_2)

            # Calculate the error for layer 1
            layer1_error = layer2_delta.dot(self.layer2.weights.T)
            layer1_delta = layer1_error * self.sigmoid_derivative(output_from_layer_1)

            # Calculate how much to adjust the weights by
            layer1_adjustment = training_set_inputs.T.dot(layer1_delta)
            layer2_adjustment = output_from_layer_1.T.dot(layer2_delta)

            # Adjust the weights.
            self.layer1.weights += learning_rate*layer1_adjustment
            self.layer2.weights += learning_rate*layer2_adjustment

    def forward(self, inputs):
        output_from_layer1 = self.sigmoid(dot(inputs, self.layer1.weights))
        output_from_layer2 = self.sigmoid(dot(output_from_layer1, self.layer2.weights))
        return output_from_layer1, output_from_layer2

""" Code to generate data """
MEAN1 = [0,0]
COV1 = [[1,0],[0,1]]
MEAN2 = [1,5]
COV2 = [[3,1],[1,2]]
TRAIN_SET_SIZE = 100
TEST_SET_SIZE = 40
LEARNING_RATE = 0.01
EPOCHS = 100000

def get_dataset(size):
    dataset = []
    dataset_labels = []

    x1,y1 = random.multivariate_normal(MEAN1, COV1, size/2).T
    x2,y2 = random.multivariate_normal(MEAN2, COV2, size/2).T

    for i in range(size/2):
        dataset.append([x1[i],y1[i]])
        dataset_labels.append([0])
        dataset.append([x2[i],y2[i]])
        dataset_labels.append([1])

    return dataset, dataset_labels

""" Function to obatin accuracy of a nn on a dataset """
def get_accu(nn, dataset, dataset_labels):
    accu = 0
    for i in range(len(dataset)):
        hidden_state, output = nn.forward(dataset[i])
        if abs(output[0]*2 - dataset_labels[i][0]) < 0.5:
            accu += 1
    return accu*100.0/len(dataset)

if __name__ == "__main__":


    train_accu = []
    test_accu = []

    # Define the layers of the neural network
    n_h = 1
        
    # Initialize random seed with a number to imporve reproducability
    random.seed(1)
    
    layer1 = NeuronLayer(n_h, 2)
    layer2 = NeuronLayer(1, n_h)

    # Combine the layers to create a neural network
    neural_network = NeuralNetwork(layer1, layer2)

    training_data, training_labels = get_dataset(TRAIN_SET_SIZE)
    testing_data, testing_labels = get_dataset(TEST_SET_SIZE)
    for i in range(EPOCHS):
        neural_network.train(array(training_data), array(training_labels), 1, LEARNING_RATE)
        train_accu.append( 100-get_accu(neural_network, training_data, training_labels) )
        test_accu.append( 100-get_accu(neural_network, testing_data, testing_labels) )

    plt.plot([x for x in range(1,EPOCHS+1)], train_accu)
    plt.plot([x for x in range(1,EPOCHS+1)], test_accu, '--')
    plt.ylabel('Accuracy')
    plt.savefig('epoch_acc.png')
