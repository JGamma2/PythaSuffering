import numpy as np
import torchvision.datasets as datasets
import random
testingSet = datasets.MNIST(root='./data', train=False, download=False, transform=None)
trainingSet = datasets.MNIST(root='./data', train=True, download=False, transform=None)
print(testingSet)
print(trainingSet)
print(testingSet[0])
print(trainingSet[0][0])
pixelValues = list(trainingSet[0][0].getdata())
print(pixelValues)
print(len(pixelValues))
print('\n\n')

#Apologies for the verbosity, but this will work. It's not going to scale, sorry about that, but I just need the network to exist.
#Should return an array with all the information of the network.
#784 input nodes, 2 layers of 12 hidden nodes, and 10 output nodes.
def initialize4LayerNetwork():
    #network is the full network array. This will be returned at the end.
    network = []
    #network[0] will contain 784 instances of the number 0.
    #These will be populated with values later, but they are currently needed to be extant.
    network.append([])
    for i in range(784):
        network[0].append(0)
    #network[1] will be the weights of the neurons between layers 1 and 2.
    #This array will have 12 arrays, which will contain 784 floats. These will be the weights for each neuron.
    #These will all be initialized to a random float between 0 and 1.
    network.append([])
    for i in range(12):
        network[1].append([])
        for j in range(784):
            network[1][i].append(random.random())
    #network[2] will be the biases of the 12 nodes in layer 2.
    #This will also be a random float between 0 and 1.
    network.append([])
    for i in range(12):
        network[2].append(random.random())
    #network[3] will be the weights of the neurons in between layers 2 and 3.
    network.append([])
    for i in range(12):
        network[3].append([])
        for j in range(12):
            network[3][i].append(random.random())
    #network[4] will be the biases of the layer 3 nodes.
    network.append([])
    for i in range(12):
        network[4].append(random.random())
    #network[5] will be the weights of the neurons in between layers 3 and 4.
    network.append([])
    for i in range(10):
        network[5].append([])
        for j in range(12):
            network[5][i].append(random.random())
    #network[6] will be the biases of the layer 4 nodes.
    network.append([])
    for i in range(10):
        network[6].append(random.random())

    return network

#def loadNetwork:
    #Here's the thing. I don't think we can just train it all in one sitting.
    #We need to be able to save a network state so that we can use it later, already trained.
    #Like to a file, that we can access later.

network = initialize4LayerNetwork()
print(network)