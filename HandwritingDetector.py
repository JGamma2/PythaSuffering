import numpy as np
import torchvision.datasets as datasets
testingSet = datasets.MNIST(root='./data', train=False, download=False, transform=None)
trainingSet = datasets.MNIST(root='./data', train=True, download=False, transform=None)
print(len(testingSet))
print(len(trainingSet))
