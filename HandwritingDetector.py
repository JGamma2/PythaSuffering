import numpy as np
import torchvision.datasets as datasets
testingSet = datasets.MNIST(root='./data', train=False, download=False, transform=None)
trainingSet = datasets.MNIST(root='./data', train=True, download=False, transform=None)
print(testingSet)
print(trainingSet)
print(testingSet[0])
print(trainingSet[0][0])
pixelValues = list(trainingSet[0][0].getdata())
print(pixelValues)
print(len(pixelValues))