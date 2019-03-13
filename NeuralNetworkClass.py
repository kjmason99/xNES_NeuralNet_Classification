# This simple code creates a neural network and makes predictions based on inputs passed into the network



import numpy as np
import random
from math import exp

class NeuralNetwork:
    
    def __init__(self, n_input=None, n_output=None, n_hidden_nodes=None):
        self.n_input = n_input  
        self.n_output = n_output  
        self.n_hidden_nodes = n_hidden_nodes  
        self.network = self._createNN()

    
    
    # Get outputs y for inputs X
    
    def predict(self, X):
        y_predict = np.zeros(len(X), dtype=np.int)
        for i, x in enumerate(X):
            output = self._forward(x)  
            y_predict[i] = np.argmax(output)  

        return y_predict

    
    def _createNN(self):
        
        
        def _build_layer(n_input, n_output):
            layer = list()
            for out in range(n_output): 
                weights = list()
                for idx_in in range(n_input):
                    weights.append(random.random())
                layer.append({"weights": weights,
                              "output": None,
                              "delta": None})
            return layer

        
        n_hidden_layers = len(self.n_hidden_nodes)
        network = list()
        if n_hidden_layers == 0:
            network.append(_build_layer(self.n_input, self.n_output))
        else:
            network.append(_build_layer(self.n_input, self.n_hidden_nodes[0]))
            for i in range(1,n_hidden_layers):
                network.append(_build_layer(self.n_hidden_nodes[i-1],
                                            self.n_hidden_nodes[i]))
            network.append(_build_layer(self.n_hidden_nodes[n_hidden_layers-1],
                                        self.n_output))

        return network

    
    # Forward pass 
    def _forward(self, x):
        
        def activate(weights, inputs):
            activation = 0.0
            for i in range(len(weights)):
                activation += weights[i] * inputs[i]
            return activation

        
        input = x
        for layer in self.network:
            output = list()
            for node in layer:
                activation = activate(node['weights'], input)
                node['output'] = self._sigmoid(activation)
                output.append(node['output'])
            input = output

        return input

    

    def getWeights(self):
        weightsList = array([])
        for i_layer, layer in enumerate(self.network):
            if i_layer == 0:
                inputs = np.zeros(self.n_input)
            else:
                inputs = np.zeros(len(self.network[i_layer - 1]))
            
            for node in layer:
                for j, input in enumerate(inputs):
                    w = node['weights'][j]
                    weightsList.append(w)
        print ('weights : ',weightsList)
        return weightsList

    def setWeights(self, w):
        count = 0
        for i_layer, layer in enumerate(self.network):
            if i_layer == 0:
                inputs = np.zeros(self.n_input)
            else:
                inputs = np.zeros(len(self.network[i_layer - 1]))
            for node in layer:
                for j, input in enumerate(inputs):
                    node['weights'][j] = float(w[count])
                    count = count+1

    def countWeights(self):
        count = 0
        for i_layer, layer in enumerate(self.network):
            if i_layer == 0:
                inputs = np.zeros(self.n_input)
            else:
                inputs = np.zeros(len(self.network[i_layer - 1]))
            for node in layer:
                for j, input in enumerate(inputs):
                    count = count+1
        return count
    
    def _sigmoid(self, x):
        return 1.0/(1.0+exp(-x))

























