import numpy as np


class NeuralNetwork:
    """Clas represent neural network"""
    
    def __init__(self, layer_dims, activations:list, derivatives:list, eta=0.5):
        
        self.layer_dims = layer_dims
        self.activations = activations
        self.derivatives = derivatives
        self.weights = [[] for _ in range(len(self.layer_dims))]
        self.layers = [[] for _ in range(len(self.layer_dims))]
        self.eta = eta
    
    def insert_data(self, x, y):
        """Function insert given data to neural network
        
        Weights matrix are randomized
        """

        self.input = x
        self.y = y
        # Initial weights matrix
        self.weights[0] = np.random.rand(self.layer_dims[0], self.input.shape[1])
        for i in range(1, len(self.layer_dims)):
            self.weights[i] = np.random.rand(self.layer_dims[i], self.layer_dims[i-1])
        # Output given by network
        #self.output = np.zeros(self.y.shape)
        print(self.layers)
        print(self.activations)
        print(self.derivatives)
        for w in self.weights:
            print(w.shape)

    def feedfoward(self):
        """Compute layers output using feed foward"""

        self.layers[0] = self.activations[0](np.dot(self.input, self.weights[0].T))
        for i in range(1, len(self.layer_dims)):
            self.layers[i] = self.activations[i](np.dot(self.layers[i-1], self.weights[i].T))

    def backprop(self):
        """Find better working weihts 
        
        first Compute weights gradient
        then use it to find possible better solution
        using gradient descent algorithm
        (backprop represent one step)
        """

        self.deltas = [[] for _ in range(len(self.layers))]
        self.d_weights = [[] for _ in range(len(self.layers))]
        self.deltas[-1] = (self.y - self.layers[-1]) * self.derivatives[-1](np.dot(self.layers[-2], self.weights[-1].T))
        self.d_weights[-1] = self.eta * np.dot(self.deltas[-1].T, self.layers[-2])
        for i in range(len(self.layers) - 2, 0, -1):
            self.deltas[i] = self.derivatives[i](np.dot(self.layers[i-1], self.weights[i].T)) * np.dot(self.deltas[i+1], self.weights[i+1])
            self.d_weights[i] = self.eta * np.dot(self.deltas[i].T, self.layers[i-1])
        self.deltas[0] = self.derivatives[0](np.dot(self.input, self.weights[0].T)) * np.dot(self.deltas[1], self.weights[1])
        self.d_weights[0] = self.eta * np.dot(self.deltas[0].T, self.input)
        for i in range(len(self.layers)):
            self.weights[i] += self.d_weights[i]

    def cost_function(self, orginal_y, min_value_y, max_value_y):
        """Function return square error cost
        
        Where min_value_y, max_value_y are param used to scale y
        we need them to scale bax neural network output
        """

        self.feedfoward()
        cost = 0
        for i in range(len(orginal_y)):
            cost += (self.layers[-1][i] * (max_value_y - min_value_y)
                + min_value_y - orginal_y[i])**2
        return 1 / len(orginal_y) * cost[0]

    def output(self):
        return self.layers[-1]
           