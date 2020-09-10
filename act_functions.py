import numpy as np

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def sigmoid_derivative(y):
    """Compute sigmoid derivative of x"""

    return sigmoid(y) * (1.0 - sigmoid(y))

def relu(x):
    # print("x:", x)
    # print("relu:", np.maximum(0, x))
    return np.maximum(0, x)

def relu_derivative(y):
    """Compute relu derivative of x"""

    # print("y:", y)
    # y[y<=0] = 0.01
    # y[y>0] = 0.99
    # print("der:", y)
    # return y
    return (y > 0) * 1

def tanh(x):
    return np.tanh(x)

def tanh_derivative(y):
    """Compute tanh derivative of x"""

    return 1.0 - np.tanh(y)**2