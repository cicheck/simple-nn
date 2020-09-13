import numpy as np


def sigmoid(x):
    """Compute sigmoid function of x"""
    return 1.0 / (1 + np.exp(-x))


def sigmoid_derivative(y):
    """Compute sigmoid derivative of x"""
    return sigmoid(y) * (1.0 - sigmoid(y))


def relu(x):
    """Compute relu function of x"""
    return np.maximum(0, x)


def relu_derivative(y):
    """Compute relu derivative of x"""

    return (y > 0) * 1


def tanh(x):
    """Compute tanh function of x"""
    return np.tanh(x)

def tanh_derivative(y):
    """Compute tanh derivative of x"""
    return 1.0 - np.tanh(y)**2