# src/model.py

import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


activation_functions = {
    "sigmoid": (sigmoid, sigmoid_derivative),
    "tanh": (tanh, tanh_derivative),
    "relu": (relu, relu_derivative),
}


class NeuralNetwork:
    """
    A customizable feedforward neural network supporting any number of hidden layers and units.

    Attributes:
        layers (list): Number of units in each layer (including input and output layers).
        activations (list): Activation function names for each hidden layer.
        weights (list): Weight matrices for each layer.
        biases (list): Bias vectors for each layer.
    """

    def __init__(self, layer_sizes, activations):
        """
        Initializes the neural network.

        Args:
            layer_sizes (list of int): Number of units in each layer (input → hidden(s) → output).
            activations (list of str): Activation names for each hidden layer (e.g., ['tanh', 'relu']).
        """
        self.layers = layer_sizes
        self.activations = activations
        self.weights = []
        self.biases = []

        for i in range(len(layer_sizes) - 1):
            # He or Xavier init could be used here
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            # weight = np.random.randn(layer_sizes[i + 1], layer_sizes[i]) * np.sqrt(2. / layer_sizes[i])
            weight = np.random.randn(fan_out, fan_in) * np.sqrt(1. / fan_in)
            
            bias = np.zeros((layer_sizes[i + 1], 1))
            self.weights.append(weight)
            self.biases.append(bias)

    def forward(self, X):
        """
        Performs a forward pass through the network.

        Args:
            X (ndarray): Input data of shape (input_dim, num_samples).

        Returns:
            tuple: List of pre-activations (z) and activations (a) for each layer.
        """
        a = X
        activations = [a]
        zs = []

        for i in range(len(self.weights)):
            z = self.weights[i] @ a + self.biases[i]
            zs.append(z)
            if i < len(self.activations):
                func = activation_functions[self.activations[i]][0]
                a = func(z)
            else:
                a = z  # This should NOT happen if output layer has an activation
            activations.append(a)

        return zs, activations

    def compute_loss(self, Y_pred, Y_true, loss_type='cross_entropy'):
        """
        Computes the loss between predictions and ground truth.

        Args:
            Y_pred (ndarray): Predictions from the network.
            Y_true (ndarray): Ground truth labels.
            loss_type (str): 'mse' or 'cross_entropy'.

        Returns:
            float: Computed loss value.
        """
        if loss_type == 'mse':
            return np.mean((Y_pred - Y_true) ** 2)
        elif loss_type == 'cross_entropy':
            eps = 1e-8
            return -np.mean(Y_true * np.log(Y_pred + eps) + (1 - Y_true) * np.log(1 - Y_pred + eps))
        else:
            raise ValueError("Unsupported loss type")

    def backward(self, zs, activations, Y_true, loss_type='cross_entropy', learning_rate=0.01):
        """
        Performs backpropagation and updates weights and biases.

        Args:
            zs (list): List of pre-activations from forward pass.
            activations (list): List of activations from forward pass.
            Y_true (ndarray): Ground truth labels.
            loss_type (str): 'mse' or 'cross_entropy'.
            learning_rate (float): Learning rate for gradient descent.
        """
        grads_w = [0] * len(self.weights)
        grads_b = [0] * len(self.biases)

        # Output layer delta
        delta = activations[-1] - Y_true
        if loss_type == 'mse':
            pass  # Already the derivative of MSE

        for i in reversed(range(len(self.weights))):
            a_prev = activations[i]
            grads_w[i] = delta @ a_prev.T / Y_true.shape[1]
            grads_b[i] = np.mean(delta, axis=1, keepdims=True)

            if i > 0:
                deriv = activation_functions[self.activations[i - 1]][1]
                delta = (self.weights[i].T @ delta) * deriv(zs[i - 1])

        # Update
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * grads_w[i]
            self.biases[i] -= learning_rate * grads_b[i]

    def predict(self, X):
        """
        Performs inference and returns the output.

        Args:
            X (ndarray): Input data.

        Returns:
            ndarray: Output predictions.
        """
        _, activations = self.forward(X)
        return activations[-1]
