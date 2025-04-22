# src/utils.py

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

def load_xor_dataset():
    """
    Loads a manually defined modified XOR dataset.

    Returns:
        X (ndarray): Input features, shape (2, 4)
        Y (ndarray): Binary class labels, shape (1, 4)
    """
    X = np.array([
        [-1, -1, 1, 1],
        [-1,  1, -1, 1]
    ])
    Y = np.array([
        [1, 0, 0, 1]
    ])
    return X, Y


def load_regression_dataset(path):
    """
    Loads and formats the regression dataset from Excel.

    Args:
        path (str): File path to the Excel dataset.

    Returns:
        X (ndarray): Input features, shape (1, N)
        Y (ndarray): Target values, shape (1, N)
    """
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip()
    x_col = df.columns[0]
    y_col = df.columns[1]
    X = df[x_col].values.reshape(1, -1)
    Y = df[y_col].values.reshape(1, -1)
    return X, Y


def plot_loss_curve(losses, title="Loss Curve"):
    """
    Plots the loss value over training epochs.

    Args:
        losses (list of float): Loss values at each epoch.
        title (str): Title of the plot.
    """
    plt.figure()
    plt.plot(losses, label='Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_decision_surface(model, X, Y):
    """
    Plots the decision surface for 2D classification (XOR problem).

    Args:
        model (NeuralNetwork): Trained model.
        X (ndarray): Input features (2, N).
        Y (ndarray): Ground truth labels (1, N).
    """
    h = 0.01
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()].T
    Z = model.predict(grid)
    Z = Z.reshape(xx.shape)

    plt.figure()
    plt.contourf(xx, yy, Z, levels=50, cmap='coolwarm', alpha=0.6)
    scatter = plt.scatter(X[0, :], X[1, :], c=Y.flatten(), cmap='coolwarm', edgecolors='k')
    plt.title("Decision Surface")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.colorbar(scatter)
    plt.show()

def plot_3d_decision_surface(model, X, Y):
    """
    Plots a 3D surface for the decision boundary of the model.

    Args:
        model (NeuralNetwork): Trained model.
        X (ndarray): Input features (2, N).
        Y (ndarray): Labels (1, N).
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    h = 0.05
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    grid = np.c_[xx.ravel(), yy.ravel()].T
    Z = model.predict(grid).reshape(xx.shape)

    ax.plot_surface(xx, yy, Z, cmap='coolwarm', alpha=0.8)
    ax.scatter(X[0, :], X[1, :], Y.flatten(), c=Y.flatten(), cmap='coolwarm', s=50, edgecolor='k')

    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Output')
    ax.set_title("3D Decision Surface - XOR")
    plt.show()


def plot_regression_fit(X, Y_true, model, title="Regression Fit"):
    """
    Plots the smooth model prediction curve against ground truth.

    Args:
        X (ndarray): Input features, shape (1, N)
        Y_true (ndarray): Ground truth targets, shape (1, N)
        model (NeuralNetwork): Trained model
        title (str): Plot title
    """
    # Sort X and create smooth X range
    X_sorted = np.sort(X.flatten())
    X_dense = np.linspace(X_sorted[0], X_sorted[-1], 500).reshape(1, -1)
    Y_dense = model.predict(X_dense)

    plt.figure()
    plt.scatter(X.flatten(), Y_true.flatten(), color='blue', label='Ground Truth')
    plt.plot(X_dense.flatten(), Y_dense.flatten(), color='red', label='Model Prediction')
    mse = np.mean((model.predict(X) - Y_true) ** 2)
    plt.title(f"{title} (MSE: {mse:.4f})")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.show()

