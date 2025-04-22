# src/train.py

from src.model import NeuralNetwork
import src.utils as utils


def train_model(X, Y, layer_sizes, activations, loss_type='mse',
                epochs=1000, lr=0.01, verbose=True):
    """
    Trains a feedforward neural network on the given data.

    Args:
        X (ndarray): Input data of shape (input_dim, num_samples).
        Y (ndarray): Ground truth labels.
        layer_sizes (list): List specifying architecture (input to output).
        activations (list): Activation names for each hidden layer.
        loss_type (str): 'mse' or 'cross_entropy'.
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
        verbose (bool): Whether to print loss during training.

    Returns:
        model (NeuralNetwork): Trained model.
        losses (list): Recorded loss values over epochs.
    """
    model = NeuralNetwork(layer_sizes, activations)
    losses = []

    for epoch in range(epochs):
        zs, activations_list = model.forward(X)
        loss = model.compute_loss(activations_list[-1], Y, loss_type=loss_type)
        losses.append(loss)
        model.backward(zs, activations_list, Y, loss_type=loss_type, learning_rate=lr)

        if verbose and epoch % (epochs // 10) == 0:
            print(f"Epoch {epoch:4d} | Loss: {loss:.5f}")

    return model, losses


def run_xor_experiment(epochs=2000, lr=0.1):
    """
    Trains and evaluates the network on the XOR classification task.
    """
    print("\n🧠 Running XOR Classification Task")
    X, Y = utils.load_xor_dataset()
    input_dim = X.shape[0]

    layer_sizes = [input_dim, 2, 1]
    activations = ['tanh', 'sigmoid']
    model, losses = train_model(X, Y, layer_sizes, activations,
                                loss_type='mse', epochs=5000, lr=0.05)

    utils.plot_loss_curve(losses, title="XOR - Loss Curve")
    utils.plot_decision_surface(model, X, Y)
    utils.plot_3d_decision_surface(model, X, Y)


def run_regression_experiment(file_path, hidden_units=3, epochs=5000, lr=0.05):
    """
    Trains and evaluates the network on the regression task.

    Args:
        file_path (str): Path to the Excel file containing regression data.
        hidden_units (int): Number of hidden layer units.
    """
    print(f"\n📈 Running Regression Task with {hidden_units} hidden units")
    X, Y = utils.load_regression_dataset(file_path)
    input_dim = X.shape[0]

    layer_sizes = [input_dim, hidden_units, 1]
    activations = ['tanh']
    model, losses = train_model(X, Y, layer_sizes, activations,
                                loss_type='mse', epochs=5000, lr=0.05)

    Y_pred = model.predict(X)
    utils.plot_loss_curve(losses, title=f"Regression - {hidden_units} Units Loss")
    utils.plot_regression_fit(X, Y, model, title=f"Regression - {hidden_units} Units")


if __name__ == "__main__":
    run_xor_experiment()

    regression_path = "../data/Proj5Dataset.xlsx"

    run_regression_experiment(regression_path, hidden_units=3)
    run_regression_experiment(regression_path, hidden_units=20)
