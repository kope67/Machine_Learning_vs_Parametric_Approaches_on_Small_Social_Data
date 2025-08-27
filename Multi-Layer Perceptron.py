import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error

# Fonctions d'activation
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

# Initialisation
def initialize_parameters(n_input, n_hidden, n_output):
    np.random.seed(0)
    W1 = np.random.randn(n_input, n_hidden) * 0.1
    b1 = np.zeros((1, n_hidden))
    W2 = np.random.randn(n_hidden, n_output) * 0.1
    b2 = np.zeros((1, n_output))
    return W1, b1, W2, b2

# Entraînement
def train_nn(X, y, n_hidden=10, lr=0.01, epochs=1000):
    n_samples, n_features = X.shape
    n_output = 1

    # Initialisation
    W1, b1, W2, b2 = initialize_parameters(n_features, n_hidden, n_output)

    for epoch in range(epochs):
        # FORWARD
        Z1 = X @ W1 + b1           # (n_samples, n_hidden)
        A1 = tanh(Z1)
        Z2 = A1 @ W2 + b2          # (n_samples, 1)
        y_hat = Z2

        # Perte
        loss = np.mean((y_hat - y) ** 2)

        # BACKWARD
        dZ2 = 2 * (y_hat - y) / n_samples     # (n_samples, 1)
        dW2 = A1.T @ dZ2                      # (n_hidden, 1)
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dA1 = dZ2 @ W2.T                      # (n_samples, n_hidden)
        dZ1 = dA1 * tanh_derivative(Z1)       # (n_samples, n_hidden)
        dW1 = X.T @ dZ1                       # (n_features, n_hidden)
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        # Mise à jour des poids
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

        if epoch % 100 == 0:
            print(f"Epoch {epoch} - MSE: {loss:.4f}")

    return W1, b1, W2, b2

# Prédiction
def predict(X, W1, b1, W2, b2):
    A1 = tanh(X @ W1 + b1)
    y_hat = A1 @ W2 + b2
    return y_hat

# MAIN
def main():
    # Données de régression
    X, y = make_regression(n_samples=100, n_features=1, noise=15, random_state=0)
    y = y.reshape(-1, 1)

    # Entraînement
    W1, b1, W2, b2 = train_nn(X, y, n_hidden=10, lr=0.01, epochs=2000)

    # Prédiction
    y_pred = predict(X, W1, b1, W2, b2)

    # Évaluation
    print("MSE final:", mean_squared_error(y, y_pred))

    # Visualisation
    plt.scatter(X, y, label="Données réelles", color='blue')
    plt.scatter(X, y_pred, label="Prédictions", color='red', s=10)
    plt.legend()
    plt.title("Réseau à deux couches (régression)")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()