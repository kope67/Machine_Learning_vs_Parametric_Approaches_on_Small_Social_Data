import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error

# Activation linéaire (identité)
def identity(z):
    return z

# Fonction de coût : MSE
def loss_fn(params, X, y):
    n = X.shape[1]
    W = params[:n].reshape((n, 1))
    b = params[-1]
    y_hat = identity(X @ W + b)
    return np.mean((y - y_hat) ** 2)

# Entraînement par Conjugate Gradient
def train_gradient_descent(X, y, lr=0.01, epochs=1000):
    n_samples, n_features = X.shape
    W = np.zeros((n_features, 1))
    b = 0.0

    for epoch in range(epochs):
        y_hat = X @ W + b
        error = y_hat - y

        # Gradients
        dW = (2 / n_samples) * (X.T @ error)
        db = (2 / n_samples) * np.sum(error)

        # Mise à jour
        W -= lr * dW
        b -= lr * db

        # (Optionnel) Affichage de la perte
        if epoch % 100 == 0:
            mse = np.mean(error ** 2)
            print(f"Epoch {epoch} - MSE: {mse:.4f}")

    return W, b

# Prédiction : sortie continue
def predict(X, W, b):
    return identity(X @ W + b)

def main():
    # Génération de données de régression
    X, y = make_regression(n_samples=100, n_features=2, noise=10, random_state=0)
    y = y.reshape(-1, 1)

    W, b = train_gradient_descent(X, y)
    y_pred = predict(X, W, b)

    print("MSE (CG):", mean_squared_error(y, y_pred))

    # Affichage
    plt.scatter(y, y_pred, c='blue', edgecolors='k')
    plt.xlabel("Vraies valeurs")
    plt.ylabel("Valeurs prédites")
    plt.title("Régression linéaire (CG)")
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')  # diagonale
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()