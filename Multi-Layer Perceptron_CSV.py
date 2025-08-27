import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# --- Fonctions de base ---
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

def initialize_parameters(n_input, n_hidden, n_output):
    np.random.seed(0)
    W1 = np.random.randn(n_input, n_hidden) * 0.1
    b1 = np.zeros((1, n_hidden))
    W2 = np.random.randn(n_hidden, n_output) * 0.1
    b2 = np.zeros((1, n_output))
    return W1, b1, W2, b2

def train_nn(X, y, n_hidden=10, lr=0.01, epochs=1000):
    n_samples, n_features = X.shape
    n_output = 1

    W1, b1, W2, b2 = initialize_parameters(n_features, n_hidden, n_output)
    losses = []
    for epoch in range(epochs):
        Z1 = X @ W1 + b1
        A1 = tanh(Z1)
        Z2 = A1 @ W2 + b2
        y_hat = Z2

        loss = np.mean((y_hat - y) ** 2)
        losses.append((epoch, loss))

        dZ2 = 2 * (y_hat - y) / n_samples
        dW2 = A1.T @ dZ2
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dA1 = dZ2 @ W2.T
        dZ1 = dA1 * tanh_derivative(Z1)
        dW1 = X.T @ dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

        if epoch % 100 == 0:
            print(f"Epoch {epoch} - MSE: {loss:.4f}")

    loss_df = pd.DataFrame(losses, columns=["epoch", "loss"])
    return W1, b1, W2, b2, loss_df

def predict(X, W1, b1, W2, b2):
    A1 = tanh(X @ W1 + b1)
    y_hat = A1 @ W2 + b2
    return y_hat

# --- Programme principal ---
def main():
    # 1. Chargement des données
    df = pd.read_csv("data/caractéristique numérique_candidats2.csv",header=0,encoding="utf-8",sep=";")

    # 2. Séparer X (toutes les colonnes sauf la cible) et y (colonne cible)
    X = df.drop(columns=["20. Nombre moyen de projets "]).values
    y = df["20. Nombre moyen de projets "].values.reshape(-1, 1)

    # 3. Standardisation (important pour les réseaux)
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)

    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y)

    # 4. Entraînement du réseau
    W1, b1, W2, b2, loss_df = train_nn(X_scaled, y_scaled, n_hidden=100, lr=0.01, epochs=300000)


    # Affichage de la courbe MSE (loss) en fonction de l'epoch
    plt.figure()
    plt.plot(loss_df["epoch"], loss_df["loss"])
    plt.xlabel("Epoch")
    plt.ylabel("MSE (loss)")
    plt.title("Loss curve (MSE) as a function of epoch")
    plt.grid(True)
    plt.show()


    # 5. Prédictions et retour à l’échelle d’origine
    y_pred_scaled = predict(X_scaled, W1, b1, W2, b2)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    # 6. Évaluation
    print("MSE réel :", mean_squared_error(y, y_pred))

    # 7. Affichage
    plt.scatter(y, y_pred, c='red', edgecolors='k')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
    plt.xlabel("Valeurs réelles")
    plt.ylabel("Prédictions")
    plt.title("Réseau de neurones sur données réelles")
    plt.grid(True)
    plt.show()

    # 8. Affichage du DataFrame des pertes
    print("\nDataFrame epoch*loss :")
    print(loss_df.head())

if __name__ == "__main__":
    main()