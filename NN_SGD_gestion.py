# NN_SGD_CSV.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

def initialize(n_features):
    W = np.random.randn(n_features, 1) * 0.01
    b = 0.
    return W, b

def model(x, W, b):
    return sigmoid(np.dot(x, W) + b)

def compute_loss(y_hat, y):
    m = len(y)
    eps = 1e-15
    y_hat = np.clip(y_hat, eps, 1 - eps)
    return -1/m * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

def gradients(xi, yi, W, b):
    z = np.dot(xi, W) + b
    a = sigmoid(z)
    dz = a - yi
    dW = xi.T * dz
    db = dz
    return dW, db

def update(W, b, dW, db, learning_rate):
    W = W - learning_rate * dW
    b = b - learning_rate * db
    return W, b

def predict(x, W, b):
    y_prob = model(x, W, b)
    return (y_prob >= 0.5).astype(int)

def train_SGD(x, y, learning_rate=0.1, n_iter=100):
    m, n = x.shape
    W, b = initialize(n)
    losses = []

    for epoch in range(n_iter):
        loss_epoch = 0
        for i in range(m):
            xi = x[i].reshape(1, -1)
            yi = y[i].reshape(1, 1)
            y_hat = model(xi, W, b)
            loss_epoch += compute_loss(y_hat, yi)
            dW, db = gradients(xi, yi, W, b)
            W, b = update(W, b, dW, db, learning_rate)
        losses.append(loss_epoch / m)

    return W, b, losses

def main():
    # Charger le fichier CSV
    df = pd.read_csv("chemin/vers/ton_fichier.csv")  # ⬅️ adapte ce chemin
    y_col = "cible"  # ⬅️ adapte si la colonne cible a un autre nom

    X = df.drop(columns=[y_col]).values
    y = df[y_col].values.reshape(-1, 1)

    # Standardisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Séparer en train/test
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    # Entraînement
    W, b, losses = train_SGD(X_train, y_train, learning_rate=0.1, n_iter=100)

    # Prédiction
    y_pred_train = predict(X_train, W, b)
    y_pred_test = predict(X_test, W, b)

    # Évaluation
    print("Accuracy entraînement:", accuracy_score(y_train, y_pred_train))
    print("Accuracy test:", accuracy_score(y_test, y_pred_test))

    # Affichage des pertes
    plt.plot(losses)
    plt.xlabel("Époques")
    plt.ylabel("Log Loss")
    plt.title("Évolution de la perte avec SGD")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()