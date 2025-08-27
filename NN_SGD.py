# NN_SGD.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score

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
    # éviter log(0)
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
    # Génération de données fictives
    X, y = make_blobs(n_samples=200, centers=2, n_features=2, random_state=42)
    y = y.reshape(-1, 1)

    W, b, losses = train_SGD(X, y, learning_rate=0.1, n_iter=50)
    y_pred = predict(X, W, b)

    print("Accuracy:", accuracy_score(y, y_pred))

    # Affichage de la courbe de perte
    plt.plot(losses)
    plt.xlabel("Époques")
    plt.ylabel("Log Loss")
    plt.title("Évolution de la perte avec SGD")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()