# NN_MiniBatchGD_VariableLR.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

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

def gradients(X, y, y_hat):
    m = len(y)
    dz = y_hat - y
    dW = (1/m) * np.dot(X.T, dz)
    db = (1/m) * np.sum(dz)
    return dW, db

def update(W, b, dW, db, learning_rate):
    W = W - learning_rate * dW
    b = b - learning_rate * db
    return W, b

def predict(x, W, b):
    y_prob = model(x, W, b)
    return (y_prob >= 0.5).astype(int)

def create_batches(X, y, batch_size):
    m = X.shape[0]
    indices = np.random.permutation(m)
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    for i in range(0, m, batch_size):
        yield X_shuffled[i:i+batch_size], y_shuffled[i:i+batch_size]

def train_minibatchGD(X, y, base_lr=0.1, n_iter=100, batch_size=32, method='fixed'):
    m, n = X.shape
    W, b = initialize(n)
    losses = []

    for epoch in range(n_iter):
        batch_losses = []

        # 🔁 Appliquer taux variable ou fixe
        if method == 'variable':
            learning_rate = base_lr / (1 + 0.05 * epoch)  # exemple : learning rate diminue
        else:
            learning_rate = base_lr

        for X_batch, y_batch in create_batches(X, y, batch_size):
            y_hat = model(X_batch, W, b)
            loss = compute_loss(y_hat, y_batch)
            dW, db = gradients(X_batch, y_batch, y_hat)
            W, b = update(W, b, dW, db, learning_rate)
            batch_losses.append(loss)

        losses.append(np.mean(batch_losses))

    return W, b, losses

def main():
    X, y = make_blobs(n_samples=300, centers=2, n_features=2, random_state=42)
    y = y.reshape(-1, 1)

    method = 'variable'  # 'fixed' ou 'variable'

    W, b, losses = train_minibatchGD(X, y, base_lr=0.1, n_iter=100, batch_size=32, method=method)
    y_pred = predict(X, W, b)

    print(f"Accuracy ({method} learning rate):", accuracy_score(y, y_pred))

    plt.plot(losses)
    plt.xlabel("Époques")
    plt.ylabel("Log Loss")
    plt.title(f"Mini-Batch GD ({method} learning rate)")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()