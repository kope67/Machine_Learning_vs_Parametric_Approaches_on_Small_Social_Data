# -*- coding: utf-8 -*-

from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Génération des données
X, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=0)
y = y.reshape((y.shape[0], 1))

print('dimension de X:', X.shape)
print('dimension de y:', y.shape)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='summer')
plt.title("Nuage initial des données")
plt.show()

# Initialisation aléatoire des poids
def initialisation(X):
    W = np.random.randn(X.shape[1], 1)
    b = np.random.randn(1)
    return (W, b)

# Modèle (sigmoïde)
def model(X, W, b):
    z = X.dot(W) + b
    A = 1 / (1 + np.exp(-z))
    return A

# Fonction de coût log-loss
def log_loss(A, y):
    eps = 1e-15  # Pour éviter log(0)
    A = np.clip(A, eps, 1 - eps)
    return 1 / len(y) * np.sum(-y * np.log(A) - (1 - y) * np.log(1 - A))

# Calcul du gradient
def gradients(A, X, y):
    dw = 1 / len(y) * np.dot(X.T, A - y)
    db = 1 / len(y) * np.sum(A - y)
    return (dw, db)

# Mise à jour des paramètres
def update(dw, db, W, b, learning_rate):
    W = W - learning_rate * dw
    b = b - learning_rate * db
    return (W, b)

# Prédiction binaire
def predict(X, W, b):
    A = model(X, W, b)
    return A >= 0.5

# 🔍 Tracer la frontière de décision
def plot_decision_boundary(X, y, W, b):
    x0_min, x0_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x1_min, x1_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx0, xx1 = np.meshgrid(np.linspace(x0_min, x0_max, 200),
                           np.linspace(x1_min, x1_max, 200))
    grid = np.c_[xx0.ravel(), xx1.ravel()]
    probs = model(grid, W, b).reshape(xx0.shape)

    plt.contourf(xx0, xx1, probs, levels=[0, 0.5, 1], alpha=0.2, colors=["red", "green"])
    plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap='summer')
    plt.title("Frontière de décision")
    plt.show()

# Fonction principale
def artificial_neuron(X, y, learning_rate=0.1, n_iter=100):
    W, b = initialisation(X)
    Loss = []

    for i in range(n_iter):
        A = model(X, W, b)
        Loss.append(log_loss(A, y))
        dW, db = gradients(A, X, y)
        W, b = update(dW, db, W, b, learning_rate)

    y_pred = predict(X, W, b)
    print("Accuracy:", accuracy_score(y, y_pred))

    # Tracer la courbe de coût
    plt.plot(Loss)
    plt.title("Évolution du log-loss")
    plt.xlabel("Itérations")
    plt.ylabel("Log-loss")
    plt.grid(True)
    plt.show()

    # Tracer la frontière de décision
    plot_decision_boundary(X, y, W, b)

# Appel
artificial_neuron(X, y)