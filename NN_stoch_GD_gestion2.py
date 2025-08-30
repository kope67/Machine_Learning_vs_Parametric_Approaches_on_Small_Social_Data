# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# === 1. Lecture du fichier CSV ===
csv_path = "data/simplifier_caractéristique numérique_gestionnaires.csv"  # 🔁 adapte ce chemin
feature_cols = ["13.exp_pro", "15. Taille_équipe"]  # 🔁 colonnes d'entrée
target_col = "formation_complement"                # 🔁 colonne de sortie

df = pd.read_csv(csv_path, sep=";", engine="python", encoding="utf-8")

# Remplacer les virgules par des points pour les colonnes numériques
for col in feature_cols:
    df[col] = df[col].astype(str).str.replace(',', '.').astype(float)

# Encodage de la cible
if df[target_col].dtype == object:
    df[target_col] = df[target_col].astype("category").cat.codes

# Extraction des données
X = df[feature_cols].values
y = df[target_col].values.reshape(-1, 1)

print('Dimension de X:', X.shape)
print('Dimension de y:', y.shape)

# Visualisation
plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='summer')
plt.title("Visualisation des données")
plt.show()

# === 2. Modèle ===
def initialisation(X):
    W = np.random.randn(X.shape[1], 1)
    b = np.random.randn(1)
    return (W, b)

def model(X, W, b):
    z = X.dot(W) + b
    z = np.clip(z, -500, 500)
    A = 1 / (1 + np.exp(-z))
    return A

def log_loss(A, y):
    epsilon = 1e-15
    A = np.clip(A, epsilon, 1 - epsilon)
    return 1 / len(y) * np.sum(-y * np.log(A) - (1 - y) * np.log(1 - A))

def gradients(xi, yi, W, b):
    zi = np.dot(xi, W) + b
    zi = np.clip(zi, -500, 500)
    ai = 1 / (1 + np.exp(-zi))
    dW = np.dot(xi.T, (ai - yi))
    db = ai - yi
    return dW, db

def update(dW, db, W, b, learning_rate):
    W = W - learning_rate * dW
    b = b - learning_rate * db
    return (W, b)

def predict(X, W, b):
    A = model(X, W, b)
    return A >= 0.5

def artificial_neuron_SGD(X, y, learning_rate=0.01, n_iter=100):
    m = X.shape[0]
    W, b = initialisation(X)
    Loss = []

    for epoch in range(n_iter):
        loss_epoch = 0
        for i in range(m):
            xi = X[i].reshape(1, -1)
            yi = y[i].reshape(1, 1)
            ai = model(xi, W, b)
            loss_epoch += log_loss(ai, yi)
            dW, db = gradients(xi, yi, W, b)
            W, b = update(dW, db, W, b, learning_rate)
        Loss.append(loss_epoch / m)

    y_pred = predict(X, W, b)
    print("Accuracy :", accuracy_score(y, y_pred))

    plt.plot(Loss)
    plt.title("Évolution de la log-loss (SGD)")
    plt.xlabel("Époques")
    plt.ylabel("Log Loss")
    plt.grid(True)
    plt.show()

# === 3. Appel du modèle ===
artificial_neuron_SGD(X, y)