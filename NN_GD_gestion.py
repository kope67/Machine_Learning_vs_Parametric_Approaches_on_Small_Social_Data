# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# === 1. Lecture du fichier CSV ===
csv_path = "data/simplifier_caractéristique numérique_gestionnaires.csv"  # 🔁 adapte ce chemin
feature_cols = ["13.exp_pro", "15. Taille_équipe"]  # 🔁 noms des colonnes d'entrée
target_col = "formation_complement"             # 🔁 nom de la colonne de sortie

# Lecture avec gestion des virgules et encodage
df = pd.read_csv(csv_path, sep=";", engine="python", encoding="utf-8")

# Conversion : remplacer les virgules décimales par des points
for col in feature_cols:
    df[col] = df[col].astype(str).str.replace(',', '.').astype(float)

# Encodage de la cible si nécessaire
if df[target_col].dtype == object:
    df[target_col] = df[target_col].astype("category").cat.codes

# Extraction des features et de la cible
X = df[feature_cols].values
y = df[target_col].values.reshape(-1, 1)

print('Dimension de X:', X.shape)
print('Dimension de y:', y.shape)

# Visualisation (si X est 2D)
plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='summer')
plt.title("Visualisation des données")
plt.show()

# === 2. Réseau de neurone logistique ===
def initialisation(X):
    W = np.random.randn(X.shape[1], 1)
    b = np.random.randn(1)
    return (W, b)

def model(X, W, b):
    z = X.dot(W) + b
    z = np.clip(z, -500, 500)  # prévention overflow
    A = 1 / (1 + np.exp(-z))
    return A

def log_loss(A, y):
    epsilon = 1e-15
    A = np.clip(A, epsilon, 1 - epsilon)  # force A à rester dans [ε, 1-ε]
    return 1 / len(y) * np.sum(-y * np.log(A) - (1 - y) * np.log(1 - A))

def gradients(A, X, y):
    dw = 1 / len(y) * np.dot(X.T, A - y)
    db = 1 / len(y) * np.sum(A - y)
    return (dw, db)

def update(dw, db, W, b, learning_rate):
    W = W - learning_rate * dw
    b = b - learning_rate * db
    return (W, b)

def predict(X, W, b):
    A = model(X, W, b)
    return A >= 0.5

def artificial_neuron(X, y, learning_rate=0.1, n_iter=100):
    W, b = initialisation(X)
    Loss = []

    for i in range(n_iter):
        A = model(X, W, b)
        Loss.append(log_loss(A, y))
        dW, db = gradients(A, X, y)
        W, b = update(dW, db, W, b, learning_rate)

    y_pred = predict(X, W, b)
    print("Accuracy :", accuracy_score(y, y_pred))

    plt.plot(Loss)
    plt.title("Évolution de la log-loss")
    plt.xlabel("Itérations")
    plt.ylabel("Log Loss")
    plt.show()

# === 3. Appel du modèle ===
artificial_neuron(X, y)