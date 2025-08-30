import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def soft_thresholding(x, threshold):
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

def lasso_loss(X, y, w, lambd):
    """Fonction de perte : MSE + pénalité L1"""
    n = len(y)
    mse = (1 / (2 * n)) * np.sum((X @ w - y) ** 2)
    l1 = lambd * np.sum(np.abs(w))
    return mse + l1

def lasso_gradient_proximal(X, y, lambd, lr=1e-2, max_iter=1000, tol=1e-6):
    n, p = X.shape
    w = np.zeros(p)
    loss_history = []

    for i in range(max_iter):
        # Calcul du gradient de la partie MSE
        grad = (1 / n) * X.T @ (X @ w - y)

        # Mise à jour avec soft-thresholding (proximal L1)
        w_new = soft_thresholding(w - lr * grad, lr * lambd)

        # Calcul de la perte et stockage
        loss = lasso_loss(X, y, w_new, lambd)
        loss_history.append(loss)

        # Critère d'arrêt
        if np.linalg.norm(w_new - w) < tol:
            break

        w = w_new

    return w, loss_history

# === 1. Lecture du fichier CSV ===
csv_path = "data/simplifier_caractéristique numérique_gestionnaires.csv"
feature_cols = ["13.exp_pro", "15. Taille_équipe"]
target_col = "formation_complement"

df = pd.read_csv(csv_path, sep=";", engine="python", encoding="utf-8")

# Séparer X et y
X = df[feature_cols]
y = df[target_col]

# Normalisation
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = y - np.mean(y)

# Paramètres
lambda_ = 0.1
learning_rate = 0.1

# Entraînement
w_lasso, loss_history = lasso_gradient_proximal(X, y, lambd=lambda_, lr=learning_rate)

print("Poids appris (w):\n", w_lasso)

# === 2. Affichage de l'évolution de la perte ===
plt.figure(figsize=(8, 5))
plt.plot(loss_history, label="Fonction perte (MSE + L1)")
plt.xlabel("Itérations")
plt.ylabel("Valeur de la perte")
plt.title("Évolution de la fonction perte - Lasso")
plt.legend()
plt.grid(True)
plt.show()