import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# --- Fonctions utilitaires ---
def soft_thresholding(x, threshold):
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

def lasso_loss(X, y, w, lambd):
    """Fonction perte : MSE + pénalité L1"""
    n = len(y)
    mse = (1 / (2 * n)) * np.sum((X @ w - y) ** 2)
    l1 = lambd * np.sum(np.abs(w))
    return mse + l1

def lasso_gradient_proximal(X, y, lambd, lr=1e-2, max_iter=1000, tol=1e-6):
    n, p = X.shape
    w = np.zeros(p)
    loss_history = []

    for i in range(max_iter):
        # Gradient de la partie MSE
        grad = (1 / n) * X.T @ (X @ w - y)
        
        # Mise à jour (proximal step L1)
        w_new = soft_thresholding(w - lr * grad, lr * lambd)
        
        # Stocker la perte
        loss = lasso_loss(X, y, w_new, lambd)
        loss_history.append(loss)

        # Critère d'arrêt
        if np.linalg.norm(w_new - w) < tol:
            break

        w = w_new

    return w, loss_history

# --- Lecture du CSV ---
csv_path = "data/simplifier_caractéristique numérique_gestionnaires.csv"
feature_cols = ["13.exp_pro", "15. Taille_équipe"]
target_col = "formation_complement"

df = pd.read_csv(csv_path, sep=";", engine="python", encoding="utf-8")

# Séparer X et y
X = df[feature_cols]
y = df[target_col]

# Standardisation
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = y - np.mean(y)

# --- Paramètres ---
learning_rate = 0.1
lambdas = [0.1, 0.01]  # λ fort et λ faible

# --- Entraînements ---
results = {}
for lambd in lambdas:
    w, loss_hist = lasso_gradient_proximal(X, y, lambd=lambd, lr=learning_rate)
    results[lambd] = {"w": w, "loss": loss_hist}
    print(f"λ = {lambd} → poids appris : {w}")

# --- Tracé ---
plt.figure(figsize=(8, 5))
for lambd in lambdas:
    plt.plot(results[lambd]["loss"], label=f"λ = {lambd}")
plt.xlabel("Itérations")
plt.ylabel("Perte (MSE + L1)")
plt.title("Impact de la régularisation L1 sur la convergence du Lasso")
plt.legend()
plt.grid(True)
plt.show()