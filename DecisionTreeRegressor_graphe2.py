import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# --- Lecture du CSV ---
csv_path = "data/simplifier_caractéristique numérique_gestionnaires.csv"
feature_cols = ["13.exp_pro", "15. Taille_équipe"]
target_col = "14.H_trav_moy"

df = pd.read_csv(csv_path, sep=";", engine="python", encoding="utf-8")

# Séparer X et y
X = df[feature_cols]
y = df[target_col]

# Séparer train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modèle
tree = DecisionTreeRegressor(max_depth=3, random_state=42)
tree.fit(X_train, y_train)

# Évaluation
y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)

print("R² entraînement :", r2_score(y_train, y_train_pred))
print("R² test         :", r2_score(y_test, y_test_pred))
print("RMSE test       :", mean_squared_error(y_test, y_test_pred))

# ---------------------------
# Visualisation : Courbe en escalier sur 1 variable
# ---------------------------
# On fixe la deuxième variable à sa moyenne
exp_pro_range = np.linspace(X["13.exp_pro"].min(), X["13.exp_pro"].max(), 300)
taille_moy = np.full_like(exp_pro_range, X["15. Taille_équipe"].mean())

X_plot = np.column_stack((exp_pro_range, taille_moy))
y_plot = tree.predict(X_plot)

plt.figure(figsize=(8,6))
plt.scatter(X["13.exp_pro"], y, color="blue", alpha=0.6, label="Données réelles")
plt.plot(exp_pro_range, y_plot, color="red", linewidth=2, drawstyle="steps-post", label="Prédiction de l'arbre")
plt.xlabel("Expérience professionnelle (années)")
plt.ylabel("Heures de travail par semaine moyenne")
plt.title("Arbre de régression : Prédiction en escalier")
plt.legend()
plt.show()