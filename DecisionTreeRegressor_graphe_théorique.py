import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error

# ---------------------------
# 1. Génération d'un jeu de données simple
# ---------------------------
np.random.seed(42)
X = np.sort(np.random.rand(80, 1) * 5, axis=0)  # variable entre 0 et 5
y = np.sin(X).ravel() + np.random.normal(scale=0.2, size=X.shape[0])  # signal bruité

# ---------------------------
# 2. Découpage train/test
# ---------------------------
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------
# 3. Modèle : DecisionTreeRegressor
# ---------------------------
tree = DecisionTreeRegressor(max_depth=3, random_state=42)
tree.fit(X_train, y_train)

# ---------------------------
# 4. Évaluation
# ---------------------------
y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)

r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

print(f"R² entraînement : {r2_train}")
print(f"R² test         : {r2_test}")
print(f"RMSE test       : {rmse_test}")

# ---------------------------
# 5. Visualisation
# ---------------------------
# On crée un maillage fin pour dessiner la courbe en escalier
X_plot = np.linspace(0, 5, 500).reshape(-1, 1)
y_plot = tree.predict(X_plot)

plt.figure(figsize=(8,5))
plt.scatter(X_train, y_train, color="blue", label="Données entraînement", alpha=0.6)
plt.scatter(X_test, y_test, color="green", label="Données test", alpha=0.6)
plt.plot(X_plot, y_plot, color="red", linewidth=2, label="Prédiction arbre (escalier)")
plt.title("Régression avec DecisionTreeRegressor")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()