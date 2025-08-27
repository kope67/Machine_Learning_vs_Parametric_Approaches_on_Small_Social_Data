import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# === 1. Lecture du fichier CSV ===
csv_path = "data/simplifier_caractéristique numérique_gestionnaires.csv"  # 🔁 adapte ce chemin
feature_cols = ["13.exp_pro", "15. Taille_équipe"]  # 🔁 colonnes d'entrée
target_col = "formation_complement"                # 🔁 colonne de sortie

#df = pd.read_csv(csv_path)
df = pd.read_csv(csv_path, sep=";", engine="python", encoding="utf-8")

# Séparer X et y
X = df[feature_cols]
y = df[target_col]

# === 2. Modèle Gradient Boosting ===
gb = GradientBoostingClassifier(
    n_estimators=100,     # Nombre d’arbres faibles
    learning_rate=0.1,    # Taux d’apprentissage
    max_depth=3,          # Profondeur max des arbres
    subsample=1.0,        # <1.0 pour stochastic gradient boosting
    random_state=42
)

# Entraînement
gb.fit(X, y)

# Prédiction
y_pred = gb.predict(X)

# Résultat
print("Accuracy Boosting:", accuracy_score(y, y_pred))

# === 3. Visualiser l'évolution du loss ===
plt.plot(gb.train_score_)
plt.title("Évolution du loss à chaque itération")
plt.xlabel("Nombre d’arbres")
plt.ylabel("Déviance (log-loss)")
plt.grid()
plt.show()