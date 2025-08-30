import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# --- Lecture du CSV ---
csv_path = "data/simplifier_caractéristique numérique_gestionnaires.csv"
feature_cols = ["13.exp_pro", "15. Taille_équipe"]
target_col = "formation_complement"

df = pd.read_csv(csv_path, sep=";", engine="python", encoding="utf-8")

# Séparer X et y
X = df[feature_cols]
y = df[target_col]

# --- Découpage train/test ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Modèle Arbre de régression ---
reg_tree = DecisionTreeRegressor(
    max_depth=3,      # profondeur maximale (à ajuster)
    random_state=42
)
reg_tree.fit(X_train, y_train)

# --- Évaluation ---
y_pred_train = reg_tree.predict(X_train)
y_pred_test = reg_tree.predict(X_test)

print("R² entraînement :", r2_score(y_train, y_pred_train))
print("R² test         :", r2_score(y_test, y_pred_test))
print("RMSE test       :", mean_squared_error(y_test, y_pred_test))