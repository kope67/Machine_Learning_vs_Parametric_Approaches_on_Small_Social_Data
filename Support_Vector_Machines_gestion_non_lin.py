# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# === 1. Lecture et préparation des données ===
csv_path = "data/simplifier_caractéristique numérique_gestionnaires.csv"
feature_cols = ["13.exp_pro", "15. Taille_équipe"]
target_col = "formation_complement"

# Lecture CSV
df = pd.read_csv(csv_path, sep=";", engine="python", encoding="utf-8")

# Remplacer virgules par points pour les colonnes numériques
for col in feature_cols:
    df[col] = df[col].astype(str).str.replace(',', '.').astype(float)

# Encodage de la cible si elle est catégorielle
if df[target_col].dtype == object:
    df[target_col] = df[target_col].astype("category").cat.codes

# Extraction des features et de la cible
X = df[feature_cols].values
y = df[target_col].values

# === 2. Séparation des données ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardisation
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# === 3. SVM avec kernel linéaire ===
clf = SVC(kernel='rbf', random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# === 4. Évaluation ===
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Rapport de classification :\n", classification_report(y_test, y_pred))

# === 5. Visualisation (si 2D) ===
plt.figure(figsize=(8, 6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='coolwarm', edgecolors='k')
plt.title("Prédictions du SVM sur les données de test")
plt.xlabel(feature_cols[0])
plt.ylabel(feature_cols[1])
plt.grid(True)
plt.show()




