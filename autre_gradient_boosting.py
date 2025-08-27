import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# --- 1. Lecture CSV ---
df = pd.read_csv(r"data/simplifier_caractéristique numérique_gestionnaires.csv")

# --- 2. Nettoyage des noms de colonnes ---
df.columns = df.columns.str.strip()            # Supprimer espaces début/fin
df.columns = df.columns.str.replace('\xa0', ' ', regex=False)  # Remplacer espaces insécables
df.columns = df.columns.str.replace(' +', ' ', regex=True)     # Réduire multiples espaces

# Vérification
print("Colonnes détectées :", df.columns.tolist())

# --- 3. Définition des variables explicatives et cible ---
feature_cols = ['13.exp_pro', '15. Taille_équipe']  # à adapter selon affichage print
target_col = 'formation_complement'  # Remplacer par le nom exact de ta variable cible

# Vérifie si toutes les colonnes existent
for col in feature_cols + [target_col]:
    if col not in df.columns:
        raise ValueError(f"Colonne manquante : {col}")

X = df[feature_cols]
y = df[target_col]

# --- 4. Découpage train/test ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 5. Modèle Gradient Boosting ---
model = GradientBoostingClassifier(random_state=42)
model.fit(X_train, y_train)

# --- 6. Évaluation ---
y_pred = model.predict(X_test)
print("\nAccuracy :", accuracy_score(y_test, y_pred))
print("\nClassification Report :\n", classification_report(y_test, y_pred))