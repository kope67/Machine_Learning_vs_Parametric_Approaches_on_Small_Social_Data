import pandas as pd

# --- Lecture du fichier Excel ---
# Remplace le chemin par celui de ton fichier
fichier_excel = r"data/les_gestionnaires2.xlsx"
df = pd.read_excel(fichier_excel)

print("Aperçu du fichier original :")
print(df.head())

# --- Encodage basé sur la fréquence ---
def frequency_encoding(df, colonne):
    """Transforme une colonne qualitative en valeurs numériques selon leur fréquence"""
    freq = df[colonne].value_counts(normalize=True)  # fréquence relative
    df[colonne + "_freq_encoded"] = df[colonne].map(freq)
    return df

# Exemple : encoder toutes les colonnes object/catégorielles
colonnes_categ = df.select_dtypes(include=['object']).columns

for col in colonnes_categ:
    df = frequency_encoding(df, col)

print("\nDonnées après encodage par fréquence :")
print(df.head())

# --- Sauvegarde dans un nouveau fichier Excel ---
fichier_sortie = r"data/donnees_freq_encoded.xlsx"
df.to_excel(fichier_sortie, index=False)
print(f"Fichier encodé sauvegardé sous : {fichier_sortie}")