import csv

# === Paramètres ===
input_path = r"C:\Users\USER\Desktop\ton_fichier_original.csv"  # CSV source
output_path = r"C:\Users\USER\Desktop\ton_fichier_nettoye.csv"  # CSV nettoyé
sep = ";"   # Séparateur réel du fichier

# === Lecture du fichier et correction ===
with open(input_path, "r", encoding="utf-8", errors="replace") as infile:
    reader = csv.reader(infile, delimiter=sep)
    rows = list(reader)

# On suppose que la première ligne est l'en-tête
header = rows[0]
expected_cols = len(header)

cleaned_rows = [header]  # on garde l'en-tête

for i, row in enumerate(rows[1:], start=2):  # ligne 2 → index 1
    if not row:  # ligne vide → on saute
        print(f"⚠ Ligne vide ignorée : {i}")
        continue

    if len(row) < expected_cols:
        # On complète avec des valeurs vides
        print(f"⚠ Ligne {i} : colonnes manquantes ({len(row)} au lieu de {expected_cols}), complétée avec vides")
        row.extend([""] * (expected_cols - len(row)))

    elif len(row) > expected_cols:
        # On coupe les colonnes en trop
        print(f"⚠ Ligne {i} : trop de colonnes ({len(row)}), les colonnes en trop seront supprimées")
        row = row[:expected_cols]

    cleaned_rows.append(row)

# === Sauvegarde du fichier nettoyé ===
with open(output_path, "w", encoding="utf-8", newline="") as outfile:
    writer = csv.writer(outfile, delimiter=sep)
    writer.writerows(cleaned_rows)

print(f"✅ Fichier nettoyé sauvegardé sous : {output_path}")