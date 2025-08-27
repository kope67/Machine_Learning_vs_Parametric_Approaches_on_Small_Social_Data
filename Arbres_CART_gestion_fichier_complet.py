import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split

# ----------------------------
# CLASSIFICATION à partir d'un fichier
# ----------------------------
def cart_classification_from_csv(csv_path, feature_cols, target_col):
    print("=== Classification ===")
    df = pd.read_csv(csv_path, sep=";",encoding="utf-8",on_bad_lines="skip")

    X = df[feature_cols].values
    y = df[target_col].values

    criteria = ['gini', 'entropy']
    for criterion in criteria:
        clf = DecisionTreeClassifier(criterion=criterion, max_depth=3, random_state=0)
        clf.fit(X, y)
        y_pred = clf.predict(X)
        acc = accuracy_score(y, y_pred)

        print(f"\nCritère : {criterion}")
        print(f"Accuracy : {acc:.2f}")

        # Affichage de l'arbre
        plt.figure(figsize=(8, 4))
        plot_tree(clf, filled=True, feature_names=feature_cols, class_names=[str(c) for c in np.unique(y)])
        plt.title(f"Arbre de décision (CART) - {criterion}")
        plt.show()

# ----------------------------
# RÉGRESSION à partir d'un fichier
# ----------------------------
def cart_regression_from_csv(csv_path, feature_col, target_col):
    print("\n=== Régression ===")
    df = pd.read_csv(csv_path)

    X = df[[feature_col]].values
    y = df[target_col].values

    reg = DecisionTreeRegressor(criterion='squared_error', max_depth=3)
    reg.fit(X, y)
    y_pred = reg.predict(X)
    mse = mean_squared_error(y, y_pred)

    print(f"Erreur quadratique moyenne (MSE) : {mse:.2f}")

    # Visualisation
    X_plot = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
    y_plot = reg.predict(X_plot)

    plt.scatter(X, y, color='lightblue', label='Données')
    plt.plot(X_plot, y_plot, color='red', label='Prédiction arbre')
    plt.title("Arbre de régression (CART)")
    plt.xlabel(feature_col)
    plt.ylabel(target_col)
    plt.legend()
    plt.show()

# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    # === Exemple d'utilisation ===

    # --- Classification ---
    csv_path_classif = r"data/donnees_freq_encoded_stricte.csv"
    feature_cols_classif = ["taille_equipe","budget_gere","nb_jour_formation_an","nb_projet","Universite_freq_encoded","Specialisation_freq_encoded","annee_experience_freq_encoded","type_emploi_freq_encoded","secteur_activite_freq_encoded","competence_professionnelle_freq_encoded","salaire_freq_encoded","source_emploi_freq_encoded"]  # à adapter
    target_col_classif = "formation_complement"            # à adapter

    cart_classification_from_csv(csv_path_classif, feature_cols_classif, target_col_classif)

    # --- Régression ---
    csv_path_regr = "C:/chemin/vers/votre_fichier_regr.csv"
    feature_col_regr = "x"    # à adapter
    target_col_regr = "y"     # à adapter

    # cart_regression_from_csv(csv_path_regr, feature_col_regr, target_col_regr)