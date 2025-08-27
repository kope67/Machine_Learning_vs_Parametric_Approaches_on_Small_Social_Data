
def main():
    # === 1. Lire les données Excel ===
    excel_path = r"data/quantitative_vs_formation.xlsx"
    df = pd.read_excel(excel_path)

    # === 2. Choisir la colonne cible ===
    target_column = "y"  # Remplacer par le nom exact de la colonne cible
    y = df[target_column].values.reshape(-1, 1)
    X = df.drop(columns=[target_column]).values

    # === 3. Séparation train/test ===
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # === 4. Standardisation ===
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # === 5. Entraînement ===
    W, b, losses = train_mini_batchGD(X_train, y_train, learning_rate=0.1, n_iter=100, batch_size=32)

    # === 6. Prédiction et évaluation ===
    y_pred = predict(X_test, W, b)
    print("Accuracy sur test:", accuracy_score(y_test, y_pred))

    # === 7. Affichage des pertes ===
    plt.plot(losses)
    plt.xlabel("Époques")
    plt.ylabel("Log Loss")
    plt.title("Mini-batch Gradient Descent")
    plt.grid(True)
    plt.show()

    # === 8. Visualisation si 2 variables ===
    if X.shape[1] == 2:
        plot_decision_boundary(X_test, y_test, W, b)

if __name__ == "__main__":
    main()