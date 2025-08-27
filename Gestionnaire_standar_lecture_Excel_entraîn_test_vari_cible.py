import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def sigmoid(z):
    z = np.array(z, dtype=float)  # assure que z est bien un numpy.array
    return 1 / (1 + np.exp(-z))

def initialize(n_features):
    W = np.random.randn(n_features, 1) * 0.01
    b = 0.
    return W, b

def model(x, W, b):
    return sigmoid(np.dot(x, W) + b)

def compute_loss(y_hat, y):
    m = len(y)
    eps = 1e-15
    y_hat = np.clip(y_hat, eps, 1 - eps)
    return -1/m * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

def gradients(X, y, y_hat):
    m = len(y)
    dz = y_hat - y
    dW = (1/m) * np.dot(X.T, dz)
    db = (1/m) * np.sum(dz)
    return dW, db

def update(W, b, dW, db, learning_rate):
    W = W - learning_rate * dW
    b = b - learning_rate * db
    return W, b

def predict(x, W, b):
    y_prob = model(x, W, b)
    return (y_prob >= 0.5).astype(int)

def train_mini_batchGD(X, y, learning_rate=0.1, n_iter=100, batch_size=32):
    m, n = X.shape
    W, b = initialize(n)
    losses = []
    
    for epoch in range(n_iter):
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        for i in range(0, m, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            y_hat = model(X_batch, W, b)
            loss = compute_loss(y_hat, y_batch)
            dW, db = gradients(X_batch, y_batch, y_hat)
            W, b = update(W, b, dW, db, learning_rate)
        
        losses.append(loss)

    return W, b, losses

def plot_decision_boundary(X, y, W, b):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = model(grid, W, b).reshape(xx.shape)
    
    plt.contourf(xx, yy, probs, levels=[0, 0.5, 1], alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), edgecolors='k')
    plt.title("Frontière de décision")
    plt.show()

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