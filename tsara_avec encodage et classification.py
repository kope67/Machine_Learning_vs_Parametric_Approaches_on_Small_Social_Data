import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 1. Exemple de jeu de données
data = pd.DataFrame({
    'sexe': ['homme', 'femme', 'femme', 'homme', 'femme', 'homme'],
    'diplome': ['licence', 'master', 'doctorat', 'licence', 'master', 'doctorat'],
    'age': [23, 25, 30, 22, 27, 31],
    'note_moyenne': [12.5, 14.0, 15.2, 11.3, 13.5, 16.0],
    'employe': [0, 1, 1, 0, 1, 1]
})

# 2. Séparer variables explicatives et cible
X = data.drop('employe', axis=1)
y = data['employe']

# 3. Définir colonnes catégorielles et numériques
cat_features = ['sexe', 'diplome']
num_features = ['age', 'note_moyenne']

# 4. Encodage et standardisation via ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), cat_features),
        ('num', StandardScaler(), num_features)
    ]
)

# 5. Pipeline : prétraitement + modèle
pipeline = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# 6. Division train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 7. Entraîner le modèle
pipeline.fit(X_train, y_train)

# 8. Évaluer le modèle
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))