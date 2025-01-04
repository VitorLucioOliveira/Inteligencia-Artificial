import pickle
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Carregar os dados do arquivo .pkl
file_path = '/content/sample_data/Heart.pkl'
with open(file_path, 'rb') as f:
    X_train, X_test, y_train, y_test = pickle.load(f)

# Converter os dados para DataFrames e Series (caso necessário)
X_train = pd.DataFrame(X_train)
y_train = pd.Series(y_train)
X_test = pd.DataFrame(X_test)
y_test = pd.Series(y_test)

# Normalizar os dados para melhorar a performance do KNN
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Definir o modelo base KNN
knn = KNeighborsClassifier()

# Definir os parâmetros a serem testados no Grid Search
param_grid = {
    'n_neighbors': range(1, 21),  # Testar de 1 a 20 vizinhos
    'weights': ['uniform', 'distance'],  # Pesos uniformes ou baseados na distância
    'metric': ['euclidean', 'manhattan', 'minkowski'],  # Métricas de distância
}

# Configurar o Grid Search com validação cruzada
grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Ajustar o modelo aos dados de treino
grid_search.fit(X_train, y_train)

# Melhor combinação de parâmetros
best_params = grid_search.best_params_

# Melhor acurácia no Grid Search
best_score = grid_search.best_score_

# Fazer previsões com o melhor modelo
best_knn = grid_search.best_estimator_
y_pred = best_knn.predict(X_test)

# Avaliar o modelo otimizado
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Exibir os resultados
print("\nMelhores Hiperparâmetros:", best_params)
print(f"Melhor Acurácia no Treinamento: {best_score:.2f}")
print(f"Acurácia no Teste: {accuracy:.2f}")
print("\nMatriz de Confusão:")
print(conf_matrix)
print("\nRelatório de Classificação:")
print(report)
