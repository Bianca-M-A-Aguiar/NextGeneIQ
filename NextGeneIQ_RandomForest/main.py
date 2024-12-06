# Importar bibliotecas necessárias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Carregar a base de dados
data = pd.read_csv('C:/Users/bianc/OneDrive/Ambiente de Trabalho/DNA_Dataset_Normalized.csv')

# Dividir em variáveis de entrada (genes) e saída (Class)
X = data.drop(columns=['Class'])  # Todas as colunas exceto 'Class'
y = data['Class']  # Coluna alvo

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar o modelo de Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Fazer previsões
y_pred = clf.predict(X_test)

# Avaliar o modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred))

# Exemplo de predição com novos dados
new_sample = X_test.iloc[0:1]  # exemplo do conjunto de teste
new_prediction = clf.predict(new_sample)
print("Predição para novo exemplo:", new_prediction)
