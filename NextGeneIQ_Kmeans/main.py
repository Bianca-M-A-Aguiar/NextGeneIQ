# Importar bibliotecas necessárias
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Carregar a base de dados
data = pd.read_csv('C:/Users/bianc/OneDrive/Ambiente de Trabalho/DNA_Dataset_Normalized.csv')

# Remover a coluna 'Class' para realizar clustering
X = data.drop(columns=['Class'])

# Normalizar os dados para que todas as variáveis tenham a mesma escala
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicar o algoritmo K-means
kmeans = KMeans(n_clusters=5, random_state=42)  # Considerando 5 clusters como o número de classes
clusters = kmeans.fit_predict(X_scaled)

# Adicionar a coluna de clusters ao DataFrame original
data['Cluster'] = clusters

# Mostrar os primeiros registros com os clusters atribuídos
print(data[['Class', 'Cluster']].head())

# Visualizar os clusters
plt.scatter(data['gene_1'], data['gene_2'], c=data['Cluster'], cmap='viridis')
plt.xlabel('Gene 1')
plt.ylabel('Gene 2')
plt.title('Clusters de Câncer com K-means')
plt.colorbar()
plt.show()