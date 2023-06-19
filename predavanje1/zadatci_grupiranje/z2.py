import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Učitavanje podataka
df = pd.read_csv('Mall_Customers.csv')
spending_score = df['Spending_Score'].values.reshape(-1, 1)
genre = df['Genre']

# 1. Zadatak - K-means algoritam i centri klastera
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(spending_score)
cluster_centers = kmeans.cluster_centers_
print(cluster_centers)

# 2. Zadatak - Grafički prikaz klasterskih podataka
plt.scatter(spending_score, [0] * len(spending_score), c=kmeans.labels_)
plt.xlabel('Spending Score')
plt.ylabel('Cluster Label')
plt.title('K-means Clustering')
plt.show()

# 3. Zadatak - Ovisnost kriterijske funkcije o broju klastera
num_clusters = range(1, 11)
inertia_values = []
for n in num_clusters:
    kmeans = KMeans(n_clusters=n, random_state=42)
    kmeans.fit(spending_score)
    inertia_values.append(kmeans.inertia_)

plt.plot(num_clusters, inertia_values, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Inertia vs Number of Clusters')
plt.show()

