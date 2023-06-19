import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#1. Zadatak
#Učitajte Mall_Customers.csv podatke. Koristeći Spending_score veličinu, te jednu veličinu po
#izboru odredite centre klastera koristeći Kmeans algoritam.

df=pd.read_csv('Mall_Customers.csv')

spending_score=df['Spending_Score'].values.reshape(-1, 1)
genre=df['Genre']

kmeans=KMeans(n_clusters=2)
kmeans.fit(spending_score)
print(kmeans.cluster_centers_)


#2. Zadatak
#Grafički prikažite izračunate klastere, odnosno obojite pojedini podatak ovisno o njegovoj
#pripadnosti određenom klasteru.
def show_clusters(points,cluster_labels):
    first_cluster_points=[]
    second_cluster_points=[]
    for i in range(len(cluster_labels)):
        cluster_class=cluster_labels[i]
        if cluster_class==0:
            first_cluster_points.append(points[i])
        elif cluster_class==1:
            second_cluster_points.append(points[i])
            
    first_cluster_points_x=[point[0] for point in first_cluster_points]
    first_cluster_points_y=[point[1] for point in first_cluster_points]
    
    second_cluster_points_x=[point[0] for point in second_cluster_points]
    second_cluster_points_y=[point[1] for point in second_cluster_points]
    
    plt.scatter(first_cluster_points_x,first_cluster_points_y,c='red')
    plt.scatter(second_cluster_points_x,second_cluster_points_y,c='blue')
    plt.show()

print(kmeans.labels_)
show_clusters(genre,kmeans.labels_)

plt.scatter(spending_score,genre)
#plt.show()

#3. Zadatak
#Mijenjajte broj klastera, te grafički prikažite ovisnost kriterijske funkcije o broju klastera. Na
#temelju grafičkog prikaza odredite koji je optimalan broj klastera.














