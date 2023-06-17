import numpy as np 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


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

X=np.array([[1,2],[1,4],[1,0],[10,2],[10,4],[10,0]])
plt.scatter(X[:,0],X[:,1])
#plt.show()


kmeans=KMeans(n_clusters=2)
kmeans.fit(X)

print(kmeans.labels_)
print(kmeans.cluster_centers_)

show_clusters(X,kmeans.labels_)


real_data_points=np.array([[0,0],[12,3]])

point_cluster_labels=kmeans.predict(real_data_points)
print(point_cluster_labels)
print(kmeans.cluster_centers_)
print(kmeans.inertia_)
















