import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler=pd.read_csv("musteriler.csv")

X= veriler.iloc[:,3:] #yas ve hacimi cektik

#kmeans
from sklearn.cluster import KMeans

kmeans=KMeans(n_clusters=3,init="k-means++")

kmeans.fit(X)
print(kmeans.cluster_centers_)
sonuclar=[]
for i in range(1,10): # k icin optimum deger
    kmeans=KMeans(n_clusters=i,init="k-means++",random_state=123)
    kmeans.fit(X)
    sonuclar.append(kmeans.inertia_)

plt.plot(range(1,10),sonuclar)    
plt.show()

from sklearn.cluster import AgglomerativeClustering
ac=AgglomerativeClustering(n_clusters=3,affinity="euclidean",linkage="ward")
Y_tahmin=ac.fit_predict(X)
print(Y_tahmin)
X=np.array(X)
plt.scatter(X[Y_tahmin==0,0],X[Y_tahmin==0,1],s=100,c="red")
plt.scatter(X[Y_tahmin==1,0],X[Y_tahmin==1,1],s=100,c="yellow")
plt.scatter(X[Y_tahmin==2,0],X[Y_tahmin==2,1],s=100,c="purple")
plt.show()
import scipy.cluster.hierarchy as sch
dendrogram=sch.dendrogram(sch.linkage(X,method="ward"))
plt.show()











