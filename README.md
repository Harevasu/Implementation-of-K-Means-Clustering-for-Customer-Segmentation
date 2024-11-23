# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
STEP 1 :
Choose the number of clusters (K): Decide how many clusters you want to identify in your data. This is a hyperparameter that you need to set in advance.

STEP 2 :
Initialize cluster centroids: Randomly select K data points from your dataset as the initial centroids of the clusters.

STEP 3 :
Assign data points to clusters: Calculate the distance between each data point and each centroid. Assign each data point to the cluster with the closest centroid. This step is typically done using Euclidean distance, but other distance metrics can also be used.

STEP 4 :
Update cluster centroids: Recalculate the centroid of each cluster by taking the mean of all the data points assigned to that cluster.

STEP 5 :
Repeat steps 3 and 4: Iterate steps 3 and 4 until convergence. Convergence occurs when the assignments of data points to clusters no longer change or change very minimally. 

## Program:


### Developed by: Harevasu S
### RegisterNumber: 212223230069


```
Program to implement the K Means Clustering for Customer Segmentation.
import pandas as pd
df=pd.read_csv("Mall_Customers.csv")
df.head()
df.info()
df.isnull().sum()
from sklearn.cluster import KMeans
wcss=[] #within-CLuster sum of square
for i in range(1,11):
    kmeans = KMeans(n_clusters = i,init = "k-means++")
    kmeans.fit(df.iloc[:,3:])
    wcss.append(kmeans.inertia_)
import matplotlib.pyplot as plt
plt.plot(range(1,11),wcss)
plt.xlabel("No. of clusters")
plt.ylabel("wcss")
plt.title("Elbow Method")
km = KMeans(n_clusters = 5)
km.fit(df.iloc[:,3:])
y_pred = km.predict(df.iloc[:,3:])
y_pred
df["cluster"] = y_pred
a = df[df["cluster"]==0]
b = df[df["cluster"]==1]
c = df[df["cluster"]==2]
d = df[df["cluster"]==3]
e = df[df["cluster"]==4]
plt.scatter(a["Annual Income (k$)"],a["Spending Score (1-100)"],c="red",label="cluster0")
plt.scatter(b["Annual Income (k$)"],b["Spending Score (1-100)"],c="blue",label="cluster1")
plt.scatter(c["Annual Income (k$)"],c["Spending Score (1-100)"],c="black",label="cluster2")
plt.scatter(d["Annual Income (k$)"],d["Spending Score (1-100)"],c="green",label="cluster3")
plt.scatter(e["Annual Income (k$)"],e["Spending Score (1-100)"],c="magenta",label="cluster4")
plt.legend()
plt.title("Customer Segments")
```
## Output:


ELBOW GRAPH:

![image](https://github.com/user-attachments/assets/b985acd5-588b-435b-aca6-2c6721ceb7ff)


PREDICTED VALUES:

![image](https://github.com/user-attachments/assets/baa9c787-3632-4ab2-99aa-7dac3c53dba2)


FINAL GRAPH:

![image](https://github.com/user-attachments/assets/8b594557-cacd-4b42-a5ae-f83c9df4792c)



## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
