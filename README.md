# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries and load the Mall Customers dataset.
2. Check the dataset for information and missing values.
3. Use the K-means clustering algorithm and apply the Elbow Method to find the optimal number of clusters.
4. Train the K-Means model with 5 clusters and predict the cluster values for each customer.
5. Visualize the clusters using a scatter plot of Annual Income versus Spending Score. 


## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: SHALINI D
RegisterNumber:  25011579
*/
```
```
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("C:/Users/acer/Downloads/Mall_Customers.csv")

print(data.head())

print(data.info())

data.isnull().sum()

from sklearn.cluster import KMeans
wcss = []

for i in range(1,11):
    kmeans = KMeans(n_clusters = i,init = "k-means++")
    kmeans.fit(data.iloc[:,3:])
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.xlabel("No of Cluster")
plt.ylabel("wcss")
plt.title("Elbow Method")
plt.figure()

km = KMeans(n_clusters = 5)
km.fit(data.iloc[:,3:])

KMeans(n_clusters=5)

y_pred = km.predict(data.iloc[:,3:])
print("Predicted values: \n",y_pred)

data["cluster"]=y_pred
df0 = data[data["cluster"]==0]
df1 = data[data["cluster"]==1]
df2 = data[data["cluster"]==2]
df3 = data[data["cluster"]==3]
df4 = data[data["cluster"]==4]
plt.scatter(df0["Annual Income (k$)"],df0["Spending Score (1-100)"],c="red",label="cluster0")
plt.scatter(df1["Annual Income (k$)"],df1["Spending Score (1-100)"],c="black",label="cluster1")
plt.scatter(df2["Annual Income (k$)"],df2["Spending Score (1-100)"],c="blue",label="cluster2")
plt.scatter(df3["Annual Income (k$)"],df3["Spending Score (1-100)"],c="green",label="cluster3")
plt.scatter(df4["Annual Income (k$)"],df4["Spending Score (1-100)"],c="magenta",label="cluster4")
plt.legend()
plt.title("Customer Segments")
plt.show()
```


## Output:
<img width="822" height="426" alt="Screenshot 2026-02-25 093003" src="https://github.com/user-attachments/assets/3df07ddd-d406-4661-afd7-53729354965c" />
<img width="973" height="744" alt="Screenshot 2026-02-25 093155" src="https://github.com/user-attachments/assets/fb0cd4c8-a6d8-4ea7-a556-e95c213062d0" />
<img width="832" height="555" alt="Screenshot 2026-02-25 093202" src="https://github.com/user-attachments/assets/895bf606-ba6c-470b-86bc-cf45880d5f0c" />





## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
