import pandas
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn import decomposition

data = pandas.read_csv("data/expanded")
X = pandas.get_dummies(data.drop('edibility', axis='columns')).to_numpy()
scores = []
for i in range(0,10):
    kmeans = KMeans(n_clusters=i+2)
    kmeans.fit(X)
    predicted = kmeans.predict(X)
    scores.append(silhouette_score(X, kmeans.fit_predict(X)))

print("Optimal number of clusters: ", scores.index(max(scores))+2)


# Todo: Select cluster location based on PCA to X i.e select the index of the x-feature as basis for centre coordinates
fig1 = plt.figure()
fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
pca = decomposition.PCA(n_components=3)
pca.fit(X)
X2 = pca.transform(X)

kmeans = KMeans(n_clusters=scores.index(max(scores))+2)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

ax2.scatter(X2[:, 0], X2[:, 1], X2[:, 2], c=y_kmeans, s=50, alpha=0.3, cmap='viridis')

centers = kmeans.cluster_centers_

pca = decomposition.PCA(n_components=3)
pca.fit(centers)
centers = pca.transform(centers)
print(centers.shape)
ax2.scatter(centers[:, 0], centers[:, 1], centers[:,2],  c='black', s=400, alpha=0.5)
plt.show()

