import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from sklearn import cluster, datasets, preprocessing, metrics
from sklearn.metrics import pairwise_distances
from sklearn import cluster, preprocessing, metrics
from sklearn import decomposition

# Load training data

df = pd.read_csv('~/.kaggle/competitions/forest-cover-type-kernels-only/train.csv')

# Print mean of the first 10 columns, excluding the id and the the binary fields 
print(df.columns)
print('Mean of each column:\n', df[df.columns[1:11]].mean(), '\n')
print('Median of each column:\n', df[df.columns[1:11]].median(), '\n')
print('Variance of each column:\n', df[df.columns[1:11]].var())

ax = plt.scatter(x=df['Horizontal_Distance_To_Hydrology'], y=df['Vertical_Distance_To_Hydrology'], c=df['Cover_Type'], cmap='jet')
plt.xlabel('Horizontal Distance')
plt.ylabel('Vertical Distance')
plt.title("Distance to Hydrology with Elevation")
plt.show()

# First analyse variable singularly, for instance to check if they are normally distributed
'''
for col in df.columns[1:11]:
    plt.figure(col)
    plt.hist(df[col], 50)

plt.figure('Wilderness area')
values = []
for col in df.columns[11:15]:
    values.append(df[col].astype(bool).sum(axis=0))

plt.bar(np.arange(1, 5), values)
plt.xticks(np.arange(1, 5, 1.0))

plt.figure('Soil type')
values.clear()
for col in df.columns[15:55]:
    values.append(df[col].astype(bool).sum(axis=0))

plt.bar(np.arange(1, 41), values)
plt.xticks(np.arange(1, 41, 1.0))

'''

# Identifying potential features by looking at correlations between features and cover type 
'''
for col in df.columns[1:11]:
    plt.figure(col)
    plt.scatter(df[col], df['Cover_Type'])
    plt.xlim(xmin=min(df[col]))
plt.show()
'''
# Identifying potential features by looking at soil type / wilderness area in correlation with cover type.
# Soil type and wilderness areas are binary features hence I count the occurencies of both and bar plot it.
'''
for col in df.columns[15:55]:
    plt.figure(col + ' / Cover Type' )
    values = []
    for i in range(1, 8):
        d = df[(df['Cover_Type'] == i) & (df[col] == 1)].count() 
        values.append(d[col]) 
    plt.bar(np.arange(1,8), values)
plt.show()
'''
'''
for col in df.columns[11:15]:
    plt.figure(col + ' / Cover Type' )
    values = []
    for i in range(1, 8):
        d = df[(df['Cover_Type'] == i) & (df[col] == 1)].count() 
        values.append(d[col]) 
    plt.bar(np.arange(1,8), values)
plt.show()
'''

# Clustering
# Drop target col and normalize data. We have values all over the place like 'Elevation' which goes up to thousands,
# while other values are inherently low.

X_scaled = preprocessing.normalize(df.drop(['Id', "Cover_Type"], axis=1), axis=0)

# Apply k-manes clustering
k = 7
kmeans = cluster.KMeans(n_clusters=k)
kmeans.fit(X_scaled)

inertia = kmeans.inertia_
print('Silhouette Score:', metrics.silhouette_score(X_scaled, df['Cover_Type'], metric='euclidean'))

# Dimensionality reduction with PCA
pca = decomposition.PCA(n_components=40)
pca.fit(X_scaled)
X = pca.transform(X_scaled)

plt.figure()
colors = ['navy', 'turquoise', 'darkorange', 'red', 'blue', 'green', 'black']
lw = 2
target_names = ['Spruce/Fir', 'Lodgepole Pine', 'Ponderosa Pine', 'Cottonwood/Willow', 'Aspen','Douglas-fir', 'Krummholz']

for color, i, target_name in zip(colors, np.arange(1,8), target_names):
    plt.scatter(X[df['Cover_Type'] == i, 0], X[df['Cover_Type'] == i, 1], color=color, alpha=.8, lw=lw,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of Forest Cover Type')
#plt.show()
print(pca.explained_variance_ratio_)
