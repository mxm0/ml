import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from collections import Counter

# Compute average error of classifier
def loss_function(y_predicted, y_expected):
    err = 0
    n_points = len(y_predicted)
    for i in range(n_points):
      err += np.square((y_expected[i] - y_predicted[i]))
    return 1/n_points * err

# K-nearest neighbor implementation
def knn(X_train, y_train, X_test, y_predicted, k):
    test_point_n = len(X_test)
    current_test_point = 0
    for x_test_p in X_test:
        distances = []
        k_neighbors = []
        i = 0
        # Compute euclidean distane and sort in ascendind order
        for x_train_p in X_train:
            eu_distance = np.sqrt(np.sum(np.square(x_test_p - x_train_p)))
            distances.append([eu_distance, i])
            i += 1
        print('### Distances computed for test point number', current_test_point, 'sorting distances...')
        distances = sorted(distances)
        print('### Distances sorted, selecting k nearest neighbors...')
        # take k nearest neighbors labels
        for j in range(k):
            y_index = distances[j][1]
            k_neighbors.append(y_train[y_index])

        # return label for most common neighbor
        y_predicted.append(Counter(k_neighbors).most_common(1)[0][0])
        current_test_point += 1
        print('Y predicted for test point number', current_test_point)

#load data
df = pd.read_csv('housing.csv')
# Print max, min and their index for each data column.
'''
for col in df:
    if col != 'ocean_proximity':
        data = np.array(df[col])
        data = [x for x in data if ~np.isnan(x)]
        print('Max value for', col, ': ', np.max(data), ', at index=', np.argmax(data))
        print('Min value for', col, ': ', np.min(data), ', at index=', np.argmin(data))
        print('Avg value for', col, ': ', np.average(data), '\n')
        plt.figure(col)
        plt.title('Histogram values')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.hist([x for x in df[col] if ~np.isnan(x)])
    else:
        plt.figure(col)
        plt.title('Histogram values')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.hist([x for x in df[col] if ~pd.isnull(x)])
'''
# Plotting data
'''
From the data plotted it looks like the median income, housing median age and
median house value data are the closest to a normal distribution.
'''
'''
# Plot latitude/longitude
plt.figure('GeoMap')
plt.title('Geographic Map House values')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
latitudes = df['latitude']
longitudes = df['longitude']
house_values = df['median_house_value']
cm = plt.cm.get_cmap('RdBu')
plt.scatter(longitudes, latitudes, alpha=0.2, c=house_values, cmap=cm)
plt.colorbar()
plt.show()
'''
# Splitting data set 8/20-training/test
train_set, test_set = train_test_split(df, test_size = 0.2, random_state = 42)
'''
# Recaculting max, min, avg for both training and test dataset
for train_col, test_col in zip(train_set, test_set):
    if train_col != 'ocean_proximity' or test_col != 'ocean_proximity':
        train_data = np.array(train_set[train_col])
        test_data = np.array(test_set[test_col])
        train_data = [x for x in train_data if ~np.isnan(x)]
        test_data = [x for x in test_data if ~np.isnan(x)]
        # Train data
        print('Max value for', train_col, 'training: ', np.max(train_data))
        print('Min value for', train_col, 'training: ', np.min(train_data))
        print('Avg value for', train_col, 'training: ', np.average(train_data))
        # Test data
        print('Max value for', test_col, 'test: ', np.max(test_data))
        print('Min value for', test_col, 'test: ', np.min(test_data))
        print('Avg value for', test_col, 'test: ', np.average(test_data), '\n')
'''
'''
Since the mean values seem to be equal or just differ for few points, we can
assume that the distribution match. Plus plotting an histogram of the two
distribution shows it too.
'''
#kNN Algorithm to predict housing value
# Set targetis
y_train = np.array(train_set['median_house_value'])
y_test = np.array(test_set['median_house_value'])
X_train = train_set.copy()
X_test = test_set.copy()
X_train.drop(['median_house_value', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'ocean_proximity'], axis=1, inplace=True)
X_test.drop(['median_house_value', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'ocean_proximity'], axis=1, inplace=True)

# Run KNN
y_predicted = []
k = 5
knn(np.array(X_train), y_train, np.array(X_test), y_predicted, k)
# Compute test error
print('Test error:', loss_function(y_predicted, y_test))
