import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, train_test_split 
from collections import Counter
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

# Part a - load data
df = pd.read_csv('housing.csv')

# Part b - Print max, min and their index for each data column.
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

# Part c & d - Plotting data
'''
From the data plotted it looks like the median income, housing median age and
median house value data are the closest to a normal distribution.
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

# Part e - Splitting data set 80/20-training/test
train_set, test_set = train_test_split(df, test_size = 0.2, random_state = 42)

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
Since the mean values seem to be equal or just differ for few points, we can
assume that the distribution match. Plus plotting an histogram of the two
distribution shows it too.
'''
#kNN Algorithm to predict housing value

# Part a - Compute average error of classifier
def loss_function(y_predicted, y_expected):
    err = 0
    n_points = len(y_predicted)
    for i in range(n_points):
      err += np.square((y_expected[i] - y_predicted[i]))
    return 1/n_points * err

# Part b - Defining targets for distance function
y_train = np.array(train_set['median_house_value'])
y_test = np.array(test_set['median_house_value'])
X_train = train_set.copy()
X_test = test_set.copy()
X_train.drop(['median_income','median_house_value', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'ocean_proximity'], axis=1, inplace=True)
X_test.drop(['median_income','median_house_value', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households','ocean_proximity'], axis=1, inplace=True)

# Part c - Implement and run KNN

# Possible K-nearest neighbor implementation
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
        print('Y predicted for test point number', current_test_point, '/', test_point_n)


# Part d - predict house values

# KNN Predict using own implementation (really slow)
#knn(np.array(X_train), y_train, np.array(X_train), y_predicted_training, 7)
#knn(np.array(X_train), y_train, np.array(X_test), y_predicted_test, k)
#print('For k =', k, 'Training error:', loss_function(y_predicted_training, y_train))
# Compute test error
#print('For k =', k, 'Test error:', loss_function(y_predicted_test, y_test))

# KNN Predict using sklearn library

k_neighbors = list(range(1, 16))

for k in k_neighbors:
    y_predicted_test = []
    y_predicted_training = []

    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(np.array(X_train), y_train)
    y_predicted_test = knn.predict(np.array(X_test))
    y_predicted_training = knn.predict(np.array(X_train))

    # Compute training error and test error
    print('For k =', k, 'Training error:', loss_function(y_predicted_training, y_train), '- Test error:', loss_function(y_predicted_test, y_test))

'''
For k = 1 Training error: 1157688439.28 - Test error: 3934862427.7
For k = 2 Training error: 1294050713.2 - Test error: 3011038938.45
For k = 3 Training error: 1492765697.91 - Test error: 2755330118.68
For k = 4 Training error: 1689210720.05 - Test error: 2688112393.12
For k = 5 Training error: 1833389653.7 - Test error: 2647927295.45
For k = 6 Training error: 1960164540.69 - Test error: 2638628688.25
For k = 7 Training error: 2056890537.35 - Test error: 2601429794.2
For k = 8 Training error: 2142259040.65 - Test error: 2618539152.57
For k = 9 Training error: 2219563131.23 - Test error: 2636927550.59
For k = 10 Training error: 2282856818.53 - Test error: 2669452621.89
For k = 11 Training error: 2340523953.29 - Test error: 2701497007.42
For k = 12 Training error: 2400187154.89 - Test error: 2749068204.96
For k = 13 Training error: 2452363231.13 - Test error: 2789885265.78
For k = 14 Training error: 2495917770.92 - Test error: 2809223293.7
For k = 15 Training error: 2546859061.11 - Test error: 2829809248.24i
'''
