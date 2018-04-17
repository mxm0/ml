import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 

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
train_set, test_set = train_test_split(df, test_size = 0.2)
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
