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

longitude_train, longitude_test,\
latitude_train, latitude_test,\
house_age_train, house_age_test,\
tot_rooms_train, tot_rooms_test,\
tot_bedrooms_train, tot_bedrooms_test,\
population_train, population_test,\
households_train, households_test,\
median_income_train, median_income_test,\
median_house_value_train, median_house_value_test,\
ocean_proximity_train, ocean_proximity_test = train_test_split(df['longitude'],
                                                                     df['latitude'],
                                                                     df['housing_median_age'],
                                                                     df['total_rooms'],
                                                                     df['total_bedrooms'],
                                                                     df['population'],
                                                                     df['households'],
                                                                     df['median_income'],
                                                                     df['median_house_value'],
                                                                     df['ocean_proximity'],
                                                                     test_size =0.2) 


# Recaculting max, min, avg for both training and test dataset
tot_bedrooms_train = [x for x in tot_bedrooms_train if ~np.isnan(x)]
tot_bedrooms_test = [x for x in tot_bedrooms_test if ~np.isnan(x)]

train_test_set = {'longitude':[longitude_train, longitude_test],\
                  'latitude':[latitude_train, latitude_test],\
                  'house_age':[house_age_train, house_age_test],\
                  'tot_rooms':[tot_rooms_train, tot_rooms_test],\
                  'tot_bedrooms':[tot_bedrooms_train, tot_bedrooms_test],\
                  'population':[population_train, population_test],\
                  'households':[households_train, households_test],\
                  'median_income':[median_income_train, median_income_test],\
                  'house_value':[median_house_value_train, median_house_value_test],\
                  'ocean_proximity':[ocean_proximity_train, ocean_proximity_test]}

for key in train_test_set:
    ts_set = train_test_set[key]
    train = np.array(ts_set[0])
    test = np.array(ts_set[1])
    if key != 'ocean_proximity':
        # Training data
        print('Max value for', key, 'training:', np.max(train))
        print('Min value for', key, 'training:', np.min(train))
        print('Avg value for', key, 'training:', np.average(train), '\n')
        # Test data
        print('Max value for', key, 'test:', np.max(test))
        print('Min value for', key, 'test:', np.min(test))
        print('Avg value for', key, 'test:', np.average(test), '\n')

'''
Since the mean values seem to be equal or just differ for few points, we can
assume that the distribution match. Plus plotting an histogram of the two
distribution shows it too.
'''
