import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

# Load training data
# We need to shufle the data first cause they are ordered

df = pd.read_csv('~/.kaggle/competitions/forest-cover-type-kernels-only/train.csv')
df_test = pd.read_csv('~/.kaggle/competitions/forest-cover-type-kernels-only/test.csv')
df = df.sample(frac=1)

# Reduce dimensionality of the data, taken from skillsmuggler https://www.kaggle.com/skillsmuggler/eda-and-dimension-reduction

def wilderness_features(df):
    df[['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4']] = df[['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4']].multiply([1, 2, 3, 4], axis=1)
    df['Wilderness_Area'] = df[['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4']].sum(axis=1)
    return df

def soil_features(df):
    soil_types = ['Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type7', 'Soil_Type8', 'Soil_Type9', \
                  'Soil_Type10', 'Soil_Type11', 'Soil_Type12', 'Soil_Type13', 'Soil_Type14', 'Soil_Type15', 'Soil_Type16', 'Soil_Type17', \
                  'Soil_Type18', 'Soil_Type19', 'Soil_Type20', 'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24', 'Soil_Type25', \
                  'Soil_Type26', 'Soil_Type27', 'Soil_Type28', 'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32', 'Soil_Type33', \
                  'Soil_Type34', 'Soil_Type35', 'Soil_Type36', 'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40']
    df[soil_types] = df[soil_types].multiply([i for i in range(1, 41)], axis=1)
    df['soil_type'] = df[soil_types].sum(axis=1)
    return df

df = wilderness_features(df)
df = soil_features(df)

df_test = wilderness_features(df_test)
df_test = soil_features(df_test)

cols = ['Elevation', 'Aspect', 'Slope',
       'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
       'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon',
       'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area',
       'soil_type']

X = df[cols].values
#X = df.drop(['Id', 'Cover_Type', 'Hillshade_Noon'], axis=1).values
y_train = df['Cover_Type'].values

# Normalize data
min_max_scaler = preprocessing.MinMaxScaler()

X_train = min_max_scaler.fit_transform(X)
X_test = min_max_scaler.fit_transform(df_test.drop(['Id', 'Hillshade_Noon'], axis=1).values)

# Apply kNN, trying multiple Ks
knn = KNeighborsClassifier(n_neighbors=3)

#Fit the model
knn.fit(X_train, y_train)

# Predict
predicted_cover = knn.predict(X_test)

# Prepare submission file
my_submission = pd.DataFrame({'Cover_Type': predicted_cover, 'Id': df_test.Id})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)
