import numpy as np
import pandas as pd
from sklearn import preprocessing
from xgboost import XGBClassifier

# Load training data
# We need to shufle the data first cause they are ordered

df = pd.read_csv('~/.kaggle/competitions/forest-cover-type-kernels-only/train.csv', dtype=int)
df_test = pd.read_csv('~/.kaggle/competitions/forest-cover-type-kernels-only/test.csv', dtype=int)
df = df.sample(frac=1)

def wilderness_feature(df):
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

df = wilderness_feature(df)
df = soil_features(df)

df_test = wilderness_feature(df_test)
df_test = soil_features(df_test)

####################### Train data #############################################
df['HF1'] = df['Horizontal_Distance_To_Hydrology']+df['Horizontal_Distance_To_Fire_Points']
df['HF2'] = abs(df['Horizontal_Distance_To_Hydrology']-df['Horizontal_Distance_To_Fire_Points'])
df['HR1'] = abs(df['Horizontal_Distance_To_Hydrology']+df['Horizontal_Distance_To_Roadways'])
df['HR2'] = abs(df['Horizontal_Distance_To_Hydrology']-df['Horizontal_Distance_To_Roadways'])
df['FR1'] = abs(df['Horizontal_Distance_To_Fire_Points']+df['Horizontal_Distance_To_Roadways'])
df['FR2'] = abs(df['Horizontal_Distance_To_Fire_Points']-df['Horizontal_Distance_To_Roadways'])
#df['ele_vert'] = df.Elevation-train.Vertical_Distance_To_Hydrology

df['slope_hyd'] = (df['Horizontal_Distance_To_Hydrology']**2+df['Vertical_Distance_To_Hydrology']**2)**0.5
df.slope_hyd=df.slope_hyd.map(lambda x: 0 if np.isinf(x) else x) # remove infinite value if any

#Mean distance to Amenities
df['Mean_Amenities']=(df.Horizontal_Distance_To_Fire_Points + df.Horizontal_Distance_To_Hydrology + df.Horizontal_Distance_To_Roadways) / 3
#Mean Distance to Fire and Water
df['Mean_Fire_Hyd']=(df.Horizontal_Distance_To_Fire_Points + df.Horizontal_Distance_To_Hydrology) / 2

####################### Test data #############################################
df_test['HF1'] = df_test['Horizontal_Distance_To_Hydrology']+df_test['Horizontal_Distance_To_Fire_Points']
df_test['HF2'] = abs(df_test['Horizontal_Distance_To_Hydrology']-df_test['Horizontal_Distance_To_Fire_Points'])
df_test['HR1'] = abs(df_test['Horizontal_Distance_To_Hydrology']+df_test['Horizontal_Distance_To_Roadways'])
df_test['HR2'] = abs(df_test['Horizontal_Distance_To_Hydrology']-df_test['Horizontal_Distance_To_Roadways'])
df_test['FR1'] = abs(df_test['Horizontal_Distance_To_Fire_Points']+df_test['Horizontal_Distance_To_Roadways'])
df_test['FR2'] = abs(df_test['Horizontal_Distance_To_Fire_Points']-df_test['Horizontal_Distance_To_Roadways'])
#df['ele_vert'] = df.Elevation-train.Vertical_Distance_To_Hydrology

df_test['slope_hyd'] = (df_test['Horizontal_Distance_To_Hydrology']**2+df_test['Vertical_Distance_To_Hydrology']**2)**0.5
df_test.slope_hyd=df_test.slope_hyd.map(lambda x: 0 if np.isinf(x) else x) # remove infinite value if any

#Mean distance to Amenities
df_test['Mean_Amenities']=(df_test.Horizontal_Distance_To_Fire_Points + df_test.Horizontal_Distance_To_Hydrology + df_test.Horizontal_Distance_To_Roadways) / 3
#Mean Distance to Fire and Water
df_test['Mean_Fire_Hyd']=(df_test.Horizontal_Distance_To_Fire_Points + df_test.Horizontal_Distance_To_Hydrology) / 2

cols = ['Elevation', 'Aspect', 'Slope',
       'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
       'Horizontal_Distance_To_Roadways',
       'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area',
       'soil_type', 'HF1', 'HF2', 'HR1', 'FR1', 'FR2', 'Mean_Amenities', 'slope_hyd',
       'Mean_Fire_Hyd']

# Create train and test data. For the moment we split the train data 80/20 to performs3# test locally without submitting the result to Kaggle.
X_train = df[cols].values
y_train = df['Cover_Type']
X_test = df_test[cols].values

# Apply Extra-trees classifier
# Apply XGBoost classifier
model = XGBClassifier(objective='multi:softmax', num_class=7)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Prepare submission file
my_submission = pd.DataFrame({'Id': df_test.Id, 'Cover_Type': predictions})
# you could use any filename. We choose submission here
my_submission.to_csv('submission_xgb.csv', index=False)
