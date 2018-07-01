import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from sklearn import cluster, datasets, preprocessing, metrics
from sklearn.metrics import pairwise_distances
from sklearn import cluster, preprocessing, metrics
from sklearn import decomposition
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn import ensemble
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Load training data
# We need to shufle the data first cause they are ordered

df = pd.read_csv('~/.kaggle/competitions/forest-cover-type-kernels-only/train.csv', dtype=int)
df = df.sample(frac=1)

# Reduce dimensionality of the data taken from skillsmuggler
# Basically compress soil type and wilderness area into a single column each

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


cols = ['Elevation', 'Aspect', 'Slope',
       'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
       'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon',
       'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area',
       'soil_type', 'HF1', 'HF2', 'HR1', 'FR1', 'FR2', 'Mean_Amenities', 'slope_hyd',
       'Mean_Fire_Hyd']

#cols = ['Elevation', 'Aspect', 
#       'Wilderness_Area',
#       'soil_type', 'HF1', 'HF2', 'HR1' ,'HR2','Mean_Amenities', 'slope_hyd',
#       'Mean_Fire_Hyd']

# Create train and test data. For the moment we split the train data 80/20 to performs3# test locally without submitting the result to Kaggle.
X = df[cols].values
y = df['Cover_Type'].values

X = df.drop(['Id', 'Cover_Type'], axis=1).values
y = df['Cover_Type'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalize data

min_max_scaler = preprocessing.MinMaxScaler()
X_train = min_max_scaler.fit_transform(X_train)
X_test = min_max_scaler.fit_transform(X_test)

#################### Apply kNN, trying multiple Ks ####################
#Setup arrays to store training and test accuracies
Ks = np.arange(1,20)
train_accuracy = []
test_accuracy = []

for k in (Ks):
    #Setup a knn classifier with k neighbors
    knn = KNeighborsClassifier(n_neighbors=k)
    #Fit the model
    knn.fit(X_train, y_train)
    
    #Compute accuracy on the training set
    train_accuracy.append(knn.score(X_train, y_train))
    
    #Compute accuracy on the test set
    test_accuracy.append(knn.score(X_test, y_test)) 
print(test_accuracy)

#Generate plot
plt.title('k-NN Varying number of neighbors')
plt.plot(Ks, test_accuracy, label='Testing Accuracy')
plt.plot(Ks, train_accuracy, label='Training accuracy')
plt.legend()
plt.xticks(np.arange(1,20))
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.show()

#################### Apply SVM ####################


#################### Apply Neural Network ####################
seed = 13
np.random.seed(seed)

# define baseline model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(128, input_dim=len(cols), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

estimator = baseline_model()
# Hot one encoding of the target values
y_train_hot = np_utils.to_categorical(y_train - 1)
y_test_hot = np_utils.to_categorical(y_test - 1)
# Train model
estimator.fit(X_train, y_train_hot, epochs=100, batch_size=128)

# Predict on test data
#predicted_cover = estimator.predict(X_test[:1000])
print(estimator.evaluate(X_test, y_test_hot, batch_size = 128))

#################### Apply Extra-trees classifier ####################

n_estimators = np.arange(50, 1050, 50)
test_accuracy = []

for n_est in (n_estimators):
    estimator = ensemble.ExtraTreesClassifier(n_estimators = n_est, random_state=1)
    estimator.fit(X_train, y_train)
    #Compute accuracy on the test set
    test_accuracy.append(estimator.score(X_test, y_test))

print(test_accuracy)
plt.title('ExtraTreesClassifier')
plt.plot(n_estimators, test_accuracy, label='Testing Accuracy')
plt.legend()
plt.xticks(n_estimators)
plt.xlabel('Number of estimators')
plt.ylabel('Accuracy')
plt.show()

#################### Apply XGBoost classifier ####################
model = XGBClassifier(objective='multi:softmax', num_class=7)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(y_pred[100:])
predictions = [round(value) for value in y_pred]
print(accuracy_score(y_test, y_pred)*100)
