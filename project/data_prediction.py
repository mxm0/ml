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
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

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
cols = ['Elevation', 'Aspect', 'Slope',
       'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
       'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon',
       'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area',
       'soil_type']

# Create train and test data. For the moment we split the train data 80/20 to performs3# test locally without submitting the result to Kaggle.
X = df[cols].values
y = df['Cover_Type'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalize data
min_max_scaler = preprocessing.MinMaxScaler()

X_train = min_max_scaler.fit_transform(X_train)
X_test = min_max_scaler.fit_transform(X_test)

'''
# Apply kNN, trying multiple Ks
#Setup arrays to store training and test accuracies
Ks = np.arange(1,10)
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
'''

# Apply SVM

# Apply Neural Network
seed = 13
np.random.seed(seed)



# define baseline model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(8, input_dim=12, activation='relu'))
	model.add(Dense(7, activation='softmax'))
	# Compile model
	model.compile(loss='', optimizer='adam', metrics=['accuracy'])
	return model

predictor = KerasClassifier(build_fn = baseline_model, epochs=200, batch_size = 5, verbose = 0)

# Convert integers to one hot encoded
y = np_utils.to_categorical(y_train)

kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(predictor, X_train, y_train , cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
