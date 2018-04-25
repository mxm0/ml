import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import scipy.io
import matplotlib.pyplot as plt
from numpy.linalg import inv
from sklearn.metrics import mean_squared_error

train_data = scipy.io.loadmat('usps/usps_train.mat')
test_data = scipy.io.loadmat('usps/usps_test.mat')


# Part a -  Prepare train and test data for classes 2 & 3
X_train = np.float64(train_data['train_data'][1000:3000])
X_test = np.float64(test_data['test_data'][100:300])
y_train = train_data['train_label'][1000:3000].flatten()
y_test = test_data['test_label'][100:300].flatten()

# Part b - Plotting few examples
# I flipped and rotate the matrix so it's easier to recognize the numbers

plt.figure('Number 2')
plt.imshow(np.fliplr(np.rot90(X_test[30].reshape(16, 16), 1, (1, 0))), cmap='gray')
plt.figure('Number 3')
plt.imshow(np.fliplr(np.rot90(X_test[170].reshape(16, 16), 1, (1, 0))), cmap='gray')
plt.figure('Number 3v2')
plt.imshow(np.fliplr(np.rot90(X_test[199].reshape(16, 16), 1, (1, 0))), cmap='gray')
plt.show()

#Part c - Run KNN for different k values and evaluate results
k_neighbors = [1, 3, 5, 7, 10, 15]
train_error = []
test_error  = []

for k in k_neighbors:
    y_predicted_test = []
    y_predicted_training = []

    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    y_predicted_test = knn.predict(X_test)
    y_predicted_training = knn.predict(X_train)

    train_err = mean_squared_error(y_train, y_predicted_training)
    test_err = mean_squared_error(y_test, y_predicted_test)

    print('For k =', k, ', Training error:', train_err ,'- Test error:', test_err)
    train_error.append(train_err)
    test_error.append(test_err)

plt.figure('Test/Train error for K neighbors')
plt.plot(k_neighbors, train_error, color='red', label='Train error')
plt.plot(k_neighbors, test_error, label='Test error')
plt.xticks(k_neighbors)
plt.legend()
plt.ylabel('Mean Squared Error')
plt.xlabel('K neighbors')
plt.show()

#Part d - Classify digit 3 from 8 and compare results

