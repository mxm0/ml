import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import scipy.io
import matplotlib.pyplot as plt
from numpy.linalg import inv

train_data = scipy.io.loadmat('usps/usps_train.mat')
test_data = scipy.io.loadmat('usps/usps_test.mat')

# Part a -  Prepare train and test data for classes 2 & 3
X_train = []
X_test = []
y_train = []
y_test = []

i = 0
for label in train_data['train_label']:
    if label == 2 or label == 3:
        X_train.append(np.float64(train_data['train_data'][i]))
        y_train.append(label)
    i += 1

i = 0
for label in test_data['test_label']:
    if label == 2 or label == 3:
        X_test.append(np.float64(test_data['test_data'][i]))
        y_test.append(label)
    i += 1

X_train, y_train, X_test, y_test = np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

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

