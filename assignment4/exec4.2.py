import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt

# Part a - Data preparation 

# Load data and plot it
data = scipy.io.loadmat('data/reg1d.mat')


X_train = data['X_train']
y_train = data['Y_train'].flatten()
X_test = data['X_test']
y_test = data['Y_test'].flatten()

plt.figure("Train data")
plt.scatter(X_train, y_train)
plt.xlabel("X train")
plt.ylabel("Y train")
plt.figure("Test data")
plt.scatter(X_test, y_test)
plt.xlabel("X test")
plt.ylabel("Y test")

# Concatenate 1 for the bias term
X_train_1 = np.insert(X_train, 0, 1.0, axis=1)

# Part b - Learning

def least_squares(X, Y):
    a = X.T.dot(X)
    b = X.T.dot(Y)
    w = np.linalg.solve(a, b)
    return w

w = least_squares(X_train_1, y_train.flatten())
p = np.poly1d(w)
y_pred = p(X_train)
plt.figure("Train data")
plt.plot(np.sort(X_train.flatten()), np.sort(y_pred.flatten()), 'r')
#plt.show()

# Part c - Evaluation

def lossL2(y, y_pred):
  err = 0
  n_points = len(y_pred)
  for i in range(n_points):
    err += np.square((y[i] - y_pred[i]))
  return 1/n_points * err 

y_pred = p(X_test).flatten()
test_error = lossL2(y_test, y_pred)
print("Average L2 loss on test data:", test_error)

# Part d - Onon-linear features

# add quadratic feature
x_square = np.power(X_train, 2)
X_train_1 = np.hstack((X_train_1, x_square))

# add cubic feature
x_cubic = np.power(X_train, 3)
X_train_1 = np.hstack((X_train_1, x_cubic))

# Re=learn
w = np.poly1d(least_squares(X_train_1, y_train)[::-1])
y_pred = w(X_test).flatten()
print("Average L2 loss on test data with added basis functions:", lossL2(y_test, y_pred))

# Plotting on train and test data

X_train = np.sort(X_train.flatten())
plt.plot(X_train, w(X_train), 'r')

X_test = np.sort(X_test.flatten())
plt.figure("Test data")
plt.plot(X_test, w(X_test), 'r')
plt.show()
# Part e - Outlier
