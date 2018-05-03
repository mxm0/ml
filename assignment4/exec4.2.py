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

X_train_outlier = np.vstack((X_train, [1.05]))
y_train_outlier = np.append(y_train, -10)

plt.figure("Train data")
plt.scatter(X_train, y_train)
plt.xlabel("X train")
plt.ylabel("Y train")
plt.figure("Train data with outliers")
plt.scatter(X_train_outlier, y_train_outlier)
plt.xlabel("X train")
plt.ylabel("Y train")
plt.figure("Test data")
plt.scatter(X_test, y_test)
plt.xlabel("X test")
plt.ylabel("Y test")

# Concatenate 1 for the bias term
X_train_1 = np.insert(X_train, 0, 1.0, axis=1)
X_train_1_outlier = np.insert(X_train_outlier, 0, 1.0, axis=1)


# Part b - Learning

def least_squares(X, Y):
    a = X.T.dot(X)
    b = X.T.dot(Y)
    w = np.linalg.solve(a, b)
    return w

# Train original data
w = least_squares(X_train_1, y_train.flatten())
p = np.poly1d(w)
y_pred = p(X_train)

# Train outlier
w_1 = least_squares(X_train_1_outlier, y_train_outlier.flatten())
p_1 = np.poly1d(w_1)
y_pred_outlier = p_1(X_train_outlier)

# Plot original training data
plt.figure("Train data")
plt.plot(np.sort(X_train.flatten()), np.sort(y_pred.flatten()), 'r')

# Plot outlier training data
plt.figure("Train data with outliers")
plt.plot(np.sort(X_train_outlier.flatten()), np.sort(y_pred_outlier.flatten()), 'r')

# Part c - Evaluation

def lossL2(y, y_pred):
  err = 0
  n_points = len(y_pred)
  for i in range(n_points):
    err += np.square((y[i] - y_pred[i]))
  return 1/n_points * err 

# Prediction original test data
y_pred = p(X_test).flatten()

# Prediction outlier test data
y_pred_outlier = p_1(X_test).flatten()

# Average test error original data
test_error = lossL2(y_test, y_pred)
print("Average L2 loss on test data:", test_error)

# Average test error outlier data
test_error = lossL2(y_test, y_pred_outlier)
print("Average L2 loss on test data with outliers:", test_error)

# Part d - non-linear features

# add quadratic feature
x_square = np.power(X_train, 2)
x_square_outlier = np.power(X_train_outlier, 2)
X_train_1 = np.hstack((X_train_1, x_square))
X_train_1_outlier = np.hstack((X_train_1_outlier, x_square_outlier))

# add cubic feature
x_cubic = np.power(X_train, 3)
x_cubic_outlier = np.power(X_train_outlier, 3)
X_train_1 = np.hstack((X_train_1, x_cubic))
X_train_1_outlier = np.hstack((X_train_1_outlier, x_cubic_outlier))

# Re-learn
w = np.poly1d(least_squares(X_train_1, y_train)[::-1])
y_pred = w(X_test).flatten()
print("Average L2 loss on test data with added basis functions:", lossL2(y_test, y_pred))

# Re-learn with outliers
w_1 = np.poly1d(least_squares(X_train_1_outlier, y_train_outlier)[::-1])
y_pred_outlier = w_1(X_test).flatten()
print("Average L2 loss on test data with added basis functions and outliers:", lossL2(y_test, y_pred_outlier))



# Plotting on train and test data
plt.figure("Train data")
X_train = np.sort(X_train.flatten())
plt.plot(X_train, w(X_train), 'r')

plt.figure("Train data with outliers")
X_train_outlier = np.sort(X_train_outlier.flatten())
plt.plot(X_train_outlier, w_1(X_train_outlier), 'r')

X_test = np.sort(X_test.flatten())
plt.figure("Test data")
plt.plot(X_test, w(X_test), 'r', label="Original data")
plt.plot(X_test, w_1(X_test), 'black', label="Outlier")
plt.legend()
plt.show()

# Part e - Outlier
'''
The outliers have a really high impact on the regression.
'''
