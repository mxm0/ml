import numpy as np
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
X_train_1 = np.insert(X_train, 1, 1.0, axis=1)

# Part b - Learning

def least_squares(X, Y):
    a = X.T.dot(X)
    b = X.T.dot(Y)
    w = np.linalg.solve(a, b)
    return w

m, c = least_squares(X_train_1, y_train)

y_pred = []
for x in X_test.flatten():
  y_pred.append(m * x + c)

plt.figure("Train data")
plt.plot(X_train, m * X_train + c, 'r')
#plt.show()

# Part c - Evaluation

def lossL2(y, y_pred):
  err = 0
  n_points = len(y_pred)
  for i in range(n_points):
    err += np.square((y[i] - y_pred[i]))
  return 1/n_points * err 

test_error = lossL2(y_test, y_pred)
print("Average L2 loss on test data: ", test_error)

# Part d - Onon-linear features



# Part e - Outlier
