import numpy as np
from sklearn import svm
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.metrics import zero_one_loss
from sklearn.model_selection import cross_val_score

data = sio.loadmat("data/cancer-data.mat")
X_train = data['cancerInput_train']
y_train = data['cancerTarget_train'].flatten()
X_test = data['cancerInput_test']
y_test = data['cancerTarget_test'].flatten()

# Part a - Linear SVM
Cs = [0.01, 0.1, 0.5, 1, 5, 10, 50]
Loss_train = []
Loss_test = []

# Train SVM for each C value and predict for both train and test values
'''
for C in Cs:
    clf = svm.SVC(kernel = 'linear', C = C)
    clf.fit(X_train, y_train)
    y_pred_train = clf.predict(X_train) 
    y_pred_test = clf.predict(X_test) 
    Loss_train.append(zero_one_loss(y_train, y_pred_train))
    Loss_test.append(zero_one_loss(y_test, y_pred_test))  

# Plot 0-1-Loss for train and test data as a function of C
plt.figure("Train error")
plt.xlabel("C value")
plt.ylabel("Error")
plt.xticks(Cs)
plt.yticks(Loss_train)
plt.plot(Cs, Loss_train, '-o')
plt.figure("Test error")
plt.xlabel("C value")
plt.ylabel("Error")
plt.xticks(Cs)
plt.yticks(Loss_test)
plt.plot(Cs, Loss_test, '-o')
plt.show()
'''

# A Large C does not seem to be affecting the training error. Not really I would have expected better results with a larger C.

# Part b - Trying different kernels and choosing C by cross validation
clf = svm.SVC(kernel = 'rbf', C = 1, gamma = 1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test) 
print(zero_one_loss(y_test, y_pred))

# Using 10-Fold cross validation and taking the mean score accuracy, the Radial basis fuction kernel performs the best. Average = 0.968530020704

# Part c - Switching train and test data
clf.fit(X_test, y_test)
y_pred = clf.predict(X_train) 
print(zero_one_loss(y_train, y_pred))

# The misclassifications are higher when we use test data for training.
