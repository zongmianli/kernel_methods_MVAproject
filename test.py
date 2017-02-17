import numpy as np
from multiclass_classification import *


mean1 = np.array([0, 2])
mean2 = np.array([2, 0])
cov = np.array([[0.8, 0.6], [0.6, 0.8]])
X1 = np.random.multivariate_normal(mean1, cov, 100)
y1 = np.ones(len(X1))
X2 = np.random.multivariate_normal(mean2, cov, 100)
y2 = np.zeros(len(X2))
X1_train = X1[:90]
y1_train = y1[:90]
X2_train = X2[:90]
y2_train = y2[:90]
X_train = np.vstack((X1_train, X2_train))
y_train = np.hstack((y1_train, y2_train))

parameters=one_vs_all(X_train,y_train)
#print parameters
number_of_classes=2
y_hat=predict_multiclass(X_train,number_of_classes,parameters)
correct = np.sum(y_hat == y_train)
print("%d out of %d predictions correct" % (correct, len(y_hat)))


#multivariate test
mean1 = np.array([0, 2])
cov = np.array([[0.8, 0.6], [0.6, 0.8]])
X1 = np.random.multivariate_normal(mean1, cov, 100)
y1 = np.random.randint(0,10,len(X1))
X1_train = X1[:90]
y1_train = y1[:90]
parameters=one_vs_all(X1_train,y1_train)
#print parameters
