
import numpy as np
from svm import *

def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p

def gaussian_kernel(x, y, sigma=5.0):
    return np.exp(-linalg.norm(x-y)**2 / (2 * (sigma ** 2)))


def one_vs_all(X,y):
    number_of_classes=len(np.unique(y))
    #matrix containing the results of each classifier for each example
    decision_function=np.zeros((X.shape[0],number_of_classes))
    #matrix containing all the parameters
    parameters={}

    for i in range(number_of_classes):
        y_binary=np.array([1 if label == i else -1 for label in y])
        svm = SVM()
        svm.fit(X,y_binary)
        #get the parameters for each svm
        parameters[i]={}
        parameters[i]['w']=svm.w
        parameters[i]['a']=svm.a
        parameters[i]['sv']=svm.sv
        parameters[i]['sv_y']=svm.sv_y
        parameters[i]['b']=svm.b
        parameters[i]['kernel']=svm.kernel

    return parameters

def predict_multiclass(X,number_of_classes,parameters):
    #matrix containing the results of each classifier for each example
    decision_function=np.absolute(np.zeros((X.shape[0],number_of_classes)))

    for i in range(number_of_classes):
        if parameters[i]['w'] is not None:
            decision_function[:,i]=np.dot(X,parameters[i]['w'])+parameters[i]['b']
        else:
            y_predict=np.zeros(len(X))
            for j in range(len(X)):
                s=0
                for a, sv_y, sv in zip(parameters[i]['a'],parameters[i]['sv_y'],parameters[i]['sv']):
                    s += a * sv_y * parameters[i]['kernel'](X[j], sv)
                y_predict[j] = s
            decision_function[:,i] = np.absolute(y_predict + parameters[i]['b'])

    y_hat=np.argmax(decision_function,axis=1)
    return y_hat
