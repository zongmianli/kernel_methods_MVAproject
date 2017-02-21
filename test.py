import numpy as np
from multiclass_classification import *
from sklearn import preprocessing


Xtr = np.genfromtxt ('Xtr.csv', delimiter=",")
Ytr = np.genfromtxt ('Ytr.csv', delimiter=",",skip_header=1)
Ytr=Ytr[:,1]
Xtr=Xtr[:,:-1]

Xtr = preprocessing.scale(Xtr)


parameters=one_vs_all(Xtr,Ytr)
#save the trained model
'''
np.save('parameters_linearSVM_fulltrain.npy', parameters)
# Load the model
parameters = np.load('parameters_linearSVM_fulltrain.npy').item()
'''
number_of_classes=10
y_hat=predict_multiclass(Xtr,number_of_classes,parameters)
correct = np.sum(y_hat == Ytr)
acc=correct/float(len(y_hat))
print("%d out of %d predictions correct" % (correct, len(y_hat)))
print "Accuracy : ",acc




#split the data and work on a test set for evaluation
Ytr=np.array([Ytr])
data=np.concatenate((Xtr,Ytr.T),axis=1)
msk = np.random.rand(len(data)) < 0.8
train = data[msk]
test = data[~msk]
Xtrain=train[:,:-1]
Ytrain=train[:,-1]
Xtest=test[:,:-1]
Ytest=test[:,-1]
parameters=one_vs_all(Xtrain,Ytrain,kernel=gaussian_kernel,C=10)
number_of_classes=10
y_hat=predict_multiclass(Xtest,number_of_classes,parameters)
correct = np.sum(y_hat == Ytest)
acc=correct/float(len(y_hat))
print("%d out of %d predictions correct" % (correct, len(y_hat)))
print "Accuracy : ",acc

C_values=[0.0001,0.1,1,10,10000]
accuracy_list=[]
for i in C_values:
    parameters=one_vs_all(Xtrain,Ytrain,kernel=linear_kernel,C=i)
    number_of_classes=10
    y_hat=predict_multiclass(Xtest,number_of_classes,parameters)
    correct = np.sum(y_hat == Ytest)
    acc=correct/float(len(y_hat))
    print("%d out of %d predictions correct" % (correct, len(y_hat)))
    print "Accuracy : ",acc
    accuracy_list.append(acc)
