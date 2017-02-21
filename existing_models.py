import numpy as np
from multiclass_classification import *
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.decomposition import KernelPCA

Xtr = np.genfromtxt ('Xtr.csv', delimiter=",")
Ytr = np.genfromtxt ('Ytr.csv', delimiter=",",skip_header=1)
Ytr=Ytr[:,1]
Xtr=Xtr[:,:-1]
Xtr = preprocessing.scale(Xtr)

kpca = KernelPCA(n_components=1024,kernel="rbf")
X_kpca = kpca.fit_transform(X)



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

C_values=[50,200,500,5000]
gamma_values=[1e-3,0.1,0.5,2,10]
acc_train_list=[]
acc_test_list=[]
for i in C_values:
    clf = SVC(C=i,kernel='rbf',verbose=False,gamma='auto')
    clf.fit(Xtrain, Ytrain)
    y_hat=clf.predict(Xtest)
    correct = np.sum(y_hat == Ytest)
    acc=correct/float(len(y_hat))
    acc_test_list.append(acc)

    y_hat_train=clf.predict(Xtrain)
    correct_train = np.sum(y_hat_train == Ytrain)
    acc_train=correct_train/float(len(y_hat_train))
    acc_train_list.append(acc_train)

    print 'accuracy on test' , acc
    print 'accuracy on train' , acc_train
