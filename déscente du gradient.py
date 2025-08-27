# -*- coding: utf-8 -*-

from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
X,y=make_blobs(n_samples=100,n_features=2, centers=2,random_state=0)
y=y.reshape((y.shape[0], 1))

print('dimension de X:', X.shape)
print('dimension de y:', y.shape)

plt.scatter(X[:,0],X[:,1],c=y,cmap='summer')
plt.show()

def initialisation(X):
    W = np.random.randn(X.shape[1], 1)
    b = np.random.randn(1)
    return (W, b)

def model(X,W,b):
    z=X.dot(W)+b
    A=1/(1+np.exp(-z))
    return A

def log_loss(A,y):
    return 1/len(y)*np.sum(-y*np.log(A)-(1-y)*np.log(1-A))

def gradients(A,X,y):
    dw=1/len(y) * np.dot(X.T,A-y)
    db = 1/len(y) * np.sum(A-y)
    return (dw,db)

def update(dw, db,W, b, learning_rate):
    W=W-learning_rate*dw
    b=b-learning_rate * db
    return (W,b)


def predict(X,W,b):
    A=model(X, W, b)
    return A>=0.5


def artificial_neuron(X,y, learning_rate=0.1, n_iter=100):
    #initialisation W, b
    W , b =initialisation(X)
    
    Loss= []
    
    for i in range(n_iter):
        A= model(X,W,b)
        Loss.append(log_loss(A, y))
        dW , db = gradients(A,X,y)
        W, b = update(dW, db,W,b,learning_rate)
        
    y_pred=predict(X,W,b)
    print(accuracy_score(y,y_pred))
    
    plt.plot(Loss)
    plt.show()

artificial_neuron(X,y)
