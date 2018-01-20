#!/usr/bin/python

import numpy as np
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

''' 
Function: transform_input()
          Reads train and test CSV file; 
          Applies one hot encoding for labels
          Returns train and test features and labels
'''


def read_input(train_file, test_file):
    
    if not train_file:
        print('Train file not found')
        sys.exit(0)
    elif not test_file:
        print('Test file not found')
        sys.exit(0)

        
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)
    
    
    X = df_train.iloc[:,1:].values
    y = df_train.iloc[:,0].values
    
    X_test = df_test.iloc[:,:].values
    
    print('Training size is %d x %d' % (X.shape[0],X.shape[1]))
    print('Test size is %d x %d' % (X_test.shape[0],X_test.shape[1]))
    
    
    return X, y, X_test


def encode_label(y):
    
    k = 10
    y_enc = np.zeros(shape=(len(y),k))
    for i in range(len(y)):
        y_enc[i,y[i]] = 1
    
    return y_enc

def transform_input(X, y, X_test):
    
    sc = StandardScaler()
    X_std = sc.fit_transform(X)
    X_test_std = sc.transform(X_test)
    
    pca = PCA(n_components = 300)
    X_transformed = pca.fit_transform(X_std)
    X_test_transformed = pca.transform(X_test_std)
    
    y_enc = encode_label(y)
    
    return X_transformed, y_enc, X_test_transformed


def soft_max(a):
    expa = np.exp(a)
    y = expa / expa.sum(axis=1, keepdims=True)
    return y


def net_input(X, w, b):   
    
    z = X.dot(w) + b
    return z

def compute_cost(p_y, t):
    error = t * np.log(p_y)
    cost = -error.sum()
    
    return cost

def gradW(y, p_y, X):
    return X.T.dot(y - p_y)

def gradb(y, p_y):
    return (y - p_y).sum(axis=0)

def forward(X, w, b):
    
    a = net_input(X, w, b)
   
    y = soft_max(a)
    
    return y
    
def error_rate(p_y, t):
    prediction = predict(p_y)
    
    return np.mean(prediction != t)    
    
def predict(p_y):
    return np.argmax(p_y,axis=1)
    

def plot_function(costs,epochs):
    
    plt.plot(range(1,epochs), costs, c='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Cost')
    plt.show()
    
    
def test_pca(train_file, test_file):
    
    # Read the input file
    X, y, X_test = read_input(train_file, test_file)
    X_transformed, y_enc, X_test_transformed  = transform_input(X, y, X_test)
    
    X_train, X_valid, y_train, y_valid = train_test_split(X_transformed, 
                                                         y_enc, 
                                                         test_size=0.3, 
                                                         random_state=0)
    
    
    print('[MAIN] Training Feature size is %d x %d' % (X_train.shape[0],X_train.shape[1]))
    print('[MAIN] Valid Feature size is %d x %d' % (X_valid.shape[0],X_valid.shape[1]))
    
    print('[MAIN] Training Label size is %d x %d' % (y_train.shape[0],y_train.shape[1]))
    print('[MAIN] Valid Label size is %d x %d' % (y_valid.shape[0],y_valid.shape[1]))
    
    
    eta = 0.00001
    epochs = 100
    reg = 0.01
    
    w = np.random.randn(X_train.shape[1],10)/ 28
    print(w)
    b = np.zeros(10)
    
    costs = []
    
    for i in range(1,epochs):
        p_y = forward(X_train, w, b)     
        
       
        cost = compute_cost(p_y, y_train)
        print('Epoch %d: %.3f'%(i, cost))
        costs.append(cost)
        
        w  += eta * (gradW(y_train,p_y, X_train) - reg*w)
        b  += eta * (gradb(y_train,p_y) - reg*b)
        

    
  

    #plot_function(costs,epochs)
    
    # Accuracy scores
    y_pred_train = forward(X_train,w,b)
    y_pred_train_enc = encode_label(np.argmax(y_pred_train,axis=1))
    #print(np.argmax(y_pred_train,axis=1))
    print('\n\n',y_pred_train_enc)
    #input()
    score_train  = accuracy_score(y_pred_train_enc, y_train)
    print(score_train)
    
    return 1


def test_norm(train_file, test_file):
    
    return 1


if __name__ == '__main__':
    
    if(len(sys.argv) != 2):
        print('Expected an argument..Please run as follows: python utils.py [option: 0:pca, 1:norm]')
        sys.exit(1)
        
        
    option = int(sys.argv[1])
    
    train_file = '../Data/train.csv'
    test_file  = '../Data/test.csv'
    
    if option == 0:
        test_pca(train_file,test_file)
    elif option == 1:
        test_norm(train_file,test_file)
    else:
        print('Incorrect option: Choose 0 or 1')
        sys.exit(1)
        
    
    print('End of program')
    sys.exit(1)
    


