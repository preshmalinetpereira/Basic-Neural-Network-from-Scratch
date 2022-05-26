import math
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

def activation_forward(A_p, W, bias, verbose=False):

    Z = np.dot(A_p, W.T)
    if verbose: print("Z matrix:\n {}".format(Z), end="\n")
    A = sigmoid(Z) 
    A = np.insert(A, 0, 1, 1) if bias else A
    if verbose: print("A matrix:\n {}".format(A), end="\n")
    W = W.reshape(W.shape[0],W.shape[1]) if W.ndim > 1 else W.reshape(1, W.shape[0])
    cache = (A_p, W, Z)

    return A, cache


def forward(X, parameters, n_features, verbose=False):
    # cols = X.shape[1]
    # rows = 
    A = X                           
    caches = []                     
    L = len(parameters)
    if verbose: print("Forward propagating through vectorization\n\n")        
    A = np.insert(A, 0, 1,1) # parameters["b" + str(0)]
    if verbose: print("A :\n{}".format(A), end="\n")

    for l in range(L):
        A_p = A
        if verbose: print(f"Layer {l+1}")
        A, cache = activation_forward(A_p, parameters["W" + str(l+1)], l!=L-1, verbose)
        caches.append(cache)

    if verbose: print("F(x):\n\n {}".format(A))
    return A.reshape(X.shape[0],n_features), caches


def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    return A


def backward(AL, y, lambd, caches, verbose=False):
    if verbose: print("Running backpropagation through vectorization\n\n") 
    n = len(y)
    L = len(caches)
    # grads = {}
    deltas = {}
    D ={}
    P={}
    avg_grads={}
    if verbose: print("Computing vectorized gradients\n\n")
    #get deltas
    deltas[str(L-1)] = np.array(AL - y)
    if verbose: print("Delta_{}\n\n: {}".format(L+1, deltas[str(L-1)]))
    for l in range (L-1, 0, -1):
        current_cache = caches[l]
        
        W = current_cache[1][:,1:] 
        A = current_cache[0][:,1:]
        
        deltas[str(l-1)] = np.multiply(np.multiply(np.dot(deltas[str(l)], W), A),(1-A))
        if verbose: print("Delta_{}\n\n: {}".format(l+1, deltas[str(l-1)]), end="\n")


    if verbose: # for test run
        for l in range(L-1, -1, -1):
            current_cache = caches[l]
            D[str(l)] = 0
            for d in range(len(current_cache[0])):
                curr_delta = np.multiply(deltas[str(l)][d][:,None], current_cache[0][d][:,None].T)
                D[str(l)] = D[str(l)] + curr_delta
                if verbose: print("Gradients of Theta_{} for instance {}\n: {}".format(l+1,d+1, curr_delta), end="\n")
        print("The entire training set has been processes. Computing the average (regularized) gradients\n\n")
        for l in range(L):
            current_cache = caches[l]
            P[l]= np.multiply(lambd, current_cache[1])
            P[l][:,0] = 0 
            avg_grads[str(l+1)] = (1/n) * (np.add(D[str(l)],P[l]))
            if verbose: print("Final regularized gradients of Theta_{}\n\n: {}".format(l+1, avg_grads[str(l+1)]), end="\n")
        
    #faster computation rechecked and analyzed
    
    for l in range(L-1, -1, -1):
        current_cache = caches[l]
        D[str(l)] = np.dot(deltas[str(l)].T, current_cache[0])
        P[l]= np.multiply(lambd, current_cache[1])
        P[l][:,0] = 0
        avg_grads[str(l+1)] = (1/n) * (np.add(D[str(l)],P[l]))

    return avg_grads

def update_parameters(parameters, grads, alpha, verbose=False):
    L = len(parameters) 
    
    for l in range(1, L+1):
        
        parameters["W" + str(l)] = parameters["W" + str(l)] - np.multiply(alpha,grads[str(l)])
        # if verbose: print("Final regularized gradients of Theta_{}: {}".format(l, parameters["W" + str(l)]))
            
    return parameters


def init_weights(dims):
    np.random.seed(1)               
    params = {}
    L = len(dims)            

    for l in range(1, L):           
        params["W" + str(l)] = np.random.randn(dims[l], dims[l-1]+1) * 0.3

    return params

def cost_fn(y_pred, y_true, caches, lambd, verbose=False, train=True):
    cost = 0
    L = len(y_true)
    if verbose: print("Predicted output : {}\n".format(y_pred))
    if verbose: print("Expected output  : {}\n".format(y_true))

	
	
    for i, (y, y_hat) in enumerate(zip(y_true, y_pred)):
        curr_cost = -np.multiply(y, np.log(y_hat)) - np.multiply((1-y), np.log(1 - y_hat))
        if verbose: print("Cost, J, associated for instance {}: {}".format(i+1,curr_cost))
        cost+=np.sum(curr_cost)

        # if cost < 0:
        #     print("fx : {} \ty : {} \tcost : {} \tcost_curr: {}".format(y_hat, y, cost, np.sum(curr_cost)))

    cost /= len(y_true)
    if train:
        S=0
        for cache in caches:
                S+=np.sum(np.power(cache[1], 2))
        S = (lambd/(2*(len(y_true))))*S

        cost +=S
    if verbose: print("Final (regularized) cost, J, based on the complete training set  : {}\n\n".format(cost))

    return cost

def writetofile(text):
    with open('output.txt', 'a') as output:
        output.write(str(text)+"\n\n")
        