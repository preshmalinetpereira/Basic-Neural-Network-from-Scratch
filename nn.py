from matplotlib import pyplot as plt
import numpy as np

from helper import *


def NN(X, y, params, network, lambd, n_features, alpha=0.01, epochs=1, verbose=False):
    if verbose: 
        print("Regularization parameter lambda={}\n".format(lambd))
        print(f"Initializing the network with the following structure (number of neurons per layer): [{str(network)}]\n\n")
        print("Training Set\n\n")
        for i in range(len(params)):
            print(f"Initial Theta{str(i+1)} (the weights of each neuron, including the bias weight, are stored in the rows):", end="\n")
            print(params["W" + str(i+1)], end="\n")

        print("Training instance Features", end="\n")
        print(X, end="\n")
        print("Training instance Labels", end="\n")
        print(y)
        print("----------------------------------------------------------------------")
        print("Computing the error/cost, J, of the network")
    cost_list = []
    for i in range(epochs):

        AL, caches = forward(X, params, n_features, verbose)

        cost = cost_fn(AL, y, caches, lambd, verbose)

        if verbose: print("----------------------------------------------------------------------")

        avg_grads = backward(AL, y, lambd, caches, verbose)

        params = update_parameters(params, avg_grads, alpha, verbose)

        # print(f"Cost per epoch: {cost}")
        cost_list.append(cost)

    return params, cost_list

