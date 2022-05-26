from collections import defaultdict
import contextlib
from random import sample
from tkinter import S, Y
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from helper import *
from nn import *
from global_variables import *
from utils import  *
import time

def main(k, lambd, alpha, file, dataset_name, epochs, n_features, verbose=False):
    print("-----{} Dataset------".format(dataset_name))
    data = file
    kfold = Kfold(k, data)
    metricsByFold =[]
    for i in kfold.foldrange:
        train, test = kfold.get_splits(i)
        train = pd.get_dummies(train, columns=["class"])
        test = pd.get_dummies(test, columns=["class"])
        X_train, y_train, X_test, y_test = normalize_and_split_df(train, test, n_features)

        network = [X_train.shape[1], 32, n_features]

        params = init_weights(network)
        trained_params, _ = NN(X_train.values, y_train.values, params, network, lambd=lambd,n_features=n_features, alpha=alpha, epochs=epochs)
        
    #   
        preds, caches = forward(X_test.values,trained_params, n_features)

        unq = y_test.shape[1]
        truth = y_test.values
        if unq>2:
            acc, f1 = multi_calculate_metrics(truth, preds)
        else:
            acc, f1 = std_calculate_metrics(truth, preds)
        metricsByFold.append([acc, f1])
        print("Accuracy for fold-{} is {}".format(i+1, acc))
        print("F1-Score for fold-{} is {}".format(i+1, f1))
    metrics = np.sum(np.array(metricsByFold), axis=0)/k
    writetofile(f"Hyperparameters for Dataset: {dataset_name} and Network {str(network)}: k: {k} | lambda: {lambd} | alpha: {alpha} | epochs(iter): {epochs} | Accuracy:{metrics[0]} | F1 Score:{metrics[1]}")
    print(f"Average Accuracy:{metrics[0]}  | F1 Score:{metrics[1]}")


def testbackprop(lambd, n_features, alpha, network):
    
    params={}
    params["W" + str(1)] = np.array([[0.42000,  0.15000,  0.40000],
    [0.72000, 0.10000, 0.54000], [0.01000, 0.19000, 0.42000], 
    [0.30000, 0.35000, 0.68000]], dtype=np.float64) 
    params["W" + str(2)] = np.array([[0.21000, 0.67000, 0.14000, 0.96000, 0.87000],
	[0.87000, 0.42000, 0.20000, 0.32000, 0.89000],
	[0.03000, 0.56000, 0.80000, 0.69000, 0.09000]], dtype=np.float64)
    params["W" + str(3)] = np.array([[0.04000, 0.87000, 0.42000, 0.53000],  
	[0.17000, 0.10000, 0.95000, 0.69000]], dtype=np.float64)
    X = np.array([[0.32000, 0.68000], [0.83000, 0.02000]], dtype=np.float64)
    y = np.array([[0.75000, 0.98000], [0.75000, 0.28000]], dtype=np.float64)

    trained_params, cost_list = NN(X,y, params,network, lambd, n_features, alpha=alpha, epochs=1, verbose=True)

    
def cost_graph(k, lambd, alpha, file, dataset_name, epochs, n_features, network,verbose=False):
    print("-----{} Dataset------".format(dataset_name))
    data = file
    kfold = Kfold(k, data)
    cost_list = defaultdict()
    train, test = kfold.get_splits(5)
    train = pd.get_dummies(train, columns=["class"])
    test = pd.get_dummies(test, columns=["class"])
    X_train, y_train, X_test, y_test = normalize_and_split_df(train, test, n_features)
    X_train = X_train.values
    y_train = y_train.values
    X_test = X_test.values
    y_test = y_test.values
    fig, ax = plt.subplots()
    ax.set_title('Cost Graph')
    plt.xlabel("Number of Samples")
    plt.ylabel("J (cost/performance) values")
    plt.tight_layout()
    plt.ion()
    plt.show()
    prevcost = 0
    batch = 10
    for s in range(batch,X_train.shape[0],batch):
        samples = sample(range(X_train.shape[0]), s)
        mini_X = X_train[samples]
        mini_y = y_train[samples]
        params = init_weights(network)
        trained_params, _ = NN(mini_X, mini_y, params, network, lambd=lambd,n_features=n_features, alpha=alpha, epochs=epochs)

        preds, caches = forward(X_test, trained_params, n_features)
        cur_cost = cost_fn(preds, y_test, caches, lambd, verbose, train=False)
        if s != 0:            
            ax.plot([s-batch, s], [prevcost, cur_cost], marker='o', color="blue", label="cost")
            plt.pause(0.0001) # pause required to update the graph
        
        cost_list[s] = cur_cost
        prevcost = cur_cost
    plt.savefig(f"images\cost analysis_{dataset_name}_{str(time.monotonic())}.png")



if __name__ == '__main__':
    #hyperparameters
    lambd = 0.25 
    alpha=0.3
    eval = ""


    if eval == "test":
        file_path = 'backprop.txt'
        with open(file_path, "a") as o:
            with contextlib.redirect_stdout(o):
                network =  [2, 4, 3, 2]
                testbackprop(lambd,2, alpha, network) 
    
    elif eval =="cost":
        k = 10
        alpha=0.01
        epochs = 9000
        alpha=0.2
        network = [16, 16, 2]
        h = get_dataset(house_votes, False)
        file = h["file"]
        cost_graph(k, lambd, alpha, file, h["name"],epochs,h[n_features], network, False)
    else:
        k = 10
        epochs = 5000
        f = get_dataset(cancer, False)
        file = f["file"]
        main(k, lambd, alpha, file, f["name"],epochs,f[n_features], False)
