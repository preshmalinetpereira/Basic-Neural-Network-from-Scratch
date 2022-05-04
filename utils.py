import os
from statistics import mean
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from global_variables import datasets, sep, cols, datasetpath


# functions
def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2))

def d_mse(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size

def tanh(x):
    return np.tanh(x)

def d_tanh(x):
    return 1-np.tanh(x)**2

# calculations
def std_calculate_metrics(actuals, preds, p=1):
    TP,FP,TN,FN = [],[],[],[]
        
    for actual,pred in zip(actuals, preds):
        TP.append(actual ==p and actual == pred)
        TN.append(actual !=p and actual == pred)
        FP.append(actual !=p and actual != pred)
        FN.append(actual ==p and actual != pred)

    TP = TP.count(True)
    FP = FP.count(True)
    TN = TN.count(True)
    FN = FN.count(True)
    accuracy =(TP+TN)/(TP+FP+TN+FN)
    try:
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        f1_score = (2*(precision*recall))/(precision+recall)
    except:
        if TP ==0 and FP == 0 or FN ==0:
            precision, recall, f1_score = 1,1,1
        else:
            precision, recall, f1_score = 0,0,0

    return accuracy, precision, recall, f1_score

def multi_calculate_metrics(actuals, preds, unq):
    
    acc, pre, rec, f1 = [],[],[],[]
    for u in unq:
        accuracy, precision, recall, f1_score = std_calculate_metrics(actuals, preds, u)
        acc.append(accuracy)
        pre.append(precision)
        rec.append(recall)
        f1.append(f1_score)

    return mean(acc), mean(pre), mean(rec), mean(f1)

#plots
def plot_graph(x,y, title, x_label, y_label):
    fig = plt.figure()
    plt.ion()
    # plotting the points
    plt.plot(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    fig.show()
    plt.show()
    plt.savefig(title+".png")


# dataset operations
def get_datasets():
    files = []
    for d in datasets.keys():
        file = _get_file(d, datasets[d][sep], datasets[d][cols])
        f = {'name': d, "file": file}
        files.append(f) 
    return files
    # file1 = pd.read_csv("datasets/hw3_wine.csv", sep="\t")
    # file1 = file1.rename(columns = {'# class':'class'})
    # file1 = pd.concat([file1[file1.columns[1:]],file1["class"]],axis=1)
    # file2 = pd.read_csv("datasets/hw3_house_votes_84.csv")
    # cmc_names = {"CMC":{"W_age":"n",
    #         "W_edu":"c",
    #         "H_edu":"c","n_child":"n","W_rel":"c","W_work":"c","H_work":"c","sol":"c","media":"c","cmc":"c"
    #             }}
    # file3 = pd.read_csv("datasets/hw3_cancer.csv", sep="\t")
    # file4=pd.read_csv("datasets/cmc.data", sep = ",", names=cmc_names["CMC"].keys())
    # files = [file4] #file1
    # names = ["CMC"] #"Wine", "Congressional Votes", "Cancer"

def _get_file(name, sep, cols = "" ):

    kwargs = {"names": cols} if cols !="" else {}
    
    file=pd.read_csv(os.path.join(datasetpath, name), sep = sep, **kwargs)#names=cols)
    # else:
        # file=pd.read_csv(os.path.join("datasets", name), sep = sep)

    return file


# dataset operations
class Kfold():
    def __init__(self, k, data):
        self.k = k
        self.data_columns = data.columns
        self.foldrange = list(range(k))
        self.folds =self.stratifiedkfold(data)


    def stratifiedkfold(self, data):
        classes = list(data[data.columns[-1]].value_counts().index)
        class_splits = {}
        for i in range(len(classes)):
            c = classes[i]
            splits = np.array_split(data[data[data.columns[-1]] == c].values.tolist(), self.k)
            class_splits[c] = splits

        stratified_splits = {}
        t = 0
        for i in self.foldrange:
            combined = []
            for c in classes:
                t += len(class_splits[c][i])
                if len(combined) == 0:
                    combined = class_splits[c][i]
                else:
                    combined = np.vstack((combined, class_splits[c][i]))
            np.random.shuffle(combined)
            stratified_splits[i] = combined
        return stratified_splits

    def get_splits(self, i):

        datasets = self.folds.copy()
        test = pd.DataFrame(datasets.pop(i), columns=self.data_columns)
        X_test = test[test.columns[:-1]].to_dict(orient="records")
        y_test = test[test.columns[-1]]
        combined = []
        for j in datasets:
            if len(combined) == 0:
                combined = datasets[j]
            else:
                combined = np.concatenate((combined, datasets[j]), axis=0)

        X_train = combined[test.columns[:-1]].to_dict(orient="records")
        y_train = combined[test.columns[-1]]
        
        return [X_train, y_train], [X_test,y_test]

    


    