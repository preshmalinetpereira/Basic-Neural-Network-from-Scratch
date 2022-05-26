import math
import os
from statistics import mean
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from global_variables import *
from sklearn import datasets


def std_calculate_metrics(actuals, preds, p=1):
    TP,FP,TN,FN = [],[],[],[]
    predictions = np.argmax(preds, axis=1)
    truths= np.argmax(actuals, axis=1)
    for actual,pred in zip(truths, predictions):
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
            f1_score = 1
        else:
            f1_score =0

    return accuracy, f1_score

def multi_calculate_metrics(actuals, preds):
    
    TP, FP, TN, FN = [], [], [], []
    predictions = np.argmax(preds, axis=1)
    truths= np.argmax(actuals, axis=1)
    ac = pd.Series(truths, name = 'truth')
    pr = pd.Series(predictions, name = 'pred')
    TP.append(np.sum(np.logical_and(pr == 1, ac == 1)))
    TP.append(np.sum(np.logical_and(pr == 2, ac == 2)))
    TP.append(np.sum(np.logical_and(pr == 3, ac == 3)))
    TN.append(np.sum(np.logical_and(pr == 3, ac == 3)) + np.sum(np.logical_and(pr == 2, ac == 2)))
    TN.append(np.sum(np.logical_and(pr == 3, ac == 3)) + np.sum(np.logical_and(pr == 1, ac == 1)))
    TN.append(np.sum(np.logical_and(pr == 2, ac == 2)) + np.sum(np.logical_and(pr == 1, ac == 1)))
    FP.append(np.sum(np.logical_and(pr == 1, ac == 2)) + np.sum(np.logical_and(pr == 1, ac == 3)))
    FP.append(np.sum(np.logical_and(pr == 3, ac == 1)) + np.sum(np.logical_and(pr == 3, ac == 2)))
    FP.append(np.sum(np.logical_and(pr == 2, ac == 1)) + np.sum(np.logical_and(pr == 2, ac == 3)))
    FN.append(np.sum(np.logical_and(pr == 2, ac == 1)) + np.sum(np.logical_and(pr == 3, ac == 1)))
    FN.append(np.sum(np.logical_and(pr == 3, ac == 2)) + np.sum(np.logical_and(pr == 1, ac == 2)))
    FN.append(np.sum(np.logical_and(pr == 1, ac == 3)) + np.sum(np.logical_and(pr == 2, ac == 3)))
    TP = np.sum(TP)
    TN = np.sum(TN)
    FP = np.sum(FP)
    FN = np.sum(FN)
    accuracy =(TP+TN)/(TP+FP+TN+FN)
    try:
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        f1_score = (2*(precision*recall))/(precision+recall)
    except:
        if TP ==0 and FP == 0 or FN ==0:
            f1_score = 1
        else:
            f1_score =0

    return accuracy, f1_score


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
def get_dataset(name, compute):
    # for d in datasets_dict.keys():
    file, n_features = _get_file(name, datasets_dict[name][sep],datasets_dict[name][target], datasets_dict[name][cols])
    f = {'name': name, "file": file, "n_features" : n_features}
    if type(file) !=str: 
        if compute: 
            if not os.path.isdir(os.path.join(datasetpath, "computed")): os.makedirs(os.path.join(datasetpath, "computed"))
            file.to_csv(os.path.join(datasetpath, "computed",name), ",", header=True)
    return f

def _get_file(name, sep, t, cols = "", compute=False):

    kwargs = {"names": cols} if cols !="" else {}
    label = t
    if not compute: os.path.join(datasetpath, "computed")
    if name =="digits":
        digits, y = datasets.load_digits(return_X_y=True, as_frame=True)
        file = digits.join(y, how='right')
    else:
        file=pd.read_csv(os.path.join(datasetpath, name), sep = sep, **kwargs)#names=cols)
        file.columns = file.columns.str.lower()
        file =file.dropna()
        
        if name == wine:
            file.columns = file.columns.str.strip('"')
            file = file.apply(lambda x: rep(x))
            label_column = file.pop(t)
            file.insert(len(file.columns), t, label_column)
            
            
        n_features = file[t].nunique()


        # if  name == "titanic.csv" :
        #     label_column = file.pop(target)
        #     file.insert(len(file.columns), target, label_column)
        #     file = file.drop("Name", axis=1)
        #     #file = pd.get_dummies(file, columns=["Sex"])
        #     file = categorizecolumn("Sex", file)

        # if name=="loan.csv":
        #     file = file.drop("Loan_ID", axis=1)
        #     cols = ['Gender', "Married", "Education", "Self_Employed", "Loan_Amount_Term", "Property_Area", "Loan_Status"]
        #     file = pd.get_dummies(file, columns=["Dependents"])
        #     for c in cols:
        #         file = categorizecolumn(c, file)

        # if name =="parkinsons.csv":
        #     pass
        
    return file, n_features

def rep(x):
    if type(x[0]) is str:
        x = x.str.strip("'")
        x = x.str.strip('"')
        x = np.float64(x)
    return x

def categorizecolumn(col, file):
    codes = file[col].astype('category').cat.codes
    file = file.drop(col, axis=1)
    file.insert(len(file.columns), col, codes)
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
        # X_test = test[test.columns[:-1]].to_numpy().T
        # y_test = test[test.columns[-1]].to_numpy()
        combined = []
        for j in datasets:
            if len(combined) == 0:
                combined = datasets[j]
            else:
                combined = np.concatenate((combined, datasets[j]), axis=0)
        train = pd.DataFrame(combined, columns=self.data_columns)
        # X_train = combined[test.columns[:-1]].to_numpy().T
        # y_train = combined[test.columns[-1]].to_numpy()
        
        return train, test

    
def normalize_and_split_df(train, test, n_features):
    X_train = train.iloc[:,:-n_features].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    y_train = train.iloc[:,-n_features:]
    X_test = test.iloc[:,:-n_features].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    y_test = test.iloc[:,-n_features:]
    return X_train, y_train, X_test, y_test
    # def print_cost_function():
        



    