from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from nn import NN

from utils import  Kfold, get_datasets, multi_calculate_metrics, plot_graph, std_calculate_metrics, stratifiedkfold


def main(k):
    files = get_datasets()
    for d in files:
        print("-----{} Dataset------".format(d["name"]))
        data = d["file"]
        kfold = Kfold(k, data)
        metrics =[]
        for i in kfold.foldrange:
            # print(f"For K = {i}")
            [X_train, y_train], [X_test,y_test] = kfold.get_splits(i)

            #initialize NN
            nn = NN().build_network()
            nn.fit(X_train, y_train, epochs=1000, learning_rate=0.1)
            # test
            preds = nn.predict(X_test)
            truth = np.array(y_test)

            unq = np.unique(truth)
            if len(unq)>2:
                acc, prec, rec, f1 = (multi_calculate_metrics(truth, preds, unq))
            else:
                acc, prec, rec, f1 = (std_calculate_metrics(truth, preds))
            metrics.append([acc, prec, rec, f1])
        metrics = np.sum(np.array(metrics), axis=0)/k
        print(f"Accuracy:{metrics[0]} | Precision:{metrics[1]} | Recall:{metrics[2]} | F1 Score:{metrics[3]}")
    plot_graph(n_tree, performance[0],"Accuracy Values per n_tree parameter for "+names[d]+" dataset", "n_tree", "Accuracy")
    plot_graph(n_tree, performance[1],"Precision Values per n_tree parameter "+names[d]+" dataset", "n_tree", "Precision")
    plot_graph(n_tree, performance[2],"Recall Values per n_tree parameter "+names[d]+" dataset", "n_tree", "Recall")
    plot_graph(n_tree, performance[3],"F1 score Values per n_tree parameter "+names[d]+" dataset", "n_tree", "F1 Score")




if __name__ == '__main__':
    #hyperparameters
    k = 10
    main(k)

