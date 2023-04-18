import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import random

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_regression


def mutual_info_scores(X, y):
    ''' Calcule les scores d'information mutuelle de nos caractéristiques par rapport à la cible. 
        L'information mutuelle mesure la similarité entre chacune des caractéristiques et la variable 
        cible.
    '''
    mi_scores = mutual_info_regression(X, y)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores


def plot_utility_scores(scores):
    ''' Permet d'afficher les résultats de mutual_info_scores.
    '''
    y = scores.sort_values(ascending=True)
    width = np.arange(len(y))
    ticks = list(y.index)
    plt.barh(width, y)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")



def generator_batch(batch_size, X, y):
    ''' Permet de diviser les données en batches plus petits afin de mettre à jour les poids du modèle
    à chaque itération de l'entraînement.
    '''
    num_examples = len(X)
    indices = list(range(num_examples))

    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = np.array(indices[i:min(i+batch_size,num_examples)])
        yield X[batch_indices], y[batch_indices]


def labeled_barplot(data, feature, perc=False, n=None):

    total = len(data[feature])  # length of the column
    count = data[feature].nunique()
    if n is None:
        plt.figure(figsize=(count + 1, 5))
    else:
        plt.figure(figsize=(n + 1, 5))

    plt.xticks(rotation=90, fontsize=15)
    ax = sns.countplot(
        data=data,
        x=feature,
        palette="Paired",
        order=data[feature].value_counts().index[:n].sort_values(),
    )

    for p in ax.patches:
        if perc == True:
            label = "{:.1f}%".format(
                100 * p.get_height() / total
            )  # percentage of each class of the category
        else:
            label = p.get_height()  # count of each level of the category

        x = p.get_x() + p.get_width() / 2  # width of the plot
        y = p.get_height()  # height of the plot

        ax.annotate(
            label,
            (x, y),
            ha="center",
            va="center",
            size=12,
            xytext=(0, 5),
            textcoords="offset points",
        )  # annotate the percentage

    plt.show()  # show the plot


def linear_regression(X, w, b): 
    return np.dot(X, w) + b

def initialize_params(X):
    w = np.random.normal(0, 0.1, size=(X.shape[1], 1)).astype(np.float32)
    b = np.zeros(1).astype(np.float32)
    return w,b


def mean_squared_loss(y_hat, y): 
    return ((1/2)*(y_hat - y.reshape(y_hat.shape))**2).mean()

def mbgd(X,y, params, batch_size, lr=0.005): 
    y_hat = linear_regression(X, params[0], params[1])
    error = (y_hat - y.reshape(y_hat.shape))
    w_derivative = (np.dot(X.T,error)).sum()
    params[0] = params[0] - ((lr/batch_size)*w_derivative)
    params[1] = params[1] - ((lr/batch_size)*error.sum())
    return params[0],params[1]