import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

    np.random.random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = np.array(indices[i:min(i+batch_size,num_examples)])
        yield X[batch_indices], y[batch_indices]



