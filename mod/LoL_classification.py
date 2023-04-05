from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

def get_metrics(algorithm, y_pred, y_true):
    ''' Donne certaines métriques pour le modèle choisi
    '''
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    f_score = f1_score(y_true, y_pred)

    return algorithm, accuracy, recall, precision, f_score

def find_best_k(Ks, X_train, y_train, X_test, y_test):
    '''Permet de trouver le meilleur k dans la méthode KNN'''
    Ks = 10
    mean_acc = np.zeros((Ks-1))
    for i in range(1,Ks):
        kneigh = KNeighborsClassifier(n_neighbors = i).fit(X_train, y_train)
        y_pred = kneigh.predict(X_test)
        mean_acc[i-1] = accuracy_score(y_test, y_pred)

    # Use most accurate k value to predict test values
    k = mean_acc.argmax()+1
    return k