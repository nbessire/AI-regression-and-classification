"""
Project 4 - CSCI 2400 - Classification

Name: Nolan Bessire
"""
import pandas as pd
from collections import Counter
import numpy as np
# Dataset generation
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=42)

# Dataset visualization
import matplotlib.pyplot as plt
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()



def run_knn(X_train: np.array, y_train: np.array, X_test: np.array, k: int) -> np.array:
    """
    Returns np.array of classsification results for X_test
    Result will consist of a classification label for each datapoint in X_test,
    using a K-NN classification method
    """
    results = []
    #Conduct K-NN classification for each datapoint
    for test in X_test:
        kNearest = []
        #loop through training data and find the k nearest X values
        for i in range(X_train.shape[0]):
            x = X_train[i]
            y = y_train[i]
            dist = np.sqrt(np.sum((x - test)**2))
            if len(kNearest) < k:
                kNearest.append((dist, y))
            
            else:
                maxVal = 0
                #if already storing k value find the greatest stored distance
                for tup in kNearest:
                    if tup[0] > maxVal:
                        maxVal = tup[0]
                        remove = tup
                #then replace if this distance is greater than current
                if maxVal > dist:
                    kNearest.remove(remove)
                    kNearest.append((dist, y))
        counts = Counter([label for dis, label in kNearest])
        results.append(counts.most_common(1)[0][0])
    return np.array(results)

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#y_res = run_knn(X_train, y_train, X_test, 5)
#print(np.array_equal(y_res, np.array(y_test)))


def run_perceptron(X_train: np.array, y_train: np.array, X_test: np.array) -> np.array:
    """
    Returns an array of classification results using a perceptron model
    First trains the model with a maximum of 1000 training trials, then use trained
    weights for predictions
    """
    np.concatenate([np.ones((X_train.shape[0], 1)), X_train], axis=1)
    #randomize weights initially
    w = np.random.randn(X_train.shape[1])
    y_train[y_train == 0] = -1
    y_test[y_test == 0] = -1
    #max of 1000 training runs
    for i in range(1000):
        correct = True
        #run each training X through perceptron, revising weights if prediction is incorrect
        for j in range(X_train.shape[0]):
            pred = np.dot(X_train[j], w)
            pred = np.sign(pred)
            #if prediction is wrong adjust weights accordingly
            if pred != y_train[j]:
                if pred == -1:
                    w = w + X_train[j]
                else:
                    w = w - X_train[j]
                correct = False
        #if it got nothing wrong do not need to continue training
        if correct:
            break
    results = []
    #run testing Xs through perceptron and return classification
    for i in range(X_test.shape[0]):
        res = np.dot(X_test[i], w)
        results.append(np.sign(res))
    return np.array(results)

#perc = run_perceptron(X_train, y_train, X_test)

#print(np.array_equal(perc, np.array(y_test)))


def accuracy(y_true: list, y_pred: list) -> float:
    """
    return value between 0 and 1, representing level of accuracy
    accuracy is defined by number correct/total number
    """
    correct = 0
    for i in range(y_true.shape[0]):
        #ran into issue with y_true being 0 and y_pred being -1 after perceptron
        #changed if statement to also check if both are <=0, because 0 and -1
        #mean the same thing
        if y_true[i] == y_pred[i] or (y_true[i] <= 0 and y_pred[i] <= 0):
            correct += 1
    return correct / y_true.shape[0]

def precision(y_true: list, y_pred: list) -> float:
    """
    returns num between 0 and 1 representing precision of predictions
    precision defined as true positives/total predicted positives
    """
    truePos, falsePos = 0, 0
    for i in range(y_true.shape[0]):
        #if predicted to be positive check if it was actually positive
        if y_pred[i] == 1:
            if y_true[i] == 1:
                truePos += 1
            else:
                falsePos += 1
    if truePos + falsePos == 0:
        return 0
    return truePos/(truePos + falsePos)


def recall(y_true: list, y_pred: list) -> float:
    """
    return num between 0 and 1 representing recall of predictions
    recall defined as correctly predicted positives over total real positives
    returns 0 if no positive values in data set
    """
    truePos, falseNeg = 0, 0
    for i in range(y_true.shape[0]):
        #check if this is supposed to be posiitve
        if y_true[i] == 1:
            #if so check if prediction was correct and increment accordingly
            if y_pred[i] == 1:
                truePos += 1
            else:
                falseNeg += 1
    if truePos + falseNeg == 0:
        return 0
    return truePos/(truePos + falseNeg)


def run_comparisons() -> pd.DataFrame:
    """
    Returns DataFrame with accuracy, precision, and recall values for the KNN
    and perceptron algorithms when applied to a breast cancer dataset
    """
    from sklearn.datasets import load_breast_cancer
    cancer = load_breast_cancer()
    X = cancer.data
    y = cancer.target

    #split data into testing and traaining data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #run KNN and perceptron classifications (chose to use 5 nearest neighbors arbitrarily)
    knnY = run_knn(X_train, y_train, X_test, 5)
    percY = run_perceptron(X_train, y_train, X_test)
    #run both classifications through precision, accuracy, and recall tests
    knnPrec = precision(y_test, knnY)
    percPrec = precision(y_test, knnY)
    knnAcc = accuracy(y_test, knnY)
    percAcc = accuracy(y_test, percY)
    knnRec = recall(y_test, knnY)
    percRec = recall(y_test, percY)
    #construct DataFrame
    ret = {'Algorithm': ['KNN', 'Perceptron'],
           'Accuracy': [knnAcc, percAcc],
           'Precision': [knnPrec, percPrec],
           'Recall': [knnRec, percRec]}
    return pd.DataFrame(ret)
