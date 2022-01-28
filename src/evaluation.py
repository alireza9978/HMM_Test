from itertools import permutations

import numpy as np
from sklearn.metrics import accuracy_score


def evaluate_model(y_pred, y_true, class_count):
    result_map = {}
    for i in range(class_count):
        a = y_true == i
        b = y_pred[a]
        result = np.unique(b, return_counts=True)
        result_map[i] = result[0][result[1].argmax()]
    y_pred = np.vectorize(result_map.get)(y_pred)
    print(accuracy_score(y_true, y_pred))


def evaluate_model_two(y_pred, y_true, class_count):
    perms = permutations(range(class_count))
    results = []
    for permutation in perms:
        my_map = {}
        for i in range(class_count):
            my_map[i] = permutation[i]
        temp_y_pred = np.vectorize(my_map.get)(y_pred)
        results.append(accuracy_score(y_true, temp_y_pred))
    print(max(results))
