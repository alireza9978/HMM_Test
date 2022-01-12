import matplotlib.pyplot as plt
import numpy as np
from hmmlearn import hmm
from sklearn.metrics import accuracy_score
from itertools import permutations


def plot_data(x, user):
    x = x.reshape(-1, 100)[user]
    time = np.arange(0, x.shape[0])
    plt.plot(time, x)
    plt.show()


def plot_data_color(x, y, user):
    x = x.reshape(-1, 100)[user]
    y = y.reshape(-1, 100)[user]
    fig, ax = plt.subplots()
    time = np.arange(0, x.shape[0])
    ax.plot(time, x)
    last = None
    start = 0
    end = 0
    for i, color in enumerate(y):
        if last is not None:
            if last != color:
                color_name = "white"
                if color == 0:
                    color_name = "green"
                elif color == 1:
                    color_name = "red"
                elif color == 2:
                    color_name = "blue"
                ax.axvspan(start, end, alpha=0.5, color=color_name)
                last = color
                start = i
                end = start + 1
            else:
                end = i + 1
        else:
            last = color

    color = y[-1]
    color_name = "white"
    if color == 0:
        color_name = "green"
    elif color == 1:
        color_name = "red"
    elif color == 2:
        color_name = "blue"
    ax.axvspan(start, end, alpha=0.5, color=color_name)
    plt.show()


def generate_data(length, dim):
    x = []
    y = []
    transmat = np.array([[0.9, 0.95, 1],
                         [0.05, 0.95, 1],
                         [0.05, 0.1, 1]])

    for j in range(dim):
        last = 0
        for i in range(length):
            transition = np.random.random(1)
            transition_prob = transmat[last]
            if transition < transition_prob[0]:
                value = np.random.normal(2, 0.5, 1)
                last = 0
            elif transition < transition_prob[1]:
                value = np.random.normal(5, 0.5, 1)
                last = 1
            else:
                value = np.random.normal(8, 0.5, 1)
                last = 2
            x.append(value)
            y.append(last)
    x = np.array(x)
    y = np.array(y)
    return x, y


def fit_model(x, length):
    model = hmm.GaussianHMM(n_components=3, covariance_type='diag', tol=0.0001, n_iter=1000)
    model.fit(x, length)
    print("iterations = ", model.monitor_.iter)
    y = model.predict(x)
    y = np.array(y)
    return y


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


if __name__ == '__main__':
    user_to_plot = 400
    data, label = generate_data(1000, 100)
    # plot_data(data, user_to_plot)
    pred = fit_model(data, np.full_like(1000, 100))
    # plot_data_color(data, pred, user_to_plot)
    # evaluate_model(pred, label, 3)
    evaluate_model_two(pred, label, 3)
