import numpy as np
from hmmlearn import hmm

from src.evaluation import evaluate_model_two


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


if __name__ == '__main__':
    user_to_plot = 400
    data, label = generate_data(1000, 100)
    pred = fit_model(data, np.full_like(1000, 100))
    evaluate_model_two(pred, label, 3)
