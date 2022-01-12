import matplotlib.pyplot as plt
import numpy as np
from hmmlearn import hmm


def plot_data(x):
    time = np.arange(0, x.shape[0])
    plt.plot(time, x)
    plt.show()


def plot_data_color(x, y):
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
    x = np.array(x)
    return x


def fit_model(x, length):
    model = hmm.GaussianHMM(n_components=3, covariance_type='diag', tol=0.0001, n_iter=1000)
    model.fit(x, length)
    print("iterations = ", model.monitor_.iter)
    y = model.predict(x)
    y = np.array(y)
    return y


if __name__ == '__main__':
    data = generate_data(100, 100)
    user_data = data.reshape(-1, 100)
    plot_data(user_data[4])
    pred = fit_model(data, np.full_like(100, 100))
    user_pred = pred.reshape(-1, 100)
    plot_data_color(user_data[4], user_pred[4])
