import numpy as np
import matplotlib.pyplot as plt

colors = ["green", "red", "blue", "yellow", "brown", "orange"]


def plot_data(x, user):
    x = x.reshape(-1, 100)[user]
    time = np.arange(0, x.shape[0])
    plt.plot(time, x)
    plt.show()
    plt.close()


def plot_data_color(input_x, y):
    fig, axes = plt.subplots(input_x.shape[1])
    for dim in range(input_x.shape[1]):
        x = input_x[:, dim]
        ax = axes[dim]
        time = np.arange(0, x.shape[0])
        ax.plot(time, x)
        last = None
        start = 0
        end = 0
        for i, color in enumerate(y):
            if last is None:
                last = color
                end = i
            elif last != color:
                color_name = colors[last]
                ax.axvspan(start, end, alpha=0.5, color=color_name)
                last = color
                start = end
                end = i + 1
            else:
                end = i

        color_name = colors[last]
        ax.axvspan(start, end, alpha=0.5, color=color_name)

    plt.show()
