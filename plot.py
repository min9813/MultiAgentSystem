# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def sphere(x_state):
    return np.sum(x_state**2, axis=1)


def rastrigin(x_state):
    dim = x_state.shape[1]
    sum_m = np.sum(x_state**2 - 10 * np.cos(2 * np.pi * x_state), axis=1)
    return 10 * dim + sum_m


def rosenbrock(x_state):
    sum_m = (x_state[:, 1:] - x_state[:, :-1]**2)**2
    sum_m = 10 * sum_m + (1 - x_state[:, :-1])**2
    sum_m = np.sum(sum_m, axis=1)

    return sum_m


def griewank(x_state):
    cos_m = np.ones_like(x_state[:, 0])
    for k in range(x_state.shape[1]):
        cos_m *= np.cos(x_state[:, k] / np.sqrt(k + 1))

    return 1 + 1. / 4000 * np.sum(x_state**2, axis=1) - cos_m


def alpine(x_state):
    res = x_state * np.sin(x_state) + 0.1 * x_state
    res = np.absolute(res)

    return np.sum(res, axis=1)


def _2nminima(x_state):
    res = x_state ** 4 - 16 * x_state**2 + 5 * x_state

    return np.sum(res, axis=1)


def plot_functions(func_dict, range_dict):

    for func_name in func_dict.keys():
        function = func_dict[func_name]
        plot_range = range_dict[func_name]
        min_lim = plot_range[0]
        max_lim = plot_range[1]
        a = np.linspace(min_lim, max_lim, 10 * (max_lim - min_lim))
        b = np.linspace(min_lim, max_lim, 10 * (max_lim - min_lim))

        X, Y = np.meshgrid(a, b)
        x_space = np.vstack([X.reshape(-1), Y.reshape(-1)]).T
        z = function(x_space)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, z.reshape(
            X.shape), cmap='bwr', linewidth=0)
        plt.savefig("./{}.png".format(func_name))


if __name__ == "__main__":
    func_dict = {"sphere": sphere,
                 "rastrigin": rastrigin,
                 "rosenbrock": rosenbrock,
                 "griewank": griewank,
                 "alpine": alpine,
                 "2nminima": _2nminima
                 }

    range_dict = {"sphere": (-5, 5),
                  "rastrigin": (-5, 5),
                  "rosenbrock": (-5, 10),
                  "griewank": (-600, 600),
                  "alpine": (-10, 10),
                  "2nminima": (-5, 5)
                  }

    plot_functions(func_dict, range_dict)
