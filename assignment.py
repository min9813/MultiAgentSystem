# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D


class PSO(object):

    def __init__(self, number=10, w=0.7, c1=1, c2=1):
        self.number = number

        self.group = None
        self.best_value_history = []
        self.best_history_interval = []
        self.best_individual_history = []

        self.change_history = []

        self.function_name = None
        self.n_dim = None
        self.max_iter = None

        self.w = w
        self.c1 = c1
        self.c2 = c2

    def init_state(self, initial_distribution="uniform", search_range=(-5, 5), n_dim=2):
        if initial_distribution == "uniform":
            self.group = np.random.uniform(
                search_range[0], search_range[1], self.number * n_dim)
            self.group = self.group.reshape(self.number, n_dim)

    def fit(self, function, function_name="", search_range=(-5, 5), max_iter=100, n_dim=2, plot_interval=10):
        self.init_state(search_range=search_range, n_dim=n_dim)
        individual_best = None
        individual_best_state = None
        global_best = None
        prev_best = None
        velocity = None

        self.function_name = function_name
        self.max_iter = max_iter
        self.n_dim = n_dim

        for i in tqdm(range(max_iter)):
            value = function(self.group)

            if global_best is None:
                global_best = np.min(value)
            else:
                global_best = np.min((np.min(value), global_best))
            if individual_best is None:
                individual_best_value = value.copy()
                individual_best_state = self.group.copy()

            else:
                individual_best_value = np.min(
                    np.stack([value, individual_best_value]), axis=0)
                individual_best_state[individual_best_value ==
                                      value] = self.group[individual_best_value == value]
            self.best_value_history.append(global_best)
            if prev_best != global_best:
                #                 print("yes")
                best_id = np.argmin(value)
                best_individual = self.group[best_id, :].copy()
            self.best_individual_history.append(best_id)
            self.change_history.append(prev_best != global_best)

            r1, r2 = np.random.uniform(0, 1, 2)

            new_velocity_element = self.c1 * r1 * \
                (individual_best_state - self.group) + \
                self.c2 * r2 * (best_individual - self.group)

            if velocity is None:
                velocity = new_velocity_element
            else:
                velocity = self.w * velocity + new_velocity_element

            prev_best = global_best

            self.group += velocity

            if i % plot_interval == 0:
                self.best_history_interval.append(global_best)

        print("best value:{}".format(global_best),
              "best_individual:{}".format(best_individual))

        return global_best, best_individual

    def plot_result(self):
        plt.figure()
        plt.plot(range(len(self.best_value_history)), self.best_value_history)
        plt.title("best value by each iteration")
        ax = plt.gca()
        ax.set_xscale("log")

        plt.savefig("./pso_result/{}-I{}N{}D{}_iteration".format(
            self.function_name, self.max_iter, self.number, self.n_dim))

        plt.figure()
        plt.plot(range(len(self.best_history_interval)),
                 self.best_history_interval)
        plt.title("best value by 100 iteration")
        ax = plt.gca()
        ax.set_xscale("log")
        plt.savefig("./pso_result/{}-I{}N{}D{}_interval_iteration".format(
            self.function_name, self.max_iter, self.number, self.n_dim))


class ABC(object):

    def __init__(self, number, climit=10):
        self.number = number
        self.climit = climit
        self.group = None
        self.best_value_history = []
        self.best_individual_history = []

        self.best_history_100 = []

        self.n_dim = None
        self.max_iter = None

        self.function_name = None

        self.change_history = []

    def init_state(self, initial_distribution="uniform", search_range=(-5, 5), n_dim=2):
        if initial_distribution == "uniform":
            self.group = np.random.uniform(
                search_range[0], search_range[1], self.number * n_dim)
            self.group = self.group.reshape(self.number, n_dim)

    def no_duplicate_randomize(self, array):
        tmp = array.copy()

        while np.any(tmp == array):
            np.random.shuffle(array)

        return array

    def roulett_selection(self, function, index_array):
        fit_value = function(self.group)
        fit_value = fit_value / np.sum(fit_value)
        fit_value = np.array([0] + list(fit_value)[:-1])
        fit_cumsum = np.cumsum(fit_value)
        threshould = np.random.uniform(0, 1)
        mask = fit_cumsum <= threshould
        target = np.max(index_array[mask])

        return target

    def randomize_over_limit_inidvidual(self, value, counter):
        mask = counter >= self.climit

        min_value_position = self.group[np.argmin(value)]
        max_value_position = self.group[np.argmax(value)]

        rate = np.random.uniform(0, 1)

        self.group[mask] = min_value_position + rate * \
            (max_value_position - min_value_position)

    def fit(self, function, function_name="", search_range=(-5, 5), max_iter=100, n_dim=2, plot_interval=10):
        self.init_state(search_range=search_range, n_dim=n_dim)
        global_best = None

        self.max_iter = max_iter

        self.n_dim = n_dim

        self.function_name = function_name

        counter = np.zeros(self.number)

        for i in tqdm(range(max_iter)):
            value = function(self.group)

            best_values = value.copy()

            # get best value and best individual
            best_individual_id = np.argmin(value)

            index_array = np.arange(self.number)
            shuffled_array = self.no_duplicate_randomize(index_array)
            rate = np.random.uniform(-1, 1, self.number)
            new_group = self.group + \
                rate.reshape(-1, 1) * (self.group - self.group[shuffled_array])

            new_value = function(new_group)

            better_position_mask = new_value <= value

            best_values[better_position_mask] = new_value[better_position_mask]

            counter[better_position_mask] = 0
            counter[~better_position_mask] += 1

            self.group[better_position_mask] = new_group[better_position_mask]

            target_id = self.roulett_selection(function, index_array)

            random_id = np.random.randint(self.number)

            while target_id == random_id:
                random_id = np.random.randint(self.number)

            rate = np.random.randint(-1, 1)

            new_target_position = self.group[target_id] + rate * \
                (self.group[target_id] - self.group[random_id])

            old_target_value = function(self.group[target_id].reshape(1, -1))
            new_target_value = function(new_target_position.reshape(1, -1))

#             print(old_target_value, new_target_value)

            if new_target_value <= old_target_value:
                self.group[target_id] = new_target_position
                counter[target_id] = 0
                best_values[target_id] = old_target_value
            else:
                counter[target_id] += 1

            self.randomize_over_limit_inidvidual(best_values, counter)

            best_individual_id = np.argmin(best_values)
            global_best = best_values[best_individual_id]
            self.best_individual_history.append(best_individual_id)
            self.best_value_history.append(global_best)
            if i % plot_interval == 0:
                self.best_history_100.append(global_best)

        best_position = self.group[best_individual_id]
        print("best value:{}".format(global_best),
              "best position value:", best_position)

        return global_best, best_position

    def plot_result(self):
        plt.figure()
        plt.plot(range(len(self.best_value_history)), self.best_value_history)
        plt.title("best value by each iteration")
        ax = plt.gca()
        ax.set_xscale("log")

        plt.savefig("./abc_result/{}-I{}N{}D{}_iteration".format(
            self.function_name, self.max_iter, self.number, self.n_dim))

        plt.figure()
        plt.plot(range(len(self.best_history_100)), self.best_history_100)
        plt.title("best value by 100 iteration")
        ax = plt.gca()
        ax.set_xscale("log")
        plt.savefig("./abc_result/{}-I{}N{}D{}_interval_iteration".format(
            self.function_name, self.max_iter, self.number, self.n_dim))


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


def fitting(func_dict, range_dict, algorithm, data=None, number=100, iteration=1000, n_dim=5):

    if isinstance(func_dict, dict):
        func_num = len(func_dict.keys())
        if data is None:
            data = {
                "func_name": [],
                "best_value": [],
                "best_position": [],
                "n_dim": [n_dim] * len(func_dict.keys()),
                "number": [n_dim] * len(func_dict.keys()),
                "iteration": [n_dim] * len(func_dict.keys()),
            }
        else:
            data["algorithm"].extend([algorithm] * func_num)
            data["n_dim"].extend([n_dim] * func_num)
            data["number"].extend([number] * func_num)
            data["iteration"].extend([number] * func_num)
        print("start fitting by {}".format(algorithm))
        for func_name in func_dict.keys():
            print("---start searcing best value of {}---".format(func_name))
            function = func_dict[func_name]
            if algorithm == "abc":
                model = ABC(number)
            elif algorithm == "pso":
                model = PSO(number)
            else:
                raise NotImplementedError
            best_value, best_position = model.fit(
                function, search_range=range_dict[func_name], function_name=func_name, max_iter=iteration, n_dim=n_dim)
            data["func_name"].append(func_name)
            data["best_value"].append(best_value)
            data["best_position"].append(best_position)
            model.plot_result()

        return data

    else:
        if algorithm == "abc":
            model = ABC(number)
        elif algorithm == "pso":
            model = PSO(number)
        else:
            raise NotImplementedError
        model.fit(func_dict, search_range=range_dict,
                  max_iter=iteration, n_dim=n_dim)
        model.plot_result()


parser = argparse.ArgumentParser(
    description="This file is used to train gan model")
parser.add_argument("--algorithm", help="which algorithm to fit",
                    choices=["abc", "pso"], default="abc")
parser.add_argument("--csv", help="if write to csv file", action="store_true")
parser.add_argument("--csv_out", help="file name of csv file", default="data")
parser.add_argument(
    "--size", help="number of individuals in each algoritm", type=int, default=100)
parser.add_argument("--iteration", help="iteration time",
                    type=int, default=100)
parser.add_argument("--n_dim", help="dimension", type=int, default=5)
parser.add_argument(
    "--for_report", help="if do this algorithm for report. Not recommend because it takes a long time to finish.", action="store_true")


args = parser.parse_args()

if __name__ == "__main__":
    if not os.path.exists("./abc_result"):
        os.mkdir("./abc_result")
    if not os.path.exists("./pso_result"):
        os.mkdir("./pso_result")
    func_dict = {"sphere": sphere,
                 "rastrigin": rastrigin,
                 "rosenbrock": rosenbrock,
                 "griewank": griewank,
                 "alpine": alpine,
                 "_2nminima": _2nminima
                 }

    range_dict = {"sphere": (-5, 5),
                  "rastrigin": (-5, 5),
                  "rosenbrock": (-5, 10),
                  "griewank": (-600, 600),
                  "alpine": (-10, 10),
                  "_2nminima": (-5, 5)
                  }

    data = {
        "algorithm": [],
        "func_name": [],
        "best_value": [],
        "best_position": [],
        "n_dim": [],
        "number": [],
        "iteration": [],
    }

    if args.for_report:
        for dim in [5, 10, 20]:
            print("dimension = {}".format(dim))

            data = fitting(func_dict, range_dict, algorithm="pso",
                           data=data, number=10000, iteration=1000, n_dim=dim)
            data = fitting(func_dict, range_dict, algorithm="abc",
                           data=data, number=10000, iteration=1000, n_dim=dim)

    fitting(func_dict, range_dict, algorithm=args.algorithm,
            data=data, number=args.size, iteration=args.iteration, n_dim=5)

    if args.csv:
        data = pd.DataFrame(data)
        data.to_csv("{}.csv".format(args.csv_out), index=False)
