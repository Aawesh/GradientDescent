#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt


class GradientDescent:
    __r_xd = np.matrix([[0.8182], [0.354]])
    __r_x = np.matrix([[1, 0.8182], [0.8182, 1]])
    __threshold = 1E-6
    __max_iteration = 1000

    def __init__(self,eta):
        self.__eta = eta

    def compute_and_display(self):
        w_old = np.matrix([[0], [0]])
        w_list = [[0], [0]]
        i = 0
        gnorm = np.inf
        while gnorm > GradientDescent.__threshold and i <= GradientDescent.__max_iteration:
            g = - GradientDescent.__r_xd.T + 0.5 * np.dot(w_old.T, (GradientDescent.__r_x + GradientDescent.__r_x.T))
            w_new = w_old - self.__eta * g.T
            w_list[0].append(w_new.tolist()[0][0])
            w_list[1].append(w_new.tolist()[1][0])
            gnorm = np.linalg.norm(g)
            w_old = w_new
            i += 1
        display(w_list,self.__eta)


def display(list,eta):
    plt.plot(list[0], label="w1")
    plt.plot(list[1], label="w2")
    plt.xlabel("Number of iteraions")
    plt.ylabel("Weights w1 and w1")
    plt.legend()
    plt.title("Trajectory of weights for eta = " + str(eta))
    plt.show()


def main():
    gradient_descent1 = GradientDescent(0.3)
    gradient_descent1.compute_and_display()

    gradient_descent2 = GradientDescent(1)
    gradient_descent2.compute_and_display()


if __name__ == "__main__":
    main()