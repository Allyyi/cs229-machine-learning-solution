import matplotlib.pyplot as plt
import numpy as np
import util

from p01e_gda import GDA
from p01b_logreg import LogisticRegression


def plot_2(x, y, legend1=None, theta1=None, legend2=None, theta2=None, save_path=None, correction=1.0):
    # Plot dataset
    plt.figure()
    plt.plot(x[y == 1, -2], x[y == 1, -1], 'bx', linewidth=2)
    plt.plot(x[y == 0, -2], x[y == 0, -1], 'go', linewidth=2)

    # Plot decision boundary (found by solving for theta^T x = 0)
    x1 = np.arange(min(x[:, -2]), max(x[:, -2]), 0.01)
    x2 = -(theta1[0] / theta1[2] * correction + theta1[1] / theta1[2] * x1)
    plt.plot(x1, x2, c='red', label=legend1, linewidth=2)

    x1 = np.arange(min(x[:, -2]), max(x[:, -2]), 0.01)
    x2 = -(theta2[0] / theta2[2] * correction + theta2[1] / theta2[2] * x1)
    plt.plot(x1, x2, c='black', label=legend2, linewidth=2)

    # Add labels and save to disk
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend(loc="upper right")
    if save_path is not None:
        plt.savefig(save_path)


train_path = '../data/ds2_train.csv'
eval_path = '../data/ds2_valid.csv'
x_train, y_train = util.load_dataset(train_path, add_intercept=False)
x_eval, y_eval = util.load_dataset(eval_path, add_intercept=False)

gda = GDA()
gda.fit(x_train, y_train)

x_train, y_train = util.load_dataset(train_path, add_intercept=True)
x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
log = LogisticRegression()
log.fit(x_train, y_train)

plot_2(x_train, y_train, legend1="GDA", theta1=gda.theta,
       legend2="Logistic Regression", theta2=log.theta, save_path="output/p01_g.png")

