import matplotlib.pyplot as plt
import numpy as np
import util
import math

from p05b_lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem 5(b): Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Search tau_values for the best tau (lowest MSE on the validation set)
    # Fit a LWR model with the best tau value
    x_eval, y_eval = util.load_dataset(valid_path, add_intercept=True)
    min_mse = math.inf
    best_tau = 0
    for tau in tau_values:
        lwr = LocallyWeightedLinearRegression(tau=tau)
        lwr.fit(x_train, y_train)
        y_pred = lwr.predict(x_eval)

        mse = np.mean((y_pred-y_eval)**2)
        if mse < min_mse:
            best_tau = tau
            min_mse = mse

        plt.figure()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.plot(x_train[:, -1], y_train, 'bx', linewidth=2)
        plt.plot(x_eval[:, -1], y_pred, 'ro', linewidth=2)
        plt.savefig('output/p05c_tau{}.png'.format(tau))
    # Run on the test set to get the MSE value
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)
    lwr = LocallyWeightedLinearRegression(tau=best_tau)
    lwr.fit(x_train, y_train)
    y_pred = lwr.predict(x_test)
    mse = np.mean((y_pred-y_test)**2)
    print("The mse on test set is:", mse)
    # Save predictions to pred_path
    # Plot data
    # *** END CODE HERE ***
