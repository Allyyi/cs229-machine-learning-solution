import matplotlib.pyplot as plt
import numpy as np
import util

from linear_model import LinearModel


def main(tau, train_path, eval_path):
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    # Fit a LWR model
    lwr = LocallyWeightedLinearRegression(tau=tau)
    lwr.fit(x_train, y_train)

    # Get MSE value on the validation set
    y_pred = lwr.predict(x_eval)
    mse = np.mean((y_pred - y_eval)**2)
    print("The mean squared value is: ", mse)

    # Plot validation predictions on top of training set
    # No need to save predictions
    # Plot data
    plt.figure()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(x_train[:, -1], y_train, 'bx', linewidth=2)
    plt.plot(x_eval[:, -1], y_pred, 'ro', linewidth=2)
    plt.savefig('output/p05b.png')

    # *** END CODE HERE ***


class LocallyWeightedLinearRegression(LinearModel):
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        # *** START CODE HERE ***
        self.x = x
        self.y = y
        # *** END CODE HERE ***

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        m, n = x.shape
        y_pred = np.zeros(m)

        for i in range(m):
            w = np.diag(np.exp(-np.linalg.norm(self.x - x[i], ord=2, axis=1) ** 2 / (2 * self.tau ** 2)))
            self.theta = np.linalg.inv(self.x.T.dot(w).dot(self.x)).dot(self.x.T).dot(w).dot(self.y)
            y_pred[i] = self.theta.dot(x[i])

        return y_pred

        # *** END CODE HERE ***
