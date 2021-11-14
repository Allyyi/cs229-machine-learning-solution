import numpy as np
import util
import matplotlib.pyplot as plt

from linear_model import LinearModel


def main(lr, train_path, eval_path, pred_path):
    """Problem 3(d): Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    # Fit a Poisson Regression model
    # Run on the validation set, and use np.savetxt to save outputs to pred_path
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=False)

    pos_reg = PoissonRegression(step_size=lr, eps=1e-5)
    pos_reg.fit(x_train, y_train)
    y_pred = pos_reg.predict(x_eval)

    # save and print
    f = open(pred_path, 'w')
    np.savetxt(f, y_pred, fmt="%f")
    f.close()

    plt.figure()
    plt.plot(y_eval, y_pred, 'bx')
    plt.xlabel('true value')
    plt.ylabel('predict value')
    plt.savefig('output/p03d.png')
    # *** END CODE HERE ***


class PoissonRegression(LinearModel):
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        m, n = x.shape
        self.theta = np.zeros(n)

        # Attention, use gradient ascent instead of stochastic gradient ascent
        while True:
            step = self.step_size * np.dot(x.T, (y - np.exp(np.dot(x, self.theta)))) / m
            theta_pre = np.copy(self.theta)
            self.theta += step

            if np.linalg.norm(theta_pre - self.theta, ord=1) < self.eps:
                break

        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Floating-point prediction for each input, shape (m,).
        """
        # *** START CODE HERE ***
        return np.exp(np.dot(x, self.theta))
        # *** END CODE HERE ***
