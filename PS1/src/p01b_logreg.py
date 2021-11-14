import numpy as np
import util
import os

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    # *** START CODE HERE ***

    clf = LogisticRegression(eps=1e-5)
    clf.fit(x_train, y_train)
    print("The accuracy on training set is: ", np.mean(clf.predict(x_train) == y_train))

    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    y_pred = clf.predict(x_eval)
    print("The accuracy on validation set is: ", np.mean(y_pred == y_eval))

    # Save figure and prediction
    util.plot(x_train, y_train, clf.theta, 'output/p01b_{}.png'.format(pred_path[-5]))
    f = open(pred_path, 'w')
    np.savetxt(f, y_pred, fmt="%d")
    f.close()
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """

        # *** START CODE HERE ***
        # Init value
        m, n = x.shape
        self.theta = np.zeros(n)

        # Use Newton's method update theta
        while True:
            h_x = 1 / (1 + np.exp(-np.dot(x, self.theta)))  # (m,)
            gradient = (1/m) * np.dot(x.T, (h_x - y))  # (n,)
            hessian = (1/m) * np.dot(x.T * h_x * (1 - h_x), x)  # (n, n)

            theta_pre = np.copy(self.theta)
            self.theta -= np.dot(np.linalg.inv(hessian), gradient)  # (n,n)x(n,)=(n,)

            # Stop training when convergence
            if np.linalg.norm(self.theta-theta_pre, ord=1) < self.eps:
                break

        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        return (1 / (1 + np.exp(-np.dot(x, self.theta)))) > 0.5
        # *** END CODE HERE ***
