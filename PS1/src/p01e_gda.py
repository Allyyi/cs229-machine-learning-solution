import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    clf = GDA()
    clf.fit(x_train, y_train)
    print("The accuracy on training set is: ", np.mean(clf.predict(x_train) == y_train))

    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=False)
    y_pred = clf.predict(x_eval)
    print("The accuracy on validation set is: ", np.mean(y_pred == y_eval))

    # Save figure and prediction
    util.plot(x_train, y_train, clf.theta, 'output/p01e_{}.png'.format(pred_path[-5]))
    f = open(pred_path, 'w')
    np.savetxt(f, y_pred, fmt="%d")
    f.close()
    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        # Init value
        m, n = x.shape
        n1 = np.sum(y)
        self.theta = np.zeros(n+1)

        # Maximum likelihood estimation
        phi = n1 / m                                # constant
        mu_0 = np.dot(x.T, 1-y) / (m - n1)          # (n, m) * (m, 1) = (n, 1)
        mu_1 = np.dot(x.T, y) / n1                  # (n, 1)
        y_reshape = np.reshape(y, (m, -1))          # (m,) -> (m, 1)
        mu_x = (1-y_reshape) * mu_0 + y_reshape * mu_1  # pairwise product (m, n)
        sigma = np.dot((x-mu_x).T, x-mu_x) / m      # (n, n)

        sigma_inv = np.linalg.inv(sigma)            # (n, n)
        self.theta[1:] = np.dot(sigma_inv, (mu_1 - mu_0))  # (n, 1)
        self.theta[0] = 0.5 * np.dot(np.dot((mu_0 + mu_1).T, sigma_inv), (mu_0 - mu_1)) - np.log((1-phi) / phi)

        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        return (1 / (1 + np.exp(-np.dot(x, self.theta[1:]) - self.theta[0]))) > 0.5
        # *** END CODE HERE
