import numpy as np
import util

from p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    # *** START CODE HERE ***
    # Part (c): Train and test on true labels
    # Make sure to save outputs to pred_path_c
    # Prepare dataset
    x_train, y_train = util.load_dataset(train_path, label_col='t', add_intercept=True)
    x_test, y_test = util.load_dataset(valid_path, label_col='t', add_intercept=True)

    # Logistic regression using all true label
    log_reg = LogisticRegression(eps=1e-5)
    log_reg.fit(x_train, y_train)
    y_test_pred = log_reg.predict(x_test)

    # save and print
    f = open(pred_path_c, 'w')
    np.savetxt(f, y_test_pred, fmt="%d")
    f.close()
    print("The accuracy on test set is: ", np.mean(y_test_pred == y_test))
    util.plot(x_train, y_train, log_reg.theta, 'output/p02_c.png')

    # Part (d): Train on y-labels and test on true labels
    # Make sure to save outputs to pred_path_d
    # Prepare dataset
    x_train, y_train = util.load_dataset(train_path, label_col='y', add_intercept=True)
    x_test, y_test = util.load_dataset(valid_path, label_col='t', add_intercept=True)

    # retrain classifier using y label
    log_reg.fit(x_train, y_train)
    y_test_pred = log_reg.predict(x_test)

    # save and print
    f = open(pred_path_c, 'w')
    np.savetxt(f, y_test_pred, fmt="%d")
    f.close()
    print("The accuracy on test set is: ", np.mean(y_test_pred == y_test))
    util.plot(x_test, y_test, log_reg.theta, 'output/p02_d.png')

    # Part (e): Apply correction factor using validation set and test on true labels
    # Plot and use np.savetxt to save outputs to pred_path_e
    # estimate alpha using validation set
    x_eval, y_eval = util.load_dataset(valid_path, label_col='y', add_intercept=True)
    h_eval = 1 / (1 + np.exp(-np.dot(x_eval[y_eval == 1], log_reg.theta)))
    alpha = np.sum(h_eval) / np.sum(y_eval)

    # correct predicted y using alpha to get t
    theta_correct = log_reg.theta + np.log(2/alpha -1 ) * np.array([1, 0, 0]).T
    t_test_pred = (1 / (1 + np.exp(-np.dot(x_test, theta_correct)))) > 0.5

    # save and print
    f = open(pred_path_c, 'w')
    np.savetxt(f, y_test_pred, fmt="%d")
    f.close()
    print("The accuracy on test set is: ", np.mean(t_test_pred == y_test))
    util.plot(x_test, y_test, theta_correct, 'output/p02_e.png')

    # *** END CODER HERE
