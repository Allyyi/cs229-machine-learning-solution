import matplotlib.pyplot as plt
import numpy as np
import os

PLOT_COLORS = ['red', 'green', 'blue', 'orange']  # Colors for your plots
K = 4  # Number of Gaussians in the mixture model
NUM_TRIALS = 3  # Number of trials to run (can be adjusted for debugging)
UNLABELED = -1  # Cluster label for unlabeled data points (do not change)


def main(is_semi_supervised, trial_num):
    """Problem 3: EM for Gaussian Mixture Models (unsupervised and semi-supervised)"""
    print('Running {} EM algorithm...'
          .format('semi-supervised' if is_semi_supervised else 'unsupervised'))

    # Load dataset
    train_path = os.path.join('..', 'data', 'ds3_train.csv')
    x, z = load_gmm_dataset(train_path)
    x_tilde = None

    if is_semi_supervised:
        # Split into labeled and unlabeled examples
        labeled_idxs = (z != UNLABELED).squeeze()
        x_tilde = x[labeled_idxs, :]  # Labeled examples
        z = z[labeled_idxs, :]  # Corresponding labels
        x = x[~labeled_idxs, :]  # Unlabeled examples

    # *** START CODE HERE ***

    # (1) Initialize mu and sigma by splitting the m data points uniformly at random
    # into K groups, then calculating the sample mean and covariance for each group
    # (2) Initialize phi to place equal probability on each Gaussian
    # phi should be a numpy array of shape (K,)
    # (3) Initialize the w values to place equal probability on each Gaussian
    # w should be a numpy array of shape (m, K)

    m, _ = x.shape
    idx = np.split(np.random.permutation(m), int(m / K))
    mu = []
    sigma = []
    for i in range(K):
        x_i = x[idx[i]]
        mu_i = np.mean(x_i, axis=0)
        mu.append(mu_i)
        sigma.append((x_i - mu_i).T.dot(x_i - mu_i) / x_i.shape[0])

    phi = np.ones(K) / K
    w = np.ones((m, K)) / K
    # *** END CODE HERE ***

    if is_semi_supervised:
        w = run_semi_supervised_em(x, x_tilde, z, w, phi, mu, sigma)
    else:
        w = run_em(x, w, phi, mu, sigma)

    # Plot your predictions
    z_pred = np.zeros(m)
    if w is not None:  # Just a placeholder for the starter code
        for i in range(m):
            z_pred[i] = np.argmax(w[i])

    plot_gmm_preds(x, z_pred, is_semi_supervised, plot_id=trial_num)


def run_em(x, w, phi, mu, sigma):
    """Problem 3(d): EM Algorithm (unsupervised).

    See inline comments for instructions.

    Args:
        x: Design matrix of shape (m, n).
        w: Initial weight matrix of shape (m, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (n,).
        sigma: Initial cluster covariances, list of k arrays of shape (n, n).

    Returns:
        Updated weight matrix of shape (m, k) resulting from EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    eps = 1e-3  # Convergence threshold
    max_iter = 1000

    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        # *** START CODE HERE
        # (1) E-step: Update your estimates in w
        # (2) M-step: Update the model parameters phi, mu, and sigma
        # (3) Compute the log-likelihood of the data to check for convergence.
        # By log-likelihood, we mean `ll = sum_x[log(sum_z[p(x|z) * p(z)])]`.
        # We define convergence by the first iteration where abs(ll - prev_ll) < eps.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.
        prev_ll = ll
        for i in range(K):
            w[:, i] = gaussian(x, mu[i], sigma[i]) * phi[i]
        w /= w.sum(axis=1, keepdims=True)  # (m,k)

        phi = np.mean(w, axis=0)  # (k,)

        for i in range(K):
            mu[i] = x.T.dot(w[:, i]) / np.sum(w[:, i])
            sigma[i] = (x - mu[i]).T.dot(np.diag(w[:, i])).dot(x - mu[i]) / np.sum(w[:, i])

        p_xz = np.zeros((x.shape[0], K))
        for i in range(K):
            p_xz[:, i] = gaussian(x, mu[i], sigma[i]) * phi[i]
        ll = np.log(p_xz.sum(axis=1)).sum()
        it += 1
        # *** END CODE HERE ***
    print(f'iteration{it}')
    return w


def run_semi_supervised_em(x, x_tilde, z, w, phi, mu, sigma):
    """Problem 3(e): Semi-Supervised EM Algorithm.

    See inline comments for instructions.

    Args:
        x: Design matrix of unlabeled examples of shape (m, n).
        x_tilde: Design matrix of labeled examples of shape (m_tilde, n).
        z: Array of labels of shape (m_tilde, 1).
        w: Initial weight matrix of shape (m, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (n,).
        sigma: Initial cluster covariances, list of k arrays of shape (n, n).

    Returns:
        Updated weight matrix of shape (m, k) resulting from semi-supervised EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    alpha = 20.  # Weight for the labeled examples
    eps = 1e-3  # Convergence threshold
    max_iter = 1000
    m, n = x.shape
    m_, _ = x_tilde.shape

    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        pass  # Just a placeholder for the starter code
        # *** START CODE HERE ***
        # (1) E-step: Update your estimates in w
        # (2) M-step: Update the model parameters phi, mu, and sigma
        # (3) Compute the log-likelihood of the data to check for convergence.
        # Hint: Make sure to include alpha in your calculation of ll.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.
        prev_ll = ll
        # E-step, update w
        for i in range(K):
            w[:, i] = gaussian(x, mu[i], sigma[i]) * phi[i]
        w /= w.sum(axis=1, keepdims=True)  # (m,k)
        # M-step, update phi
        for i in range(K):
            phi[i] = w[:, i].sum(axis=0) + alpha * (z == i).sum(axis=0)
        phi /= m + alpha * m_
        # M-step, update mu and sigma
        for i in range(K):
            mu[i] = (x.T.dot(w[:, i]) + alpha * x_tilde.T.dot(np.squeeze(z==i,axis=1))) / (np.sum(w[:, i]) + alpha * np.sum(z == i))
            sigma[i] = ((x - mu[i]).T.dot(np.diag(w[:, i])).dot(x - mu[i]) + alpha * (x_tilde-mu[i]).T.dot(np.diag(np.squeeze(z==i,axis=1))).dot(x_tilde-mu[i])) / (np.sum(w[:, i]) + alpha * np.sum(z == i))
        # calculate log likelihood
        p_xz_unsup = np.zeros((x.shape[0], K))
        p_xz_sup = np.zeros((x_tilde.shape[0], K))
        for i in range(K):
            p_xz_unsup[:, i] = gaussian(x, mu[i], sigma[i]) * phi[i]
            p_xz_sup[:, i] = gaussian(x_tilde, mu[i], sigma[i]).dot(z==i) * phi[i]
        ll = np.log(p_xz_unsup.sum(axis=1)).sum() + alpha * np.log(p_xz_sup.sum(axis=1)).sum()
        it += 1
    print(f'iteration{it}')
        # *** END CODE HERE ***

    return w


# *** START CODE HERE ***
# Helper functions
def gaussian(x, mu, sigma):
    """
    :param x: (m,n)
    :param mu: (n,)
    :param sigma: (n,n)
    :return: (m,)
    """
    term = (x - mu)[:, None]
    term = -0.5 * (term @ np.linalg.pinv(sigma) @ term.transpose(0, 2, 1))[:, 0, 0]
    dia = 1 / (np.power(2 * np.pi, x.shape[1] / 2) * np.sqrt(np.linalg.det(sigma)))
    return np.exp(term) * dia
# *** END CODE HERE ***


def plot_gmm_preds(x, z, with_supervision, plot_id):
    """Plot GMM predictions on a 2D dataset `x` with labels `z`.

    Write to the output directory, including `plot_id`
    in the name, and appending 'ss' if the GMM had supervision.

    NOTE: You do not need to edit this function.
    """
    plt.figure(figsize=(12, 8))
    plt.title('{} GMM Predictions'.format('Semi-supervised' if with_supervision else 'Unsupervised'))
    plt.xlabel('x_1')
    plt.ylabel('x_2')

    for x_1, x_2, z_ in zip(x[:, 0], x[:, 1], z):
        color = 'gray' if z_ < 0 else PLOT_COLORS[int(z_)]
        alpha = 0.25 if z_ < 0 else 0.75
        plt.scatter(x_1, x_2, marker='.', c=color, alpha=alpha)

    file_name = 'p03_pred{}_{}.pdf'.format('_ss' if with_supervision else '', plot_id)
    save_path = os.path.join('output', file_name)
    plt.savefig(save_path)


def load_gmm_dataset(csv_path):
    """Load dataset for Gaussian Mixture Model (problem 3).

    Args:
         csv_path: Path to CSV file containing dataset.

    Returns:
        x: NumPy array shape (m, n)
        z: NumPy array shape (m, 1)

    NOTE: You do not need to edit this function.
    """

    # Load headers
    with open(csv_path, 'r') as csv_fh:
        headers = csv_fh.readline().strip().split(',')

    # Load features and labels
    x_cols = [i for i in range(len(headers)) if headers[i].startswith('x')]
    z_cols = [i for i in range(len(headers)) if headers[i] == 'z']

    x = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols, dtype=float)
    z = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=z_cols, dtype=float)

    if z.ndim == 1:
        z = np.expand_dims(z, axis=-1)

    return x, z


if __name__ == '__main__':
    np.random.seed(229)
    # Run NUM_TRIALS trials to see how different initializations
    # affect the final predictions with and without supervision
    for t in range(NUM_TRIALS):
        main(is_semi_supervised=False, trial_num=t)

        # *** START CODE HERE ***
        # Once you've implemented the semi-supervised version,
        # uncomment the following line.
        # You do not need to add any other lines in this code block.
        main(is_semi_supervised=True, trial_num=t)
        # *** END CODE HERE ***
