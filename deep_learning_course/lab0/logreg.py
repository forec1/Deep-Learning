import numpy as np
import data
import matplotlib.pyplot as plt


param_niter = 50000
param_delta = 1e-2


def stable_softmax(x):
    exp_x_shifted = np.exp(x - np.max(x))
    return exp_x_shifted / np.sum(exp_x_shifted)


def logreg_train(X, Y_):
    """
    Args:
        X:  podatci, np.array NxD
        Y_: indeksi razreda, np.array Nx1

    Returns:
        W, b: parametri logističke regresije
    """

    N, D = X.shape
    C = np.max(Y_) + 1
    W, b = np.random.randn(C, D), np.zeros(C)

    for i in range(param_niter):
        scores = np.dot(X, W.T) + b     # N x C

        probs = np.apply_along_axis(stable_softmax, 1, scores)  # N X C
        logprobs = np.log(probs)    # N x C

        loss = -1/N * np.sum(logprobs[np.arange(N), Y_])

        if i % 1000 == 0:
            print("iteration {}: loss {}".format(i, loss))

        Y = np.zeros((N, C))
        Y[np.arange(N), Y_] = 1
        dL_dsT = (probs - Y).T   # C x N

        grad_W = 1/N * np.dot(dL_dsT, X)     # C x D
        grad_b = 1/N * np.sum(dL_dsT, axis=1)

        W += -param_delta * grad_W
        b += -param_delta * grad_b

    return W, b


def logreg_classify(X, W, b):
    """
    Args:
        X: podatci, np.array NxD
        W: parametri logističke regresije, np.array CxD
        b: parametar logističke regresije, np.array Cx1

    Returns:
        probs: matrica vjerojatnosti, np.array NxC
    """
    scores = np.dot(X, W.T) + b
    return np.apply_along_axis(stable_softmax, 1, scores)


def logreg_decfun(W, b):
    return lambda X: logreg_classify(X, W, b)


if __name__ == "__main__":
    np.random.seed(100)

    # get the training dataset
    X, Y_ = data.sample_gauss_2d(3, 100)

    # train the model
    W, b = logreg_train(X, Y_)

    # evaluate the model on the training dataset
    probs = logreg_classify(X, W, b)
    Y = np.argmax(probs, axis=1)

    # report perforamce
    accuracy, recall, precision = data.eval_perf_multi(Y, Y_)
    print(accuracy, recall, precision)

    # graph the decision surface
    decfun = logreg_decfun(W, b)
    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface_multi(decfun, bbox, offset=0.5)

    # graph the data points
    data.graph_data(X, Y_, Y, special=[])

    # show the plot
    plt.show()
