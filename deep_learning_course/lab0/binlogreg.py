import numpy as np
import data
import matplotlib.pyplot as plt

param_niter = 100
param_delta = 1e-2


def binlogreg_train(X, Y_):
    """
    Args:
        X:  podatci, np.array NxD
        Y_: indeksi razreda, np.array Nx1

    Returns:
        w, b: parametri logističke regresije
    """

    N, D = X.shape
    w, b = np.random.randn(D, 1), 0

    for i in range(param_niter):
        # klasifikacijske mjere
        scores = np.dot(X, w) + b   # N x 1

        # vjerojatnosti razreda c_1
        exp_s = np.exp(scores)
        probs = exp_s / (1 + exp_s)   # N x 1

        # gubitak
        loss = -np.sum(np.log(probs)) / N   # scalar

        if i % 10 == 0:
            print("iteration {}: loss {}".format(i, loss))

        # derivacije gubitka po klasifikacijskim mjerama
        dL_dscores = probs - (Y_ == 1).astype(int)  # N x 1

        # gradijenti parametara
        grad_w = np.transpose(np.dot(np.transpose(dL_dscores), X) / N)    # D x 1
        grad_b = np.sum(dL_dscores) / N

        w += -param_delta * grad_w
        b += -param_delta * grad_b

    return w, b


def binlogreg_classify(X, w, b):
    """
    Args:
        X: podatci, np.array NxD
        w: parametri logističke regresije
        b: parametar logističke regresije

    Returns:
        probs: vjerojatnost razreda c1
    """

    scores = np.dot(X, w) + b
    exp_s = np.exp(scores)
    probs = exp_s / (1 + exp_s)
    return probs


def binlogreg_decfun(w, b):
    return lambda X: binlogreg_classify(X, w, b)


if __name__ == "__main__":
    np.random.seed(100)

    # get the training dataset
    X, Y_ = data.sample_gauss_2d(2, 100)
    Y_ = np.reshape(Y_, (-1, 1))

    # train the model
    w, b = binlogreg_train(X, Y_)

    # evaluate the model on the training dataset
    probs = binlogreg_classify(X, w, b)
    Y = (probs >= 0.5).astype(int)

    probs = np.reshape(probs, (-1, ))
    Y = np.reshape(Y, (-1, ))
    Y_ = np.reshape(Y, (-1, ))

    # report perforamce
    accuracy, recall, precision = data.eval_perf_binary(Y, Y_)
    AP = data.eval_AP(Y_[probs.argsort()])
    print(accuracy, recall, precision, AP)

    # graph the decision surface
    decfun = binlogreg_decfun(w, b)
    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun, bbox, offset=0.5)

    # graph the data points
    data.graph_data(X, Y_, Y, special=[])

    # show the plot
    plt.show()
