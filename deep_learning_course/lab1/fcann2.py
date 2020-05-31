import numpy as np
import data as data
import matplotlib.pyplot as plt


def stable_softmax(x):
    exp_x_shifted = np.exp(x - np.max(x))
    return exp_x_shifted / np.sum(exp_x_shifted)


def fcann2_train(X, Y_, param_niter=50000, param_delta=0.05, step=10000, h_dim=5):

    N, D = X.shape
    C = np.max(Y_) + 1
    W_1, b_1 = np.random.randn(h_dim, D), np.zeros(h_dim)
    W_2, b_2 = np.random.randn(C, h_dim), np.zeros(C)

    for i in range(param_niter):
        S_1 = np.dot(X, W_1.T) + b_1    # N x h_dim
        H_1 = S_1 * (S_1 > 0)   # ReLU

        S_2 = np.dot(H_1, W_2.T) + b_2     # N x C

        probs = np.apply_along_axis(stable_softmax, 1, S_2)     # N x C

        loss = -1/N * np.sum(np.log(probs)[np.arange(N), Y_])

        if i % step == 0:
            print("iteration {}: loss: {}".format(i, loss))

        if i % int(param_niter / 4) == 0:
            param_delta *= 0.1

        dL_ds2 = probs     # N x C
        dL_ds2[range(N), Y_] -= 1

        grad_W2 = np.dot(dL_ds2.T, H_1)/N    # C x h_dim
        grad_b2 = np.sum(dL_ds2, axis=0)/N    # C x 1

        dL_dh1 = np.dot(dL_ds2, W_2)    # N x h_dim
        dL_ds1 = dL_dh1
        dL_ds1[S_1 < 0] = 0

        grad_W1 = np.dot(dL_ds1.T, X)  # h_dim x D
        grad_b1 = np.sum(dL_ds1, axis=0)

        W_1 = W_1 - param_delta * grad_W1
        W_2 = W_2 - param_delta * grad_W2
        b_1 += -param_delta * grad_b1
        b_2 += -param_delta * grad_b2

    return W_1, b_1, W_2, b_2


def fcann2_classify(X, W_1, b_1, W_2, b_2):
    S_1 = np.dot(X, W_1.T) + b_1
    H_1 = S_1 * (S_1 > 0)
    S_2 = np.dot(H_1, W_2.T) + b_2

    return np.apply_along_axis(stable_softmax, 1, S_2)


if __name__ == "__main__":
    np.random.seed(100)

    X, Y_ = data.sample_gmm_2d(6, 2, 10)

    W_1, b_1, W_2, b_2 = fcann2_train(X, Y_)

    probs = fcann2_classify(X, W_1, b_1, W_2, b_2)
    Y = np.argmax(probs, axis=1)

    #accuracy, recall, precision = data.eval_perf(Y, Y_)
    #print(accuracy, recall, precision)

    decfun = lambda X: fcann3_classify(X, W_1, b_1, W_2, b_2)[:, 1]
    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun, bbox, offset=0.5)

    data.graph_data(X, Y_, Y)

    plt.show()

