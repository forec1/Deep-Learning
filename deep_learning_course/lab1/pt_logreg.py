import torch
import torch.nn as nn
import numpy as np
import data as data
import matplotlib.pyplot as plt


class PTLogreg(nn.Module):
    def __init__(self, D, C):
        """Arguments:
           - D: dimensions of each datapoint
           - C: number of classes
        """
        super().__init__()
        self.W = nn.Parameter(torch.randn(C, D, dtype=torch.float))
        self.b = nn.Parameter(torch.zeros(C, dtype=torch.float))

    def forward(self, X):
        s = torch.mm(X, self.W.t()) + self.b
        return s.softmax(dim=1)

    def get_loss(self, X, Yoh_, param_lambda=0):
        loss = -torch.log(X)[range(X.shape[0]), Yoh_.argmax(dim=1)].mean() + param_lambda * torch.sum(self.W**2)
        return loss


def train(model, X, Yoh_, param_niter, param_delta, param_lambda=0):
    """Arguments:
       - X: model inputs [NxD], type: torch.Tensor
       - Yoh_: ground truth [NxC], type: torch.Tensor
       - param_niter: number of training iterations
       - param_delta: learning rate
    """

    # inicijalizacija optimizatora
    optimizer = torch.optim.SGD(model.parameters(), lr=param_delta)
    model.train()

    # petlja učenja
    # ispisujte gubitak tijekom učenja
    for i in range(param_niter):
        probs = model(X)
        loss = model.get_loss(probs, Yoh_, param_lambda)

        if i % 1000 == 0:
            print("iteration {}: loss {}".format(i, loss.item()))

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()


def eval(model, X):
    """Arguments:
       - model: type: PTLogreg
       - X: actual datapoints [NxD], type: np.array
       Returns: predicted class probabilites [NxC], type: np.array
    """
    # ulaz je potrebno pretvoriti u torch.Tensor
    # izlaze je potrebno pretvoriti u numpy.array
    # koristite torch.Tensor.detach() i torch.Tensor.numpy()
    X = torch.tensor(X, dtype=torch.float)
    probs = model(X)
    return probs.detach().numpy()


if __name__ == "__main__":
    # inicijaliziraj generatore slučajnih brojeva
    np.random.seed(100)

    # instanciraj podatke X i labele Yoh_
    X, Y_ = data.sample_gauss_2d(3, 100)
    X, Y_ = torch.tensor(X, dtype=torch.float), torch.tensor(Y_, dtype=torch.long)
    Yoh_ = torch.nn.functional.one_hot(Y_)

    # definiraj model:
    ptlr = PTLogreg(X.shape[1], Yoh_.shape[1])

    # nauči parametre (X i Yoh_ moraju biti tipa torch.Tensor):
    train(ptlr, X, Yoh_, 50000, 1e-2, 1e-6)
    X, Y_ = X.numpy(), Y_.numpy()

    # dohvati vjerojatnosti na skupu za učenje
    probs = eval(ptlr, X)
    Y = np.argmax(probs, axis=1)

    # ispiši performansu (preciznost i odziv po razredima)
    accuracy, recall, precision = data.eval_perf_multi(Y, Y_)
    print(accuracy, recall, precision)

    # iscrtaj rezultate, decizijsku plohu
    decfun = lambda X: eval(ptlr, X)
    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface_multi(decfun, bbox, offset=0.5)

    data.graph_data(X, Y_, Y, special=[])

    plt.show()
