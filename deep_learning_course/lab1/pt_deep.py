import torch
import torch.nn as nn
import data as data
import numpy as np
import matplotlib.pyplot as plt


class PTDeep(torch.nn.Module):

    def __init__(self, net_conf, activation_fun):
        super(PTDeep, self).__init__()

        # Initialization of weights using normal distribution with mean 0.0 and std 0.1
        self.weights = nn.ParameterList([nn.Parameter(torch.empty(net_conf[i], net_conf[i+1], dtype=torch.float)
                                                      .normal_(mean=0.0, std=0.1)) for i in range(len(net_conf) - 1)])
        self.biases = nn.ParameterList([nn.Parameter(torch.zeros(net_conf[i+1], dtype=torch.float))
                                        for i in range(len(net_conf) - 1)])
        self.activation_fun = activation_fun

    def forward(self, X):
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            X = torch.mm(X, W) + b
            if i != len(self.weights) - 1:
                X = self.activation_fun(X)
        # X = X - X.max(dim=1).values.reshape(-1, 1)
        return X.softmax(dim=1)

    def get_loss(self, X, Yoh_, param_lambda=0):
        loss = -torch.log(X)[range(X.shape[0]), Yoh_.argmax(dim=1)].mean() + \
               param_lambda * torch.sum(torch.tensor([torch.sum(W**2) for _, W in enumerate(self.weights)]))
        return loss

    def count_params(self):
        cnt = 0
        for name, param in self.named_parameters():
            print(name, param.shape)
            cnt += param.numel()
        print("Number of parameters:", cnt)


def train(model, X, Yoh_, param_niter, param_delta, param_lambda=0, trace=None, X_val=None, Y_val_oh_=None):
    """Arguments:
       - X: model inputs [NxD], type: torch.Tensor
       - Yoh_: ground truth [NxC], type: torch.Tensor
       - param_niter: number of training iterations
       - param_delta: learning rate
    """

    # inicijalizacija optimizatora
    optimizer = torch.optim.SGD(model.parameters(), lr=param_delta)
    model.train()

    early_stopping = False
    if X_val is not None and Y_val_oh_ is not None:
        early_stopping = True
        best_weights, best_biases, best_i = model.weights, model.biases, 0
        prev_loss_val = np.inf

    for i in range(param_niter):
        probs = model(X)
        loss = model.get_loss(probs, Yoh_, param_lambda)

        if trace is not None:
            trace.append(loss.item())

        if i % 1000 == 0:
            print("iteration {}: loss {}".format(i, loss.item()))

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        if early_stopping:
            probs_val = model(X_val)
            loss_val = model.get_loss(probs_val, Y_val_oh_, param_lambda)
            if i % 1000 == 0:
                print("iteration {}: loss_val {}".format(i, loss_val.item()))
            if loss_val > prev_loss_val:
                break
            prev_loss_val = loss_val
            best_weights, best_biases, best_i = model.weights, model.biases, i
    if early_stopping:
        model.weights, model.biases = best_weights, best_biases
        print('Stopped in', i, 'iteration')
        print('CE_val: {}'.format(loss_val.item()))
    print("CE: {}".format(loss.item()))


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
    return probs.cpu().detach().numpy()


if __name__ == "__main__":
    # inicijaliziraj generatore slučajnih brojeva
    np.random.seed(69)

    # instanciraj podatke X i labele Yoh_
    X, Y_ = data.sample_gmm_2d(6, 2, 10)
    X, Y_ = torch.tensor(X, dtype=torch.float), torch.tensor(Y_, dtype=torch.long)
    Yoh_ = torch.nn.functional.one_hot(Y_)

    # definiraj model:
    model = PTDeep([2, 10, 10, 2], torch.relu)
    model.count_params()
    # nauči parametre (X i Yoh_ moraju biti tipa torch.Tensor):
    train(model, X, Yoh_, int(1e4), 0.1, 1e-4)
    X, Y_ = X.numpy(), Y_.numpy()

    # dohvati vjerojatnosti na skupu za učenje
    probs = eval(model, X)
    Y = np.argmax(probs, axis=1)

    # ispiši performansu (preciznost i odziv po razredima)
    accuracy, recall, precision = data.eval_perf_multi(Y, Y_)
    print(accuracy, recall, precision)


    # iscrtaj rezultate, decizijsku plohu
    decfun = lambda X: eval(model, X)[:, 1]
    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun, bbox, offset=0.5)

    data.graph_data(X, Y_, Y, special=[])

    plt.show()

