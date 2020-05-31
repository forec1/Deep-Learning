import torch
import data
import numpy as np
import matplotlib.pyplot as plt


class FCANN2(torch.nn.Module):

    def __init__(self, in_features, out_features, h):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.h = h

        self.linear1 = torch.nn.Linear(in_features, h)
        self.linear1.bias = torch.nn.Parameter(torch.zeros(h).float())
        self.linear1.weight = torch.nn.Parameter(torch.from_numpy(np.random.randn(h, in_features)).float())
        self.linear2 = torch.nn.Linear(h, out_features)
        self.linear2.bias = torch.nn.Parameter(torch.zeros(out_features).float())
        self.linear2.weight = torch.nn.Parameter(torch.from_numpy(np.random.randn(out_features, h)).float())
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        return self.linear2(x)


class Context():

    def __init__(self, model, loss, optimizer):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer


def fcann2_train(ctx, X, Y, niter):
    X = torch.tensor(X).float()
    Y = torch.tensor(Y)

    ctx.model.train()  # postavljanje modela u stanje za učenje

    for i in range(niter):
        output = ctx.model(X)  # unaprijedni prolaz
        loss = ctx.loss(output, Y).mean()  # izračun gubitka

        if i % 5000 == 0:
            print("iteration {}: loss {}".format(i, loss.item()))

        ctx.optimizer.zero_grad()  # postavljanje gradijenta u 0
        loss.backward()  # unatražni prolaz
        ctx.optimizer.step()  # primjena koraka optimizacije


def fcann2_classify(model, X):
    X = torch.tensor(X).float()
    output = model(X)
    softmax = torch.nn.Softmax(dim=1)
    return softmax(output).detach().numpy()


if __name__ == "__main__":
    np.random.seed(100)

    X, Y_ = data.sample_gmm_2d(6, 2, 10)
    model = FCANN2(2, 2, 5)

    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, weight_decay=1e-3)

    ctx = Context(model, loss, optimizer)

    fcann2_train(ctx, X, Y_, 100000)
    probs = fcann2_classify(model, X)
    Y = np.argmax(probs, axis=1)

    decfun = lambda X: fcann2_classify(model, X)[:, 1]
    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun, bbox, offset=0.5)

    data.graph_data(X, Y_, Y)

    plt.show()
