import torch
import torchvision
import pt_deep
import matplotlib.pyplot as plt
import data as data
import argparse
import numpy as np
import time
from sklearn.svm import SVC

dataset_root = '/tmp/mnist'  # change this to your preference
mnist_train = torchvision.datasets.MNIST(dataset_root, train=True, download=True)
mnist_test = torchvision.datasets.MNIST(dataset_root, train=False, download=True)

x_train, y_train = mnist_train.data, mnist_train.targets
x_test, y_test = mnist_test.data, mnist_test.targets
x_train, x_test = x_train.float().div_(255.0), x_test.float().div_(255.0)

N = x_train.shape[0]
D = x_train.shape[1] * x_train.shape[2]
C = y_train.max().add_(1).item()


models_conf = [[784, 10], [784, 100, 10], [784, 100, 100, 10], [784, 100, 100, 100, 10]]

parser = argparse.ArgumentParser(description='Case study: MNIST')
parser.add_argument('--model_idx', metavar='model_conf', choices=range(4), type=int)
parser.add_argument('-w', '--show_weights', action='store_true', help='Show learned weight matrices for each number')
parser.add_argument('-l', '--plot_loss', action='store_true', help='Plot loss function')
parser.add_argument('--param_niter', type=int, help='Number of iteration')
parser.add_argument('--param_delta', type=float, help='Learning rate')
parser.add_argument('--param_lambda', type=float, help='Regularization factor')
parser.add_argument('--early_stopping', '-es', action='store_true', help='Turn on early stopping')
parser.add_argument('--batch_size', type=int, default=-1)
parser.add_argument('-svm', action='store_true', help='Do SVM case study')


def show_model_weights(model):
    weights = model.weights[0].cpu().detach().numpy()
    scale = np.abs(weights).max()
    plt.figure(figsize=(10, 5))
    for i in range(weights.shape[1]):
        sp = plt.subplot(2, 5, i + 1)
        weights_i = weights[:, i].reshape((28, 28))
        sp.imshow(weights_i, cmap='gray', vmin=-scale, vmax=scale)
        sp.set_xticks(())
        sp.set_yticks(())
        sp.set_xlabel('Class %i' % i)
    plt.suptitle('Weight vectors for:')
    plt.show()


def train_mb(model, X, Y_oh_, param_niter, param_lambda, batch_size, optimizer, scheduler):
    model.train()
    N = X.shape[0]
    for epoch in range(param_niter):
        permutation = torch.randperm(N)

        for i in range(0, N, batch_size):
            optimizer.zero_grad()

            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X[indices], Y_oh_[indices]

            probs = model(batch_x)
            loss = model.get_loss(probs, batch_y, param_lambda)
            loss.backward()
            optimizer.step()
            scheduler.step()


if __name__ == "__main__":
    args = parser.parse_args()

    if not args.svm:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        model = pt_deep.PTDeep(models_conf[args.model_idx], torch.relu)
        model.to(device)

        x_train = x_train.reshape(-1, 784).to(device)
        y_train_oh = torch.nn.functional.one_hot(y_train)
        y_train_oh = y_train_oh.to(device)

        loss_trace = [] if args.plot_loss else None

        x_val, y_val_oh = None, None
        if args.early_stopping and args.batch_size == -1:
            indices = list(range(len(x_train)))
            split = int(np.floor(len(x_train)/5))
            np.random.shuffle(indices)
            train_ind, val_ind = indices[split:], indices[:split]
            x_val, y_val_oh = x_train[val_ind], y_train_oh[val_ind]
            x_train, y_train_oh, y_train = x_train[train_ind], y_train_oh[train_ind], y_train[train_ind]

        t0 = time.time()
        if args.batch_size != -1:
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1-1e-4)
            train_mb(model, x_train, y_train_oh, args.param_niter, args.param_lambda, args.batch_size, optimizer, scheduler)
        else:
            pt_deep.train(model, x_train, y_train_oh, args.param_niter, args.param_delta, trace=loss_trace,
                          param_lambda=args.param_lambda, X_val=x_val, Y_val_oh_=y_val_oh)

        x_test = x_test.reshape(-1, 784).to(device)
        probs = model(x_test)
        Y = probs.cpu().detach().numpy().argmax(axis=1)

        print('Test scores:')
        test_accuracy, recall, precision = data.eval_perf_multi(Y, y_test)
        print(test_accuracy, recall, '\n', precision)

        probs = model(x_train)
        Y = probs.cpu().detach().numpy().argmax(axis=1)

        print('Train scores:')
        train_accuracy, recall, precision = data.eval_perf_multi(Y, y_train)
        print(train_accuracy, recall, '\n', precision)

        run_time = time.time() - t0
        print('Train and eval run in %.3f s' % run_time)

        if args.model_idx == 0 and args.show_weights:
            show_model_weights(model)

        if args.plot_loss:
            plt.plot(range(len(loss_trace)), loss_trace)
            plt.suptitle('CE:{}, train accuracy:{}, test accuracy:{}, param_lambda:{}'.format(
                loss_trace[len(loss_trace)-1], train_accuracy, test_accuracy, args.param_lambda
            ))
            plt.show()

    else:
        svm_rbf = SVC(kernel='rbf', decision_function_shape='ovo')
        svm_lin = SVC(kernel='linear', decision_function_shape='ovo')

        x_train = x_train.reshape(-1, 784).detach().numpy()
        y_train = y_train.detach().numpy()

        x_test = x_test.reshape(-1, 784).detach().numpy()
        y_test = y_test.detach().numpy()

        svm_rbf.fit(x_train, y_train)
        y = svm_rbf.predict(x_train)

        print('Train scores svm_rbf:')
        train_accuracy, recall, precision = data.eval_perf_multi(y, y_train)
        print(train_accuracy, recall, '\n', precision)

        y = svm_rbf.predict(x_test)

        print('Test scores svm_rbf:')
        train_accuracy, recall, precision = data.eval_perf_multi(y, y_test)
        print(train_accuracy, recall, '\n', precision)

        svm_lin.fit(x_train, y_train)
        y = svm_lin.predict(x_train)

        print('Train scores svm_lin:')
        train_accuracy, recall, precision = data.eval_perf_multi(y, y_train)
        print(train_accuracy, recall, '\n', precision)

        y = svm_rbf.predict(x_test)

        print('Test scores svm_lin:')
        train_accuracy, recall, precision = data.eval_perf_multi(y, y_test)
        print(train_accuracy, recall, '\n', precision)


