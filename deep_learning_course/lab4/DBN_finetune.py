import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim

from torchvision import datasets, transforms
import tqdm
from torchvision.utils import make_grid

import torch.distributions as tdist

import numpy as np

import matplotlib.pyplot as plt
import pickle

BATCH_SIZE = 100
EPOCHS = 100
VISIBLE_SIZE = 784
HIDDEN_SIZE = 100


def visualize_RBM_weights(weights, grid_width, grid_height, slice_shape=(28, 28)):
    for idx in range(0, grid_width * grid_height):
        plt.subplot(grid_height, grid_width, idx+1)
        plt.imshow(weights[..., idx].reshape(slice_shape))
        plt.axis('off')


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def draw_rec(inp, title, size, Nrows, in_a_row, j):
    plt.subplot(Nrows, in_a_row, j)
    plt.imshow(inp.reshape(size), vmin=0, vmax=1, interpolation="nearest")
    plt.title(title)
    plt.axis('off')


def reconstruct(ind, states, orig, weights, biases, h1_shape=(10, 10), v_shape=(28,28)):
    j = 1
    in_a_row = 6
    Nimg = states.shape[1] + 3
    Nrows = int(np.ceil(float(Nimg+2)/in_a_row))

    plt.figure(figsize=(12, 2*Nrows))

    draw_rec(states[ind], 'states', h1_shape, Nrows, in_a_row, j)
    j += 1
    draw_rec(orig[ind], 'input', v_shape, Nrows, in_a_row, j)

    reconstr = biases.copy()
    j += 1
    draw_rec(sigmoid(reconstr), 'biases', v_shape, Nrows, in_a_row, j)

    for i in range(h1_shape[0] * h1_shape[1]):
        if states[ind,i] > 0:
            j += 1
            reconstr = reconstr + weights[:,i]
            titl = '+= s' + str(i+1)
            draw_rec(sigmoid(reconstr), titl, v_shape, Nrows, in_a_row, j)
    plt.tight_layout()


train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./files', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor()
                               ])), batch_size=BATCH_SIZE, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./files', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor()
                               ])), batch_size=BATCH_SIZE)


class DBNWithFineTuning():

    def __init__(self, base_dbn, cd_k=1):
        self.v_size = base_dbn.v_size
        self.h1_size = base_dbn.h1_size
        self.h2_size = base_dbn.h2_size
        self.cd_k = cd_k

        normal_dist = tdist.Normal(0, 0.1)

        self.R1 = base_dbn.W1.clone()  # v x h1
        self.W1_down = base_dbn.W1.T.clone()  # h1 x v
        self.v1_bias = base_dbn.v_bias.clone()
        self.h1_up_bias = base_dbn.h1_bias.clone()
        self.h1_down_bias = base_dbn.h1_bias.clone()

        self.W2 = base_dbn.W2.clone()  # h1 x h2
        self.h2_bias = base_dbn.h2_bias.clone()

    def forward(self, batch, steps=None):
        batch = batch.view(-1, 784)

        h1_up_prob = torch.sigmoid(torch.mm(batch, self.R1) + self.h1_up_bias)  # * x h1
        h1_up = h1_up_prob.bernoulli()

        v1_up_down_prob = torch.sigmoid(torch.mm(h1_up, self.W1_down) + self.v1_bias)
        v1_up_down = v1_up_down_prob.bernoulli()

        h2_up_prob = torch.sigmoid(torch.mm(h1_up, self.W2) + self.h2_bias)
        h2_up = h2_up_prob.bernoulli()

        h1_down_prob, h1_down, h2_down_prob, h2_down = self.gibbs_sampling(h2_up, steps=steps)

        v1_down_prob = torch.sigmoid(torch.mm(h1_down, self.W1_down) + self.v1_bias)
        v1_down = v1_down_prob.bernoulli()

        h1_down_up_prob = torch.sigmoid(torch.mm(v1_down, self.R1) + self.h1_up_bias)
        h1_down_up = h1_down_up_prob.bernoulli()

        return h1_up_prob, h1_up, v1_up_down_prob, v1_up_down, h2_up_prob, h2_up, h1_down_prob, h1_down, h2_down_prob, h2_down, v1_down_prob, v1_down, h1_down_up_prob, h1_down_up

    def gibbs_sampling(self, h2, steps=None):
        h2_down = h2

        steps_to_do = self.cd_k

        if steps is not None:
            steps_to_do = steps

        for step in range(0, steps_to_do):
            h1_down_prob = torch.sigmoid(torch.mm(h2_down, self.W2.T) + self.h1_down_bias)
            h1_down = h1_down_prob.bernoulli()

            h2_down_prob = torch.sigmoid(torch.mm(h1_down, self.W2) + self.h2_bias)
            h2_down = h2_down_prob.bernoulli()

        return h1_down_prob, h1_down, h2_down_prob, h2_down

    def reconstruct(self, h2, steps=None):
        h1_down_prob, h1_down, h2_down_prob, h2down = self.gibbs_sampling(h2, steps)

        v_out_tmp_prob = torch.sigmoid(torch.mm(h2down, self.W2.T))  # * x h1
        v_out_tmp = v_out_tmp_prob.bernoulli()
        v_out_prob = torch.sigmoid(torch.mm(v_out_tmp, self.W1_down))  # * x v
        v_out = v_out_prob.bernoulli()

        return v_out_prob, v_out, h2_down_prob, h2down

    def update_weights_for_batch(self, batch, learning_rate=0.01):
        h1_up_prob, h1_up, v1_up_down_prob, v1_up_down, h2_up_prob, h2_up, h1_down_prob, h1_down, h2_down_prob, h2_down, v1_down_prob, v1_down, h1_down_up_prob, h1_down_up = self.forward(batch)

        v0 = batch.view(-1, 784)

        self.W1_down = self.W1_down + learning_rate * torch.mm(h1_up.T, (v0 - v1_up_down))
        self.R1 = self.R1 + learning_rate * torch.mm(v1_down.T, (h1_down - h1_down_up))

        self.v1_bias = self.v1_bias + learning_rate * torch.mean(v0 - v1_up_down, dim=0)

        self.h1_down_bias = self.h1_down_bias + learning_rate * torch.mean(h1_up - h1_down, dim=0)
        self.h1_up_bias = self.h1_up_bias +learning_rate * torch.mean(h1_down - h1_down_up, dim=0)

        w2_positive_grad = torch.mm(h1_up.T, h2_up)
        w2_negative_grad = torch.mm(h1_down.T, h2_down)

        dw2 = (w2_positive_grad - w2_negative_grad) / h1_up.shape[0]

        self.W2 = self.W2 + learning_rate * dw2
        self.h2_bias = self.h2_bias + learning_rate * torch.mean(h2_up - h2_down, dim=0)

    def __call__(self, batch):
        return self.forward(batch)


dbnmodel = pickle.load(open('dbn.obj', 'rb'))
model = pickle.load(open('rbm.obj', 'rb'))
dbnmodel_ft = DBNWithFineTuning(dbnmodel, cd_k=2)
for curr_epoch in tqdm.tqdm(range(0, EPOCHS)):
    for sample, label in train_loader:
        sample = sample.view(-1, 784)
        dbnmodel_ft.update_weights_for_batch(sample, 0.01)

plt.figure(figsize=(12, 12), facecolor='w')
visualize_RBM_weights(dbnmodel_ft.R1.data, 10, 10)
plt.tight_layout()


plt.figure(figsize=(12, 12), facecolor='w')
visualize_RBM_weights(dbnmodel_ft.W1_down.T.data, 10, 10)
plt.tight_layout()

difference = torch.abs(dbnmodel_ft.R1.data - dbnmodel_ft.W1_down.T.data)
plt.figure(figsize=(12, 12), facecolor='w')
visualize_RBM_weights(difference, 10, 10)
plt.tight_layout()

sample, _ = next(iter(test_loader))
sample = sample.view(-1, 784)

for idx in range(0, 20):
    # rbn reconstruct
    _, _, _, _, recon1, _ = model(sample[idx, ...])

    # dbn reconstruct
    _, _, _, _, _, _, _, h2down = dbnmodel.forward(sample[idx, ...])
    recon2, _, _, _ = dbnmodel.reconstruct(h2down)

    # dbn fine tune reconstruct
    _, _, _, _, _, _, _, _, _, h2_down, _, _, _, _ = dbnmodel_ft(sample[idx, ...])
    recon3, _, _, _ = dbnmodel_ft.reconstruct(h2_down, 2)

    plt.figure(figsize=(5*3, 3))
    plt.subplot(1, 5, 1)
    plt.imshow(sample[idx, ...].view(28, 28))
    if idx == 0:
        plt.title("Original image")

    plt.subplot(1, 5, 2)
    plt.imshow(recon1.view(28, 28))
    if idx == 0:
        plt.title("Reconstruction 1")

    plt.subplot(1, 5, 3)
    plt.imshow(recon2.view(28, 28))
    if idx == 0:
        plt.title("Reconstruction 2")

    plt.subplot(1, 5, 4)
    plt.imshow(recon3.view(28, 28))
    if idx == 0:
        plt.title("Reconstruction 3")

    plt.subplot(1, 5, 5)
    plt.imshow(h2_down.view(10, 10))
    if idx == 0:
        plt.title("Top state 3")

r_input = np.random.rand(100, 100)
r_input[r_input > 0.9] = 1
r_input[r_input < 1] = 0

s = 10
i = 0
r_input[i,:] = 0
r_input[i,i]= s
i += 1
r_input[i,:] = 0
r_input[i,i]= s
i += 1
r_input[i,:] = 0
r_input[i,i]= s
i += 1
r_input[i,:] = 0
r_input[i,i]= s
i += 1
r_input[i,:] = 0
r_input[i,i]= s
i += 1
r_input[i,:] = 0
r_input[i,i]= s
i += 1
r_input[i,:] = 0
r_input[i,i]= s


v_out_prob, v_out, h2_down_prob, h2down = dbnmodel_ft.reconstruct(torch.from_numpy(r_input).float(), 100)

plt.figure(figsize=(16, 16))
for idx in range(0, 19):
    plt.figure(figsize=(14, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(r_input[idx, ...].reshape(10, 10))
    if idx == 0:
        plt.title("Set state")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(h2down[idx, ...].view(10, 10))
    if idx == 0:
        plt.title("Final state")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(v_out_prob[idx, ...].view(28, 28))
    if idx == 0:
        plt.title("Reconstruction")
    plt.axis('off')

