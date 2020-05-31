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


class RBM():

    def __init__(self, visible_size, hidden_size, cd_k=1):
        self.v_size = visible_size
        self.h_size = hidden_size
        self.cd_k = cd_k

        normal_dist = tdist.Normal(0, 0.1)

        self.W = torch.Tensor(normal_dist.sample(sample_shape=(self.v_size, self.h_size)))
        self.v_bias = torch.Tensor(torch.zeros(self.v_size))
        self.h_bias = torch.Tensor(torch.zeros(self.h_size))

    def forward(self, batch):
        return self._cd_pass(batch)

    def __call__(self, batch):
        return self.forward(batch)

    def _cd_pass(self, batch):
        batch = batch.view(-1, 784)
        h0_prob = torch.sigmoid(torch.mm(batch, self.W) + self.h_bias)  # * x h
        h0 = h0_prob.bernoulli()  # * x h

        h1 = h0

        for step in range(0, self.cd_k):
            v1_prob = torch.sigmoid(torch.mm(h1, self.W.T) + self.v_bias)  # * x v
            v1 = v1_prob.bernoulli()  # * x v
            h1_prob = torch.sigmoid(torch.mm(v1, self.W) + self.h_bias)  # * x h
            h1 = h1_prob.bernoulli()  # * x h

        return h0_prob, h0, h1_prob, h1, v1_prob, v1

    def reconstruct(self, h, gibbs_steps=None):
        h1 = h

        steps_to_do = self.cd_k
        if gibbs_steps is not None:
            steps_to_do = gibbs_steps

        for step in range(0, steps_to_do):
            v1_prob = torch.sigmoid(torch.mm(h, self.W.T) + self.v_bias)  # * x v
            v1 = v1_prob.bernoulli()  # * x v
            h1_prob = torch.sigmoid(torch.mm(v1, self.W) + self.h_bias)  # * x h
            h1 = h1_prob.bernoulli()  # * x h

        return h1_prob, h1, v1_prob, v1

    def update_weights_for_batch(self, batch, learning_rate=0.01):
        h0_prob, h0, h1_prob, h1, v1_prob, v1 = self._cd_pass(batch)
        v0 = batch.view(-1, 784)  # * x v
        w_positive_grad = torch.mm(v0.T, h0)  # v x h
        w_negative_grad = torch.mm(v1.T, h1)  # v x h

        dw = (w_positive_grad - w_negative_grad) / batch.shape[0]  # v x h

        self.W = self.W + learning_rate * dw  # v x h
        self.v_bias = self.v_bias + learning_rate * torch.mean(v0 - v1, dim=0)
        self.h_bias = self.h_bias + learning_rate * torch.mean(h0 - h1, dim=0)


model = RBM(visible_size=VISIBLE_SIZE, hidden_size=HIDDEN_SIZE, cd_k=1)
for curr_epoch in tqdm.tqdm(range(0, EPOCHS)):
    for sample, label in train_loader:
        sample = sample.view(-1, 784)
        model.update_weights_for_batch(sample, 0.1)

pickle.dump(model, open('rbm.obj', 'w'))

plt.figure(figsize=(12, 12), facecolor='w')
visualize_RBM_weights(model.W.data, 10, 10)


sample, _ = next(iter(test_loader))
sample = sample.view(-1, 784)

for idx in range(0, 20):
    h0_prob, h0, h1_prob, h1, v1_prob, v1 = model(sample)

    plt.figure(figsize=(8, 4), facecolor='w')
    plt.subplot(1, 3, 1)
    plt.imshow(sample[idx, ...].view(28, 28).cpu())
    if idx == 0:
        plt.title("Original image")

    plt.subplot(1, 3, 2)
    recon_image = v1_prob[idx, ...].view(28, 28)
    plt.imshow(recon_image.cpu().data)
    if idx == 0:
        plt.title("Reconstruction")

    plt.subplot(1, 3, 3)
    state_image = h1[idx, ...].view(10, 10)
    plt.imshow(state_image.cpu().data)
    if idx == 0:
        plt.title("Hidden state")

sample, _ = next(iter(test_loader))
sample = sample[0, ...].view(-1, 784)

h0_prob, h0, h1_prob, h1, v1_prob, v1 = model(sample)

reconstruct(0, h1.numpy(), sample.numpy(), model.W.numpy(), model.v_bias.numpy())


sample, _ = next(iter(test_loader))
sample = sample.view(-1, 784)

h0_prob, h0, h1_prob, h1, v1_prob, v1 = model(sample)

h0_prob, h0, h1_prob, h1, v1_prob, v1, model_weights, model_v_biases = list(map(lambda x: x.numpy(), [h0_prob, h0, h1_prob, h1, v1_prob, v1, model.W, model.v_bias]))


# Vjerojatnost da je skriveno stanje uključeno kroz Nu ulaznih uzoraka
plt.figure(figsize=(9, 4))
tmp = (h1.sum(0)/h1.shape[0]).reshape((10, 10))
plt.imshow(tmp, vmin=0, vmax=1, interpolation="nearest")
plt.axis('off')
plt.colorbar()
plt.title('vjerojatnosti (ucestalosti) aktivacije pojedinih neurona skrivenog sloja')

# Vizualizacija težina sortitranih prema učestalosti
plt.figure(figsize=(16, 16))
tmp_ind = (-tmp).argsort(None)
visualize_RBM_weights(model_weights[:, tmp_ind], 10, 10)
plt.suptitle('Sortirane matrice tezina - od najucestalijih do najmanje koristenih')


r_input = np.random.rand(100, HIDDEN_SIZE)
r_input[r_input > 0.9] = 1  # postotak aktivnih - slobodno varirajte
r_input[r_input < 1] = 0
r_input = r_input * 20  # pojačanje za slučaj ako je mali postotak aktivnih

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

h1_prob, h1, v1_prob, v1 = model.reconstruct(torch.from_numpy(r_input).float(), 19)

plt.figure(figsize=(16, 16))
for idx in range(0, 19):
    plt.figure(figsize=(14, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(r_input[idx, ...].reshape(10, 10))
    if idx == 0:
        plt.title("Set state")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(h1[idx, ...].view(10, 10))
    if idx == 0:
        plt.title("Final state")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(v1_prob[idx, ...].view(28, 28))
    if idx == 0:
        plt.title("Reconstruction")
    plt.axis('off')

