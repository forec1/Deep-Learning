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
import tqdm

import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt


def prepare_data_loaders(batch_size=32):
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./files', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.Resize((64, 64)),
                                       torchvision.transforms.ToTensor()
                                   ])), batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./files', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.Resize((64, 64)),
                                       torchvision.transforms.ToTensor()
                                   ])), batch_size=batch_size)

    return train_loader, test_loader


class Generator(nn.Module):
    def __init__(self, latent_size):
        super().__init__()

        self.latent_size = latent_size

        self.conv1 = nn.ConvTranspose2d(latent_size, 512, kernel_size=4, stride=1)
        self.bn1 = nn.BatchNorm2d(512)
        weights_init(self.conv1)
        weights_init(self.bn1)

        self.conv2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        weights_init(self.conv2)
        weights_init(self.bn2)

        self.conv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        weights_init(self.conv3)
        weights_init(self.bn3)

        self.conv4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        weights_init(self.conv4)
        weights_init(self.bn4)

        self.conv5 = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(1)
        weights_init(self.conv5)
        weights_init(self.bn5)

        self.act_fn = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        x = self.act_fn(self.bn1(self.conv1(x)))
        x = self.act_fn(self.bn2(self.conv2(x)))
        x = self.act_fn(self.bn3(self.conv3(x)))
        x = self.act_fn(self.bn4(self.conv4(x)))
        x = torch.tanh(self.conv5(x))
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        weights_init(self.conv1)
        weights_init(self.bn1)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        weights_init(self.conv2)
        weights_init(self.bn2)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        weights_init(self.conv3)
        weights_init(self.bn3)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        weights_init(self.conv4)
        weights_init(self.bn4)

        self.conv5 = nn.Conv2d(512, 1, kernel_size=4, stride=1)
        self.bn5 = nn.BatchNorm2d(1)
        weights_init(self.conv5)
        weights_init(self.bn5)

        self.act_fn = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        x = self.act_fn(self.bn1(self.conv1(x)))
        x = self.act_fn(self.bn2(self.conv2(x)))
        x = self.act_fn(self.bn3(self.conv3(x)))
        x = self.act_fn(self.bn4(self.conv4(x)))
        x = torch.sigmoid(self.conv5(x))
        return x


def weights_init(w):
    classname = w.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(w.weight.data, 0.0, 0.02)
    elif classname.find('Batch') != -1:
        nn.init.normal_(w.weight.data, 1.0, 0.02)
        nn.init.constant_(w.bias.data, 0)


dmodel = Discriminator()
gmodel = Generator(100)

dmodel.apply(weights_init)
gmodel.apply(weights_init)


def train(gmodel: Generator, dmodel: Discriminator, n_epochs=10, log_epochs=1, batch_size=32, learning_rate=1e-3, device='cpu'):
    train_loader, test_loader = prepare_data_loaders(batch_size=batch_size)

    gmodel = gmodel.to(device)
    dmodel = dmodel.to(device)

    gmodel.train()
    dmodel.train()

    criterion = nn.BCELoss()

    g_optim = optim.Adam(gmodel.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    d_optim = optim.Adam(dmodel.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    for epoch_idx in range(0, n_epochs):

        g_loss, d_loss = 0, 0

        for image_data, _ in tqdm.tqdm(train_loader):
            # discriminator update
            dmodel.zero_grad()

            # real data pass
            image_data = image_data.to(device)

            batch_size = image_data.shape[0]
            labels = torch.ones(batch_size, device=device).float()

            d_output = dmodel(image_data)
            d_err_real = criterion(d_output, labels)
            d_err_real.backward()
            d_loss += d_err_real.item() / batch_size

            # fake data pass
            noise = torch.randn(batch_size, gmodel.latent_size, 1, 1, device=device)
            fake_image_data = gmodel(noise)
            labels = torch.zeros(batch_size, device=device).float()

            d_output = dmodel(fake_image_data.detach())
            d_error_fake = criterion(d_output, labels)
            d_error_fake.backward()
            d_loss += d_error_fake.item() / batch_size

            d_optim.step()

            # generator update
            gmodel.zero_grad()

            labels = torch.ones(batch_size, device=device)
            d_output = dmodel(fake_image_data)
            g_error = criterion(d_output, labels)
            g_error.backward()
            g_loss += g_error.item() / batch_size
            g_optim.step()

        if (epoch_idx + 1) % log_epochs == 0:
            print(f"[{epoch_idx+1}/{n_epochs}]: d_loss = {d_loss:.2f} g_loss {g_loss:.2f}")

    gmodel.eval()
    dmodel.eval()

    return gmodel, dmodel


gmodel, dmodel = train(gmodel, dmodel, n_epochs=15, batch_size=256, device='cuda')
torch.save(gmodel.state_dict(), 'gmodel.pt')
torch.save(dmodel.state_dict(), 'dmodel.pt')

random_sample = gmodel(torch.randn(100, 100, 1, 1).to('cuda')).view(100, 64, 64).data.cpu().numpy()

plt.figure(figsize=(16, 16))
for idx in range(0, 100):
    plt.subplot(10, 10, idx+1)
    plt.imshow(random_sample[idx, ...])
    plt.clim(0, 1)
    plt.axis('off')

