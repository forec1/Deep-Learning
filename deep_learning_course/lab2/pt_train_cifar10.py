import torch
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import torch.utils.data
import torch.optim as optim
import torch.nn as nn

from pathlib import Path
import os

import pt_models
import observers

import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def evaluate(name, model, dataloader, loss, config):
    with torch.no_grad():
        confusion_m = np.zeros((10, 10))
        N = len(dataloader) * dataloader.batch_size
        num_batches = N // dataloader.batch_size
        cnt_correct, loss_avg = 0, 0
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            labels_ = torch.argmax(outputs, dim=1)

            confusion_m[labels.cpu(), labels_.cpu()] += 1
            cnt_correct += (labels == labels_).sum().item()
            loss_avg += loss(outputs, labels).item()

        acc = cnt_correct / N * 100
        loss_avg = loss_avg / num_batches

        print(name + " accuracy = %.2f" % acc)
        print(name + " avg loss = %.2f" % loss_avg)

        for i in range(10):
            TP = confusion_m[i, i]
            FP = np.sum(confusion_m[:, i])
            FN = np.sum(confusion_m[i, :])
            print(name + " - class %d, precision = %.3f" % (i, TP / (TP + FP)))
            print(name + " - class %d, recall = %.3f" % (i, TP / (TP + FN)))

        print()
        return loss_avg, acc, confusion_m


def plot_training_progress(save_dir, plot_data):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16,8))

    linewidth = 2
    legend_size = 10
    train_color = 'm'
    val_color = 'c'

    num_points = len(plot_data['train_loss'])
    x_data = np.linspace(1, num_points, num_points)
    ax1.set_title('Cross-entropy loss')
    ax1.plot(x_data, plot_data['train_loss'], marker='o', color=train_color,
             linewidth=linewidth, linestyle='-', label='train')
    ax1.plot(x_data, plot_data['valid_loss'], marker='o', color=val_color,
             linewidth=linewidth, linestyle='-', label='validation')
    ax1.legend(loc='upper right', fontsize=legend_size)
    ax2.set_title('Average class accuracy')
    ax2.plot(x_data, plot_data['train_acc'], marker='o', color=train_color,
             linewidth=linewidth, linestyle='-', label='train')
    ax2.plot(x_data, plot_data['valid_acc'], marker='o', color=val_color,
             linewidth=linewidth, linestyle='-', label='validation')
    ax2.legend(loc='upper left', fontsize=legend_size)
    ax3.set_title('Learning rate')
    ax3.plot(x_data, plot_data['lr'], marker='o', color=train_color,
             linewidth=linewidth, linestyle='-', label='learning_rate')
    ax3.legend(loc='upper left', fontsize=legend_size)

    save_path = os.path.join(save_dir, 'training_plot.png')
    print('Plotting in: ', save_path)
    plt.savefig(save_path)
    plt.close()


def show_worst(model, dataset, loss, n=20):
    model.to(torch.device('cpu'))
    to_tensor = transforms.ToTensor()
    dl = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    loss_dict = {}
    for i, (inputs, labels) in enumerate(dl):
        outputs = model(inputs)
        loss_dict[i] = loss(outputs, labels).item()

    loss_dict = {k: v for k, v in sorted(loss_dict.items(), key=lambda item: -item[1])}

    for i in list(loss_dict)[:n]:
        input_img = to_tensor(dataset.data[i]).unsqueeze_(0)
        outputs = model(input_img)

        pred = outputs.cpu().detach().topk(k=3).indices.numpy()[0,:]
        title = "Class %d" % dataset.targets[i]
        title += " Predicted " + str(pred)

        plt.title(title)
        plt.imshow(dataset.data[i])
        plt.show()


DATA_DIR = Path(__file__).parent / 'datasets' / 'CIFAR10'
SAVE_DIR = Path(__file__).parent / 'pt_cifar10_filters'
PLOTS_SAVE_DIR = Path(__file__).parent / 'pt_cifar10_plots'

config = {}
config['batch_size'] = 50
config['max_epochs'] = 50
config['weight_decay'] = 1e-4

transform = transforms.Compose([
    transforms.ToTensor()
])

ds_train = CIFAR10(DATA_DIR, train=True, download=True, transform=transform)

data_mean = ds_train.data.mean(axis=(0,1,2))/255
data_std = ds_train.data.std(axis=(0,1,2))/255

transform.transforms += [transforms.Normalize(data_mean, data_std)]

ds_train = CIFAR10(DATA_DIR, train=True, transform=transform)
ds_test = CIFAR10(DATA_DIR, train=False, transform=transform)

ds_train, ds_valid = torch.utils.data.dataset.random_split(ds_train, (len(ds_train) - 5000, 5000))

dl_train = torch.utils.data.DataLoader(ds_train, batch_size=config['batch_size'], shuffle=True)
dl_val = torch.utils.data.DataLoader(ds_valid, batch_size=config['batch_size'])

model = pt_models.ConvolutionalModel((32, 32), 3, [16, 32], [256, 128], 10)
model.to(device)
model.train()
conv_filter_vis = observers.ConvFiltersVisualizer(model, SAVE_DIR, layer_idxs=[0])

optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=config['weight_decay'])

criterion = nn.CrossEntropyLoss()
lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

model.visualize(0)

plot_data = {}
plot_data['train_loss'] = []
plot_data['valid_loss'] = []
plot_data['train_acc'] = []
plot_data['valid_acc'] = []
plot_data['lr'] = []

n_batch = len(ds_train) // config['batch_size']

for epoch in range(config['max_epochs']):
    for b_idx, (inputs, labels) in enumerate(dl_train):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        if b_idx % 100 == 0:
            print("epoch: {}, step: {}/{}, batch_loss: {}"
                  .format(epoch, b_idx, n_batch, loss.item()))

    model.visualize(epoch+1)
    train_loss, train_acc, cm_tr = evaluate('Train', model, dl_train, criterion, config)
    val_loss, val_acc, cm_val = evaluate('Val', model, dl_val, criterion, config)

    plt.imsave(os.path.join(PLOTS_SAVE_DIR, "cm_val_epoch%d.png" % (epoch+1)), cm_val)
    plt.imsave(os.path.join(PLOTS_SAVE_DIR, "cm_tr_epoch%d.png" % (epoch+1)), cm_tr)

    plot_data['train_loss'] += [train_loss]
    plot_data['valid_loss'] += [val_loss]
    plot_data['train_acc'] += [train_acc]
    plot_data['valid_acc'] += [val_acc]
    plot_data['lr'] += [lr_scheduler.get_lr()]
    lr_scheduler.step()


plot_training_progress(PLOTS_SAVE_DIR, plot_data)
show_worst(model, ds_test, criterion)
