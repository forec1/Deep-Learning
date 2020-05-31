import torch
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


import pt_models
import observers

from pathlib import Path

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def evaluate(name, model, dataloader, loss, config, writer):
    model.eval()
    N = len(dataloader) * dataloader.batch_size
    num_batches = N // dataloader.batch_size
    cnt_correct, loss_avg = 0, 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            labels_ = torch.argmax(outputs, dim=1)

            cnt_correct += (labels == labels_).sum().item()
            loss_avg += loss(outputs, labels).item()

    acc = cnt_correct / N * 100
    loss_avg = loss_avg / num_batches

    print(name + " accuracy = %.2f" % acc)
    print(name + " avg loss = %.2f\n" % loss_avg)

    writer.add_hparams({'weight_decay': config['weight_decay']},
                       {'hparam/accuracy': acc,
                        'hparam/avg_loss': loss_avg})


DATA_DIR = Path(__file__).parent / 'datasets' / 'MNIST'
SAVE_DIR = Path(__file__).parent / 'pt_filters'

config = {}
config['max_epochs'] = 8
config['batch_size'] = 50
config['weight_decay'] = 1e-2
config['SAVE_DIR'] = SAVE_DIR

transfom = transforms.Compose([
    transforms.ToTensor()
])

ds_train = MNIST(DATA_DIR, train=True, download=True, transform=transfom)
ds_test = MNIST(DATA_DIR, train=False, transform=transfom)
dl_train = torch.utils.data.DataLoader(ds_train, batch_size=config['batch_size'],
                                       shuffle=True)
dl_test = torch.utils.data.DataLoader(ds_test, batch_size=config['batch_size'])

model = pt_models.ConvolutionalModelMNIST((28, 28), 1, 16, 32, 512, 10)
model.to(device)
model.train()

conv_filter_visalizer = observers.ConvFiltersVisualizer(model, SAVE_DIR, layer_idxs=[1])

optimizer = optim.SGD(model.parameters(), lr=1e-1,
                      weight_decay=config['weight_decay'])
lr_scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[2,4,6], gamma=0.1)
criterion = nn.CrossEntropyLoss()

writer = SummaryWriter()

N = len(ds_train)
num_batches = N // dl_train.batch_size

model.visualize(0)

for epoch in range(config['max_epochs']):
    loss_sum = 0.0
    for batch_idx, data in enumerate(dl_train):
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss_sum += loss.item()
        loss.backward()

        optimizer.step()

        writer.add_scalar('Batch_loss/train', loss.item(), epoch * num_batches + batch_idx)
        if batch_idx % 100 == 0:
            print('epoch %d, step %d/%d, batch loss = %.3f' %
                  (epoch, (batch_idx*dl_train.batch_size), N, loss.item()))

    lr_scheduler.step()
    model.visualize(epoch)
    print('epoch %d, avg loss = %.3f' % (epoch, loss_sum/N))
    writer.add_scalar('Epoch_loss/train', loss_sum / N, epoch)

evaluate('Test', model, dl_test, criterion, config, writer)
writer.close()
