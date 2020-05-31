import torch.utils.data
from torchvision.datasets import CIFAR10
from pathlib import Path
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch
import pt_models
import torch.nn as nn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DATA_DIR = Path(__file__).parent / 'datasets' / 'CIFAR10'


def show_worst(model, dataset, loss, n=20):
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


criterion = nn.CrossEntropyLoss()
model = pt_models.ConvolutionalModel((32, 32), 3, [16, 32], [256, 128], 10)
ds_train = CIFAR10(DATA_DIR, train=False, transform=transforms.ToTensor())
show_worst(model, ds_train, criterion)
