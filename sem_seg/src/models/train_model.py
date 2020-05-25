import src.features.build_features as bf
import torch.utils.data as data
import torch
import yaml
from pydoc import locate
import torch.nn.functional as F


def train(net, parameters_filepath, transform):

    parameters = yaml.safe_load(open(parameters_filepath, 'r'))
    dataset = bf.SBD(parameters['input_filepath'], transform=transform)
    dataloader = data.DataLoader(dataset, batch_size=parameters['batch_size'], shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net.to(device)
    net.train()

    criterion = locate(parameters['criterion'])(ignore_index=10)
    optimizer = locate(parameters['optimizer'])(net.parameters(), lr=parameters['lr'])
    scheduler = locate(parameters['scheduler'])(optimizer, step_size=parameters['step_size'], gamma=parameters['gamma'])

    losses_per_epoch = {}

    for epoch in range(parameters['epoch']):
        losses = []
        for i, sample in enumerate(dataloader, 0):
            inputs, labels = sample['image'], sample['label']
            inputs, labels = inputs.to(device), labels.to(device)

            # Clear the gradients of all optimized tensors
            optimizer.zero_grad()

            # Calculate networks prediction
            outputs = net(inputs)

            # Upsampling
            outputs = F.interpolate(outputs, size=(320, 320), mode='bilinear', align_corners=False)

            # Calculate loss
            loss = criterion(outputs, labels)
            losses.append(loss.item())

            # Backprop
            loss.backward()

            # Update scheduler
            optimizer.step()

        print('Epoch: %d | Average loss in this epoch: %.9f' % (epoch, sum(losses) / len(losses)))
        losses_per_epoch[epoch] = losses
        scheduler.step()

    print('Finished training.')
    torch.save(net.state_dict(), parameters['output_filepath'])
