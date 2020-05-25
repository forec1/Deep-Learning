import src.features.build_features as features
import src.data.transforms as t
import torch
import torch.utils.data as data
import torch.nn.functional as F
import click
import src.models.resnet as resnet
import src.models.mynn as mynn


def accuracy(net, mode='val'):
    dataset = features.SBD(data_dir='./data/processed', transform=t.ToTensor(), mode=mode)
    dataloader = data.DataLoader(dataset=dataset, batch_size=1, shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.eval()
    net.to(device)

    corr_pred_px, num_px = 0, 0
    for i, sample in enumerate(dataloader):
        img = sample['image']
        shape = torch.squeeze(sample['shape'])
        shape = shape[1], shape[0]

        # making prediction
        output = net(img)
        output = F.interpolate(output, size=shape, mode='bilinear', align_corners=False)
        output = torch.argmax(output, dim=1)
        output = torch.squeeze(output, dim=0)

        ground_truth = sample['label']
        ground_truth = torch.squeeze(ground_truth, dim=0)

        corr_pred_px += torch.sum(ground_truth == output).item()
        num_px += ground_truth.numel()

    return float(corr_pred_px) / num_px


@click.command()
@click.argument('model_path', type=click.Path(exists=True))
@click.option('-ub', '--upsampling_blocks', is_flag=True)
def main(model_path, upsampling_blocks):
    if upsampling_blocks:
        net = mynn.create_mynet(num_classes=9)
    else:
        net = resnet.resnet50_2d_semseg(num_classes=9)

    device = torch.device("cpu")
    net.load_state_dict(torch.load(model_path, map_location=device))
    acc = accuracy(net, mode='val')
    print('Accuracy on val = %f' % acc)


if __name__ == '__main__':
    main()
