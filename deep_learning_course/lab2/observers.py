import torch
from torchvision.utils import make_grid

import re
from os.path import join
import skimage.io


class ConvFiltersVisualizer:

    def __init__(self, model, SAVE_DIR, layer_idxs=None):
        self.model = model
        self.model.attach(self)
        self.SAVE_DIR = SAVE_DIR
        self.layer_idxs = layer_idxs

    def visualize(self, epoch):
        with torch.no_grad():
            for named_params in self.model.named_parameters():
                match = re.match(r'.*(conv)(\d+).weight', named_params[0])
                if match and int(match.group(2)) in self.layer_idxs:
                    layer_idx = int(match.group(2))
                    filters = named_params[1].clone().detach().cpu()
                    three_C = True if filters.size(1) == 3 else False
                    filters -= filters.min()
                    filters /= filters.max()
                    filters_grid = make_grid(filters, padding=1, nrow=8)
                    filters_grid = filters_grid.numpy().transpose((1, 2, 0))
                    filters_grid = filters_grid if three_C else filters_grid[:,:,0]

                    filename = join(self.SAVE_DIR, 'conv%d_filters_epoch_%d.png'
                                    % (layer_idx, epoch))
                    skimage.io.imsave(filename, filters_grid)
