import numpy as np
import torch
import torch.nn as nn
import torchvision

from barlow_twins import BarlowTwins
from transforms import EvalTransform


def save_representations(model, normal_class):
    data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=EvalTransform())
    loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False)

    model.eval()

    x_data = np.zeros((0, 1024))
    y_data = np.zeros(0)

    count_normal = 0
    count_anomalous = 0

    for x, y in loader:
        x = x.to('cuda')
        if y[0] == normal_class:
            if count_normal >= 3000:
                continue
            count_normal += 1
        else:
            if count_anomalous >= 3000:
                continue
            count_anomalous += 1

        with torch.no_grad():
            z = model.module.projector(model.module.backbone(x))
            x_data = np.append(x_data, z.cpu().numpy(), axis=0)
            y_data = np.append(y_data, np.asarray([int(y[0] != normal_class)]), axis=0)

    np.savez_compressed(f'CV_by_BT/CIFAR10_{normal_class}.npz', X=x_data, y=y_data)