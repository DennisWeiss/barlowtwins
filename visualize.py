import numpy as np
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from barlow_twins import BarlowTwins
from transforms import EvalTransform
from sklearn.manifold import TSNE


def draw_tsne_visualization(model, normal_class):
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
            if count_normal >= 2000:
                continue
            count_normal += 1
        else:
            if count_anomalous >= 2000:
                continue
            count_anomalous += 1

        with torch.no_grad():
            z = model.module.projector(model.module.backbone(x))
            x_data = np.append(x_data, z.cpu().numpy(), axis=0)
            y_data = np.append(y_data, np.asarray([int(y[0] != normal_class)]), axis=0)

    x_tsne_embedded = TSNE(n_components=2, perplexity=100, n_iter=2000).fit_transform(x_data)

    plt.scatter(x_tsne_embedded[:, 0], x_tsne_embedded[:, 1], c=['red' if y_data[i] else 'green' for i in range(y_data.shape[0])], marker='2')
    plt.show()