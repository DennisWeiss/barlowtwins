import torch.nn as nn


class LinearClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearClassifier, self).__init__()

        self.classifier = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.classifier(x)