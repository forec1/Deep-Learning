import torch
import torch.nn as nn


class BaseLineModel(nn.Module):
    '''avg_pool() -> fc(300, 150) -> ReLU() -> fc(150, 150) -> ReLU() -> fc(150,1)'''

    def __init__(self, embedded_matrix):
        super(BaseLineModel, self).__init__()
        self.embedded_matrix = embedded_matrix

        self.fc1 = nn.Linear(300, 150)
        self.fc2 = nn.Linear(150, 150)
        self.fc3 = nn.Linear(150, 1)

    def forward(self, x):
        x = self.embedded_matrix(x)
        x = torch.mean(x, dim=1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = nn.functional.relu(x)
        x = self.fc3(x)
        return x
