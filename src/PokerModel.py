import torch.nn as nn


class PokerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PokerModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)
