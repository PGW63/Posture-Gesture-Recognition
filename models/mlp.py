import torch
import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, input_dim, num_classes, dropout=0.2):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # dropout is active only when self.training == True
        # controlled by model.train() / model.eval()
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x