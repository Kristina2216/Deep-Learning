import torch
import numpy as np

class Baseline(torch.nn.Module):
    def __init__(self, embeddingMatrix):
        super(Baseline, self).__init__()
        self.fc1 = torch.nn.Linear(in_features=300, out_features=150)
        self.fc2 = torch.nn.Linear(in_features=150, out_features=150)
        self.fc3 = torch.nn.Linear(in_features=150, out_features=1)
        self.emb= embeddingMatrix


    def forward(self, x):
        h1 = torch.mean(self.emb(x), dim=1 if len(x.shape) == 2 else 0)
        h2 = torch.relu(self.fc1(h1))
        h3 = torch.relu(self.fc2(h2))
        return self.fc3(h3)



