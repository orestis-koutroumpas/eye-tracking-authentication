import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, hidden_1_size, hidden_2_size, output_size):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_1_size),
            nn.ReLU(),
            nn.Linear(hidden_1_size, hidden_2_size),
            nn.ReLU(),
            nn.Linear(hidden_2_size, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)
