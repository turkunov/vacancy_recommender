import torch.nn as nn

class DQNSolver(nn.Module):
    """
    Класс с DQN-сетью для взаимодействия с предобученными весами
    """

    def __init__(self, n_actions, obvs_size):
        super().__init__()
        self.agent_net = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )

    def forward(self, x):
        return self.agent_net(x)
