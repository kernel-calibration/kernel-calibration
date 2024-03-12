from torch import nn
import torch
import torch.nn.functional as F


class DenseGMMNet(nn.Module):
    """
        Neural network model that parametrizes a Gaussian Mixture Model (GMM).
        Forward() call returns mean, std, prob which can be used to create
        torch distribution object.
    """
    def __init__(
        self,
        input_size: int = 1,
        lin1_size: int = 256,
        lin2_size: int = 256,
        lin3_size: int = 256,
        output_size: int = 1,
        num_components: int = 1,
        use_batchnorm: bool = True,
    ):
        super().__init__()

        if use_batchnorm:
            self.model = nn.Sequential(
                nn.Linear(input_size, lin1_size),
                nn.BatchNorm1d(lin1_size),
                nn.ReLU(),
                nn.Linear(lin1_size, lin2_size),
                nn.BatchNorm1d(lin2_size),
                nn.ReLU(),
                nn.Linear(lin2_size, lin3_size),
                nn.BatchNorm1d(lin3_size),
                nn.ReLU(),
                nn.Linear(lin3_size, output_size * num_components * 3),
            )
        else:
            self.model = nn.Sequential(
                nn.Linear(input_size, lin1_size),
                nn.ReLU(),
                nn.Linear(lin1_size, lin2_size),
                nn.ReLU(),
                nn.Linear(lin2_size, lin3_size),
                nn.ReLU(),
                nn.Linear(lin3_size, output_size * num_components * 3),
            )

        self.n_comp = num_components
        self.output_size = output_size

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.shape[0], self.output_size, self.n_comp * 3)
        mean = x[..., :self.n_comp]
        std = torch.exp(x[..., self.n_comp:(self.n_comp * 2)])
        prob = torch.softmax(x[..., (self.n_comp * 2):], dim=-1)
        return mean, std, prob
