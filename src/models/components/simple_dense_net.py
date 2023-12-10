from torch import nn


class SimpleDenseNet(nn.Module):
    """
        Neural network model for classificaiton.
        Forward() call returns logits.
    """
    def __init__(
        self,
        input_size: int = 784,
        lin1_size: int = 256,
        lin2_size: int = 256,
        lin3_size: int = 256,
        output_size: int = 10,
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
                nn.Linear(lin3_size, output_size),
            )
        else:
            self.model = nn.Sequential(
                nn.Linear(input_size, lin1_size),
                nn.ReLU(),
                nn.Linear(lin1_size, lin2_size),
                nn.ReLU(),
                nn.Linear(lin2_size, lin3_size),
                nn.ReLU(),
                nn.Linear(lin3_size, output_size),
            )
        self.output_size = output_size

    def forward(self, x):
        return self.model(x)
