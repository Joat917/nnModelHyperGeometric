import torch


class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m = torch.nn.Sequential(
            torch.nn.Linear(4, 15),
            torch.nn.Sigmoid(),
            torch.nn.Linear(15, 15),
            torch.nn.Sigmoid(),
            torch.nn.Linear(15, 15),
            torch.nn.Sigmoid(),
            torch.nn.Linear(15, 15),
            torch.nn.Sigmoid(),
            torch.nn.Linear(15, 10),
            torch.nn.Sigmoid(),
            torch.nn.Linear(10, 10),
            torch.nn.Sigmoid(),
            torch.nn.Linear(10, 1),
        )

    def forward(self, input):
        return self.m.forward(input)
