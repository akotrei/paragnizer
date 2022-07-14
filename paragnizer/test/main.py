from numpy import dtype
import torch
from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self, size_in, size_out, dtype=torch.float64):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.stack = nn.Sequential(nn.Linear(size_in, size_out, dtype=dtype))

        self.stack[0].weight = nn.Parameter(torch.ones_like(self.stack[0].weight))
        self.stack[0].bias = nn.Parameter(torch.ones_like(self.stack[0].bias))

    def forward(self, x):
        x = torch.from_numpy(x)
        x = self.flatten(x)
        logits = self.stack(x)
        return logits


class BiLSTM(nn.Module):
    def __init__(self, size_in, size_out):
        super(BiLSTM, self).__init__()
        self.stack = nn.Sequential(
            nn.LSTM(
            input_size=size_in,
            hidden_size=size_out,
            bidirectional=True,
            ),
        )

    def forward(self, x):
        x = torch.from_numpy(x)
        logits = self.stack(x)
        out = logits[0]
        softed = torch.softmax(out, 1)
        return softed


def main():
    x = torch.rand(5, 3)
    print(x)


def get_availability():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device: {torch.version.cuda}")


def get_nn(size_in, size_out, dtype=torch.float64):
    nn = NeuralNetwork(size_in, size_out, dtype)
    return nn

def get_bilstm(size_in, size_out):
    nn = BiLSTM(size_in, size_out)
    return nn

    