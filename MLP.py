import torch
from torch.nn import Module

# model definition
class MLP(Module):
    # define model elements
    def __init__(self, n_inputs, n_outputs):
        super(MLP, self).__init__()

        self.hid1 = torch.nn.Linear(n_inputs, int(n_inputs * 0.9))
        self.hid2 = torch.nn.Linear(int(n_inputs * 0.9), int(n_inputs * 0.8))
        self.oupt = torch.nn.Linear(int(n_inputs * 0.8), n_outputs)

        torch.nn.init.xavier_uniform_(self.hid1.weight)
        torch.nn.init.zeros_(self.hid1.bias)
        torch.nn.init.xavier_uniform_(self.hid2.weight)
        torch.nn.init.zeros_(self.hid2.bias)
        torch.nn.init.xavier_uniform_(self.oupt.weight)
        torch.nn.init.zeros_(self.oupt.bias)

 
    # forward propagate input
    def forward(self, X):
        z = torch.sigmoid(self.hid1(X))
        z = torch.sigmoid(self.hid2(z))
        # No softmax, happens in CrossEntropyLoss
        z = self.oupt(z)
        return z