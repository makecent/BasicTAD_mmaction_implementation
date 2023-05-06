import torch
from mmaction.registry import MODELS
from mmengine.model import BaseModule


@MODELS.register_module()
class TRN(BaseModule):

    def __init__(self):
        super().__init__()
        self.gamma = torch.nn.parameter.Parameter(torch.tensor(0.), requires_grad=True)
        self.beta = torch.nn.parameter.Parameter(torch.tensor(0.), requires_grad=True)

    def forward(self, x):
        l2 = torch.linalg.norm(x, ord=2, dim=(-1, -2))
        alpha = l2 / l2.sum(dim=-1, keepdim=True)
        out = self.gamma * alpha.unsqueeze(-1).unsqueeze(-1) * x + self.beta + x
        return out

