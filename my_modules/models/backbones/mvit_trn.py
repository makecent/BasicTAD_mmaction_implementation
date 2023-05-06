import torch
from mmaction.registry import MODELS
from .mvit import MViT


class TRN(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.gamma = torch.nn.parameter.Parameter(torch.tensor(0.), requires_grad=True)
        self.beta = torch.nn.parameter.Parameter(torch.tensor(0.), requires_grad=True)

    def forward(self, x):
        l2 = torch.linalg.norm(x, ord=2, dim=(-1, -2))
        alpha = l2 / l2.sum(dim=-1, keepdim=True)
        out = self.gamma * alpha.unsqueeze(-1).unsqueeze(-1) * x + self.beta + x
        return out
@MODELS.register_module()
class MViT_TRN(MViT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for block in self.blocks:
            block.mlp.act = torch.nn.ModuleList([TRN(), torch.nn.GELU()])