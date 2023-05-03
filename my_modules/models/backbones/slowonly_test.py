import torch
import torch.nn as nn
from mmaction.registry import MODELS
from einops import rearrange

@MODELS.register_module()
class SlowOnlyTest(nn.Module):

    def __init__(self,
                 out_indices=(4,),
                 freeze_bn=True,
                 freeze_bn_affine=True
                 ):
        super(SlowOnlyTest, self).__init__()
        model = torch.hub.load("facebookresearch/pytorchvideo", model='slow_r50', pretrained=True)
        self.blocks = model.blocks[:-1]     # exclude the last HEAD block
        self.out_indices = out_indices
        self._freeze_bn = freeze_bn
        self._freeze_bn_affine = freeze_bn_affine

    def forward(self, x):
        x = rearrange(x, 'n c (t1, t2) h w -> (n t2) c t1 h w', t1=16, t2=10)
        outs = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in self.out_indices:
                outs.append(x)
        outs = [rearrange(out, '(n t2) c t1 h w -> n c (t1 t2) h w', t2=10) for out in outs]
        if len(outs) == 1:
            return outs[0]
        return outs

    def train(self, mode=True):
        super(SlowOnlyTest, self).train(mode)
        if self._freeze_bn and mode:
            for name, m in self.named_modules():
                if isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
                    m.eval()
                    if self._freeze_bn_affine:
                        m.weight.register_hook(lambda grad: torch.zeros_like(grad))
                        m.bias.register_hook(lambda grad: torch.zeros_like(grad))
