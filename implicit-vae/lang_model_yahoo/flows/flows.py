import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal
from torch.nn.parameter import Parameter
import numpy as np
import math

from .transforms import *
from .nets import *


class NormalizingFlow(nn.Module):
    def __init__(self, dim=32, K=16, nh=32, flow_type='planar'):
        super().__init__()
        
        if flow_type == 'planar':
            self.transforms = nn.ModuleList([PlanarTransform(dim) for k in range(K)])
        elif flow_type == 'iaf':
            self.transforms= nn.ModuleList([IAFTransform(dim, k % 2, nh=nh) for k in range(K)])
        elif flow_type == 'maf':
            self.transforms= nn.ModuleList([MAFTransform(dim, k % 2, nh=nh) for k in range(K)])
        elif flow_type == 'slow_maf':
            self.transforms= nn.ModuleList([SlowMAFTransform(dim, k % 2, nh=nh) for k in range(K)])

    def forward(self, z, logdet=False):
        log_det = 0.
        for transform in self.transforms:
            z, ld = transform(z)
            log_det += ld
        return z, log_det
