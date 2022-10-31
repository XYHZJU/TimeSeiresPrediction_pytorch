import torch
import torch.nn as nn
import torch.nn.functional as F

class DirectLinear(nn.Module):
    def __init__(self, configs):
        super(DirectLinear, self).__init__()
        self.use_activation = configs.use_activation
        self.temporallinear = nn.Linear(configs.seq_len, configs.pred_len)
        self.outlinear = nn.Linear(configs.enc_in, configs.enc_out)
        if self.use_activation and configs.activation == "relu":
            self.activation = F.relu
        elif self.use_activation and configs.activation == "sigmoid":
            self.activation = torch.sigmoid
        elif self.use_activation:
            self.activation = F.gelu

    def forward(self, x_in):
        out = self.temporallinear(x_in.transpose(0,2,1)).transpose(0,2,1)
        if self.use_activation:
            out = self.activation(out)
        out = self.outlinear(out)
        return out