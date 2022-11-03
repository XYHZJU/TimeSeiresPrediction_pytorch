import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.use_activation = configs.use_activation
        self.temporallinear = nn.Linear(configs.seq_len, configs.pred_len)
        self.outlinear = nn.Linear(configs.enc_in, configs.c_out)
        if self.use_activation and configs.activation == "relu":
            self.activation = F.relu
        elif self.use_activation and configs.activation == "sigmoid":
            self.activation = torch.sigmoid
        elif self.use_activation:
            self.activation = F.gelu

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # print("datatype:",x_enc.shape)
        out = self.temporallinear(x_enc.permute(0,2,1)).permute(0,2,1)
        if self.use_activation:
            out = self.activation(out)
        out = self.outlinear(out)
        return out