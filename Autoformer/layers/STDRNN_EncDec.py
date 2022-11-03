import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp
from layers.Autoformer_EncDec import Decoder


class RNNEncoderLayer(nn.Module):
    """
    RNN encoder layer with the progressive decomposition architecture
    """
    def __init__(self, d_model, moving_avg=25, dropout=0.1, activation="relu"):
        super(RNNEncoderLayer, self).__init__()
        self.rnn = nn.GRU(d_model, d_model, batch_first=True, dropout=dropout)
        self.decomp = series_decomp(moving_avg)
        if activation == "sigmoid":
            self.activation = torch.sigmoid
        elif activation == "relu":
            self.activation = F.relu
        else:
            self.activation = F.gelu

    def forward(self, x, attn_mask=None):
        hidden, h_n = self.rnn(x)
        # residual connection & decomposition
        hidden = hidden + x
        hidden, _ = self.decomp(self.activation(hidden))
        return hidden, h_n


class RNNDecoderLayer(nn.Module):
    """
    RNN encoder layer with the progressive decomposition architecture
    """
    def __init__(self, d_model, dec_out, moving_avg=25, dropout=0.1, activation="relu"):
        super(RNNDecoderLayer, self).__init__()
        self.rnn = nn.GRU(d_model, d_model, batch_first=True, dropout=dropout)
        self.decomp = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Conv1d(in_channels=d_model, out_channels=dec_out, kernel_size=3, stride=1, padding=1,
        padding_mode='circular', bias=False)
        if activation == "sigmoid":
            self.activation = torch.sigmoid
        elif activation == "relu":
            self.activation = F.relu
        else:
            self.activation = F.gelu

        
        def forward(self, x, init_hidden):
            dec_out, _ = self.rnn(x, init_hidden)
            # residual connection & decomposition
            x = x + self.activation(dec_out)
            x, residual_trend = self.decomp(x)
            residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1,2)
            return x, residual_trend


class STDRNNDecoder(Decoder):
    """
    STDRNN decoder
    """
    def forward(self, x, init_hidden_list, trend=None):
        for layer, init_hidden in zip(self.layers, init_hidden_list):
            x, residual_trend = layer(x, init_hidden)
            trend = trend + residual_trend
        
        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x, trend