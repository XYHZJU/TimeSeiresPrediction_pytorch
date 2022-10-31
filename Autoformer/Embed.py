import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from einops import rearrange
import math

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

def relative_to_absolute(q):
    """
    Converts the dimension that is specified from the axis
    from relative distances (with length 2*tokens-1) to absolute distance (length tokens)
      Input: [bs, heads, length, 2*length - 1]
      Output: [bs, heads, length, length]
    """
    b, h, l, _, device, dtype = *q.shape, q.device, q.dtype
    dd = {'device': device, 'dtype': dtype}
    col_pad = torch.zeros((b, h, l, 1), **dd)
    x = torch.cat((q, col_pad), dim=3)  # zero pad 2l-1 to 2l
    flat_x = rearrange(x, 'b h l c -> b h (l c)')
    flat_pad = torch.zeros((b, h, l - 1), **dd)
    flat_x_padded = torch.cat((flat_x, flat_pad), dim=2)
    final_x = flat_x_padded.reshape(b, h, l + 1, 2 * l - 1)
    final_x = final_x[:, :, :l, (l - 1):]
    return final_x


def rel_pos_emb_1d(q, rel_emb, shared_heads):
    """
    Same functionality as RelPosEmb1D
    Args:
        q: a 4d tensor of shape [batch, heads, tokens, dim]
        rel_emb: a 2D or 3D tensor
        of shape [ 2*tokens-1 , dim] or [ heads, 2*tokens-1 , dim]
    """
    if shared_heads:
        emb = torch.einsum('b h t d, r d -> b h t r', q, rel_emb)
    else:
        emb = torch.einsum('b h t d, h r d -> b h t r', q, rel_emb)
    return relative_to_absolute(emb)


class RelPosEmb1D(nn.Module):
    def __init__(self, tokens, dim_head, heads=None):
        """
        Output: [batch head tokens tokens]
        Args:
            tokens: the number of the tokens of the seq
            dim_head: the size of the last dimension of q
            heads: if None representation is shared across heads.
            else the number of heads must be provided
        """
        super().__init__()
        scale = dim_head ** -0.5
        self.shared_heads = heads if heads is not None else True
        if self.shared_heads:
            self.rel_pos_emb = nn.Parameter(torch.randn(2 * tokens - 1, dim_head) * scale)
        else:
            self.rel_pos_emb = nn.Parameter(torch.randn(heads, 2 * tokens - 1, dim_head) * scale)

    def forward(self, q):
        return rel_pos_emb_1d(q, self.rel_pos_emb, self.shared_heads)

class CausalConv1d(torch.nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias)
        
        self.__padding = (kernel_size - 1) * dilation
        
    def forward(self, input):
        return super(CausalConv1d, self).forward(F.pad(input, (self.__padding, 0)))


class context_embedding(torch.nn.Module):
    def __init__(self,in_channels=1,embedding_size=256,k=5):
        super(context_embedding,self).__init__()
        self.causal_convolution = CausalConv1d(in_channels,embedding_size,kernel_size=k)

    def forward(self,x):
        x = self.causal_convolution(x.permute(0,2,1)).transpose(1,2)
        return F.tanh(x)


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()

        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)

class ConvEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(ConvEmbedding, self).__init__()

        self.value_embedding = context_embedding(in_channels=c_in, embedding_size=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        # print('embed: ',self.value_embedding(x).shape)
        # print('temp: ',self.temporal_embedding(x_mark).shape)
        # print('pos: ',self.position_embedding(x).shape)
        x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)

# TFT LSTM embedding layer implement, only consider targets, prior unknown inputs
class gated_residual_network(nn.Module):
    def __init__(self, input_size, d_model, additional_dim=0, dropout=0, return_gate=False):
        """
        Args
        input_size: the dimension of features in the input sequences.
        d_model: the number of hidden states dimension.
        additional_dim: the dimension of the input additional context. Default: 0.
        dropout: the dropout rate. Default: 0.
        Input
        x: shape (*, input_size)
        additional_context: shape (*, additional_dim)
        Return
        x_emb: shape (*, d_model)
        """
        super(gated_residual_network, self).__init__()
        self.input_size = input_size
        self.hidden_size = d_model
        self.dropout = nn.Dropout(dropout)
        self.return_gate = return_gate
        self.inputDense = nn.Linear(input_size, d_model, bias=False)
        self.inputlinear = nn.Linear(d_model, d_model)
        if additional_dim>0:
            self.contextlinear = nn.Linear(additional_dim, d_model, bias=False)
        self.intermedialinear = nn.Linear(d_model, d_model)
        self.afflinear = nn.Linear(d_model, d_model)
        self.weightlinear = nn.Linear(d_model, d_model)
        self.glu = nn.GLU(dim=-1)
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, x, additional_context=None):
        x_embed = self.inputDense(x)
        x_emb = self.inputlinear(x_embed)
        if additional_context is not None:
            x_emb = F.elu(x_emb+self.contextlinear(additional_context))
        else:
            x_emb = F.elu(x_emb)
        gate_weight = torch.sigmoid(self.weightlinear(x_emb))
        x_emb = torch.cat((self.afflinear(x_emb),gate_weight),dim=-1)
        x_emb = self.layernorm(x_embed+self.glu(x_emb))
        if self.return_gate:
            return x_emb, gate_weight
        else:
            return x_emb



class LSTM_embed_layer(nn.Module):
    def __init__(self, d_model, num_layers=1, dropout=0, batch_first=True, known_input=False):
        """
        Temporal Fusion Transformer LSTM embeddin layer
        Args
        d_model : the dimension of input sequence
        num_layers : the number of layers in the LSTM encoder. Default: 1.
        dropout : dropout rate. Default: 0.
        batch_first : If True, then the input and output tensors are provided as (batch, seq, feature) instead of (seq, batch, feature). Default: True
        known_input : If True, the input sequences will contain the known input sequences, which means they can be used when predicting future value, instead the input sequences will not contain the known input sequences. Default: False
        Input
        x_emb : shape (B,L,D)
        x_known : shape (B,t,D) if known_input = True
        Return
        x_lstm : shape (B,L+t,D) if known_input = True, else (B,L,D)
        """
        super(LSTM_embed_layer, self).__init__()
        self.hidden_size = d_model
        self.num_lstm = num_layers
        self.dropout = dropout
        self.batch_first = batch_first
        self.known_input = known_input
        self.lstm = nn.LSTM(d_model, d_model, num_layers, batch_first=batch_first)
        self.glu = nn.GLU(dim=-1)
        self.afflinear=nn.Linear(d_model,d_model)
        self.weightlinear=nn.Linear(d_model,d_model)
        self.layernorm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x_emb, x_known=None):
        """
        Args
        x_emb: the prior unknown input sequences, shape (batch, seq, feature) if batch_first is True, else (seq, batch, feature).
        x_known: the known input sequences, shape (batch, seq, feature) if batch_first is True, else (seq, batch, feature).
        """
        if self.known_input:
            history_lstm, (h_n, c_n) = self.lstm(x_emb)
            future_lstm ,(_, _) = self.lstm(x_known,(h_n,c_n))
            x_lstm = torch.cat((history_lstm, future_lstm), dim=1)
            x_lstm = torch.cat((self.afflinear(x_lstm), self.weightlinear(x_lstm)),dim=-1)
            x_lstm = self.dropout(self.glu(x_lstm))
            x_lstm = self.layernorm(x_lstm+x_emb)
            
            return x_lstm
        else:
            x_lstm, (_, _) = self.lstm(x_emb)
            x_lstm = torch.cat((self.afflinear(x_lstm), self.weightlinear(x_lstm)),dim=-1)
            x_lstm = self.dropout(self.glu(x_lstm))
            x_lstm = self.layernorm(x_lstm+x_emb)

            return x_lstm