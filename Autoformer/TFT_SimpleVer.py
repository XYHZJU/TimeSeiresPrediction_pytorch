import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Transformer_Encdec import TFTEncoderLayer, Encoder
from layers.Embed import gated_residual_network, LSTM_embed_layer

class TemporalFusionTransformer(nn.Module):
    """
    Simple version TFT
    """
    def __init__(self, configs):
        super(TemporalFusionTransformer, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Embedding
        self.grn = gated_residual_network(configs.input_size, configs.d_model, 0, configs.dropout)
        self.lstm_embedding = LSTM_embed_layer(configs.d_model, configs.embed_layer, configs.dropout, known_input=False)

        # Encoder
        # attnmask = TriangularCausalMask(configs.batch_size, configs.seq_len)
        self.encoder = Encoder(
            [
                TFTEncoderLayer(AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                output_attention=configs.output_attention),configs.d_model, configs.num_heads),configs.d_model, configs.dropout, activation=configs.activation)
                for l in range(configs.e_layers)
            ]
        )

        # Decoder
        self.gatelinear = nn.Linear(configs.d_model, configs.d_model)
        self.weightlinear = nn.Linear(configs.d_model, configs.d_model)
        self.glu = nn.GLU(dim=-1)
        self.norm = nn.LayerNorm(configs.d_model)
        self.declinear = nn.Linear(configs.d_model, configs.d_output)
        
    def forward(self, x_enc, x_enc_known=None, enc_self_mask=None):
        emb_out = self.grn(x_enc)
        if x_enc_known is not None:
            emb_out_known = self.grn(x_enc_known)
            lstm_out = self.lstm_embedding(emb_out, emb_out_known)
        else:
            lstm_out = self.lstm_embedding(emb_out)
        enc_out, attns = self.encoder(lstm_out, attn_mask=enc_self_mask)
        gate_weight = torch.sigmoid(self.weightlinear(enc_out[:, -self.pred_len:, :]))
        dec_out = torch.cat((self.gatelinear(enc_out[:,-self.pred_len:, :]), gate_weight),dim=1)
        dec_out = self.norm(dec_out)
        dec_out = self.declinear(dec_out)

        if self.output_attention:
            return dec_out, attns[:,:, -self.pred_len:, :]
        else:
            return dec_out


class TFTConfigs:
    def __init__(self):
        # ------------------------- Data config --------------------------#
        ...
        # ----------------------- Embedding config -----------------------#
        self.input_size = 24
        self.d_model = 12
        self.embed_layer = 1
        self.dropout = 0.1
        # ----------------------- Enc,Dec config -------------------------#
        self.factor=5
        self.num_heads = 4
        self.activation = 'sigmoid'
        self.output_attention = False
        self.e_layers = 12
        self.d_output = 1
        # ----------------------- Training config ------------------------#
        self.pred_len = 3
        self.seq_len = 10
        # ----------------------- Infering config ------------------------#
        ...
