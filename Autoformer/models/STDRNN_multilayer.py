import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.STDRNN_EncDec import RNNEncoderLayer, RNNDecoderLayer, STDRNNDecoder
from layers.Autoformer_EncDec import Encoder
from layers.Embed import DataEmbedding_wo_pos


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.dec_out = configs.dec_out

        # Embedding
        self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        # Encoder
        self.encoder = Encoder([
            RNNEncoderLayer(configs.d_model, configs.moving_avg, 
            configs.dropout, configs.activation) for l in range(configs.e_layers)
        ])
        # Decoder
        self.decoder = STDRNNDecoder([
            RNNDecoderLayer(configs.d_model, configs.dec_out, 
            configs.moving_avg, configs.dropout, configs.activation) for l in range(configs.e_layers)
        ])


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, init_hidden = self.encoder(enc_out, attn_mask=None)
        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, init_hidden, trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part
        
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]