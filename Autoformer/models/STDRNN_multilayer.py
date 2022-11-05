import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.STDRNN_EncDec import RNNEncoderLayer, RNNDecoderLayer, STDRNNDecoder
from layers.Autoformer_EncDec import Encoder, series_decomp

from layers.Embed import DataEmbedding_wo_pos


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.label_len = configs.label_len
        self.dec_out = configs.c_out
        self.decomp = series_decomp(configs.moving_avg)

        # Embedding
        self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        # Encoder
        self.encoder = Encoder([
            RNNEncoderLayer(configs.d_model, configs.moving_avg, 
            configs.dropout, configs.activation) for l in range(configs.rnn_layers)
        ])
        # Decoder
        self.decoder = STDRNNDecoder([
            RNNDecoderLayer(configs.d_model, configs.c_out, 
            configs.moving_avg, configs.dropout, configs.activation) for l in range(configs.rnn_layers)
        ])
        self.outlinear_Seansonal = nn.Linear(configs.d_model, configs.c_out)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # decomp init
        # print('dim: ',x_enc.shape, x_dec.shape)
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        # enc
        # print('enc_emb')
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, init_hidden = self.encoder(enc_out, attn_mask=None)
        # dec
        # print('sshape: ',seasonal_init.shape, x_mark_dec.shape)
        # print('dec_emb')
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, init_hidden, trend=trend_init)
        # final
        # print('size: ', dec_out.shape, trend_part.shape, seasonal_part.shape)
        seasonal_part = self.outlinear_Seansonal(seasonal_part.squeeze(1))

        dec_out = trend_part + seasonal_part
        
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]