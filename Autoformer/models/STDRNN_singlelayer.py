import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.totrain = configs.Seq2SeqtrainState
        self.pred_len = configs.pred_len
        self.dec_out = configs.c_out

        self.decompsition = series_decomp(configs.moving_avg)
        self.encoder_Seasonal = nn.GRU(configs.enc_in, configs.d_model, configs.rnn_layers, batch_first=True, dropout=configs.dropout)
        self.decoder_Seasonal = nn.GRU(configs.c_out, configs.d_model, configs.rnn_layers, batch_first=True, dropout=configs.dropout)
        self.outlinear_Seansonal = nn.Linear(configs.d_model, configs.c_out)

        self.encoder_Trend = nn.GRU(configs.enc_in, configs.d_model, configs.rnn_layers, batch_first=True, dropout=configs.dropout)
        self.decoder_Trend = nn.GRU(configs.c_out, configs.d_model, configs.rnn_layers, batch_first=True, dropout=configs.dropout)
        self.outlinear_Trend = nn.Linear(configs.d_model, configs.c_out)
        # self.target_idx_loc = configs.target_idx_loc
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        B, L, E = x_enc.shape
        source_seasonal, source_trend = self.decompsition(x_enc)
        _, hidden_seasonal = self.encoder_Seasonal(source_seasonal[:,:-1,:])
        _, hidden_trend = self.encoder_Trend(source_trend[:,:-1,:])

        x_seasonal = source_seasonal[:,-1,-self.dec_out:]
        x_trend = source_trend[:,-1,-self.dec_out:]
        outputs = torch.zeros(B, self.pred_len, self.dec_out).to(x_enc.device)
        # print('output:',outputs.shape)
        if self.totrain: # teacher forcing
            target_seasonal, target_trend = self.decompsition(x_dec[:, :, :])
            if(self.pred_len>1):
                target_seasonal = target_seasonal.permute(1,0,2)
                target_trend = target_trend.permute(1,0,2)
            # print('test0: ', target_seasonal.shape, target_trend.shape)
            for t in range(self.pred_len):
                x_seasonal = x_seasonal.unsqueeze(1)
                x_trend = x_trend.unsqueeze(1)
                # print('test: ',x_seasonal.shape, x_trend.shape)
                # print('test1: ',t,x_seasonal.shape,hidden_seasonal.shape)

                output_seasonal, hidden_seasonal = self.decoder_Seasonal(x_seasonal, hidden_seasonal)
                # print('test2:', output_seasonal.shape, hidden_seasonal.shape)
                output_seasonal = self.outlinear_Seansonal(output_seasonal.squeeze(1))

                output_trend, hidden_trend = self.decoder_Trend(x_trend, hidden_trend)
                output_trend = self.outlinear_Trend(output_trend.squeeze(1))

                # print('test2:',output_seasonal.shape, output_trend.shape)
                outputs = outputs.permute(1,0,2)
                outputs[t] = output_seasonal + output_trend

                x_seasonal = target_seasonal[t]
                x_trend = target_trend[t]
                outputs = outputs.permute(1,0,2)
        else: # dynamic decoding
            for t in range(self.pred_len):
                x_seasonal = x_seasonal.unsqueeze(1)
                x_trend = x_trend.unsqueeze(1)
                output_seasonal, hidden_seasonal = self.decoder_Seasonal(x_seasonal, hidden_seasonal)
                output_seasonal = self.outlinear_Seansonal(output_seasonal.squeeze(1))
                output_trend, hidden_trend = self.decoder_Trend(x_trend, hidden_trend)
                output_trend = self.outlinear_Trend(output_trend.squeeze(1))
                outputs[t] = output_seasonal + output_trend
                x_seasonal = output_seasonal
                x_trend = output_trend
        return outputs