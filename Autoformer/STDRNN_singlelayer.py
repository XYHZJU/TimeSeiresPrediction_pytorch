import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_Encdec import series_decomp


class STDSeq2Seq_SingleL(nn.Module):
    def __init__(self, configs):
        super(STDSeq2Seq_SingleL, self).__init__()
        self.train = configs.Seq2SeqtrainState

        self.decompsition = series_decomp(configs.kernel_size)
        self.encoder_Seasonal = nn.GRU(configs.enc_in, configs.d_model, configs.num_layer, batch_first=True, dropout=configs.dropout)
        self.decoder_Seasonal = nn.GRU(configs.enc_out, configs.d_model, configs.num_layer, batch_first=True, dropout=configs.dropout)
        self.outlinear_Seansonal = nn.Linear(configs.d_model, configs.enc_out)
        self.encoder_Trend = nn.GRU(configs.enc_in, configs.d_model, configs.num_layer, batch_first=True, dropout=configs.dropout)
        self.decoder_Trend = nn.GRU(configs.enc_out, configs.d_model, configs.num_layer, batch_first=True, dropout=configs.dropout)
        self.outlinear_Trend = nn.Linear(configs.d_model, configs.enc_out)
        self.target_idx_loc = configs.target_idx_loc
    
    def forward(self, source, target):
        B, S, E = target.shape
        source_seasonal, source_trend = self.decompsition(source)
        _, hidden_seasonal = self.encoder_Seasonal(source_seasonal[:,:-1,:])
        _, hidden_trend = self.encoder_Trend(source_trend[:,:-1,:])

        x_seasonal = source_seasonal[:,-1,self.target_idx_loc]
        x_trend = source_trend[:,-1,self.target_idx_loc]
        outputs = torch.zeros(B, S, E).to(source.device)
        if self.train: # teacher forcing
            target_seasonal, target_trend = self.decompsition(target[:, :, :])
            for t in range(S):
                x_seasonal = x_seasonal.unsqueeze(1)
                x_trend = x_trend.unsqueeze(1)
                output_seasonal, hidden_seasonal = self.decoder_Seasonal(x_seasonal, hidden_seasonal)
                output_seasonal = self.outlinear_Seansonal(output_seasonal.squeeze(1))
                output_trend, hidden_trend = self.decoder_Trend(x_trend, hidden_trend)
                output_trend = self.outlinear_Trend(output_trend.squeeze(1))
                outputs[t] = output_seasonal + output_trend
                x_seasonal = target_seasonal[t]
                x_trend = target_trend[t]
        else: # dynamic decoding
            for t in range(S):
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