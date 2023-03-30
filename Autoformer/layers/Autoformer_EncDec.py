import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.signal import find_peaks,peak_widths,hilbert
from scipy.interpolate import interp1d
import scipy

def envelope(sig, distance):
    # split signal into negative and positive parts
    u_x = np.where(sig > 0)[0]
    l_x = np.where(sig < 0)[0]
    u_y = sig.copy()
    u_y[l_x] = 0
    l_y = -sig.copy()
    l_y[u_x] = 0
    
    # find upper and lower peaks
    u_peaks, _ = scipy.signal.find_peaks(u_y, distance=distance)
    l_peaks, _ = scipy.signal.find_peaks(l_y, distance=distance)
    
    # use peaks and peak values to make envelope
    u_x = u_peaks
    u_y = sig[u_peaks]
    l_x = l_peaks
    l_y = sig[l_peaks]
    
    # add start and end of signal to allow proper indexing
    end = len(sig)
    u_x = np.concatenate((u_x, [0, end]))
    u_y = np.concatenate((u_y, [0, 0]))
    l_x = np.concatenate((l_x, [0, end]))
    l_y = np.concatenate((l_y, [0, 0]))
    
    # create envelope functions
    u = scipy.interpolate.interp1d(u_x, u_y)
    l = scipy.interpolate.interp1d(l_x, l_y)
    return u, l


class my_Layernorm(nn.Module):
    """
    Special designed layernorm for the seasonal part
    """
    def __init__(self, channels):
        super(my_Layernorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class peaks_pooling(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(peaks_pooling, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.MaxPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        #print('x size: ',x.shape)

        moving_mean = self.moving_avg(x)
        #moving_mean = x + (-x)
        res = x - moving_mean
        #print(moving_mean)

        
        return res, moving_mean

class series_decomp_multi(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp_multi, self).__init__()
        self.moving_avg = [moving_avg(kernel, stride=1) for kernel in kernel_size]
        self.layer = torch.nn.Linear(1, len(kernel_size))

    def forward(self, x):
        moving_mean=[]
        for func in self.moving_avg:
            moving_avg = func(x)
            moving_mean.append(moving_avg.unsqueeze(-1))
        moving_mean=torch.cat(moving_mean,dim=-1)
        moving_mean = torch.sum(moving_mean*nn.Softmax(-1)(self.layer(x.unsqueeze(-1))),dim=-1)
        res = x - moving_mean
        return res, moving_mean 


class FourierDecomp(nn.Module):
    def __init__(self):
        super(FourierDecomp, self).__init__()
        pass

    def forward(self, x):
        x_ft = torch.fft.rfft(x, dim=-1)

class standard_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size,model_size):
        super(standard_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)
        #self.norm = nn.InstanceNorm1d(model_size, affine=True)
        self.layer_norm = nn.LayerNorm(model_size)
    def forward(self, x):
        #print('x size: ',x.shape)

        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        #te = res.cpu()
        #print(te.dtype)
        #print(res.shape)
        norm_res = self.layer_norm(res)
        #norm_res = self.norm(res.permute(0,2,1)).permute(0,2,1)
        # bias = torch.ones(res.shape)
        
        # std_res = torch.std(res,dim=1,keepdim =True,unbiased=True)
        # std_res = std_res+0.1*bias.cuda()

        return norm_res, moving_mean

class emd_standard_decomp(nn.Module):
    """
    emd decomposition block
    """
    def __init__(self, kernel_size):
        super(emd_standard_decomp, self).__init__()
        self.find_peaks = peaks_pooling(kernel_size, stride=1)
        self.moving_avg = moving_avg(kernel_size, stride=1)
    
    def forward(self,x):
        
        # up_limit = self.find_peaks(x)
        # down_limit = -self.find_peaks(-x)

        # avg_envelope = (up_limit + down_limit) / 2

        # #print(x.shape,avg_envelope.shape)


        B,L,D = x.shape
        # 计算x的分析信号及其包络
        # 将x转换为numpy.ndarray类型

        x_np = x.reshape(-1)
        x_np = x_np.detach().cpu().numpy()

        x_np_up = np.clip(x_np, 0, np.inf)
        x_np_down = np.clip(-x_np, 0, np.inf)

        #print(x_np.shape)
        # 计算x的分析信号及其包络
        # analytic_signal_up = hilbert(x_np)
        # analytic_signal_down = hilbert(-x_np)

        analytic_signal_up = hilbert(x_np_up)
        analytic_signal_down = hilbert(x_np_down)

        envelope_max = np.abs(analytic_signal_up)
        envelope_min = -np.abs(analytic_signal_down)


        envelope_max = torch.from_numpy(envelope_max).to(x.device)
        envelope_min = torch.from_numpy(envelope_min).to(x.device)

        envelope_max = envelope_max.reshape(B,L,D)
        envelope_min = envelope_min.reshape(B,L,D)



        avg_envelope = (envelope_max + envelope_min) / 2


        imf = x - avg_envelope

        bias = torch.ones(imf.shape)
        
        std_res = torch.std(imf,dim=1,keepdim =True,unbiased=True)
        std_res = std_res+0.01*bias.cuda()

        return imf/std_res, avg_envelope
    
class sci_emd_decomp(nn.Module):
    """
    emd decomposition block
    """
    def __init__(self, kernel_size):
        super(sci_emd_decomp, self).__init__()
        self.find_peaks = peaks_pooling(kernel_size, stride=1)
    
    def forward(self,x):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #print(x.shape)
        B,L,D = x.shape
        # 计算x的分析信号及其包络
        # 将x转换为numpy.ndarray类型

        x_np = x.reshape(-1)
        x_np = x_np.detach().cpu().numpy()
        
        #print(x_np.shape)
        # 计算x的分析信号及其包络
        # x_np_up = np.clip(x_np, 0, np.inf)
        # x_np_down = np.clip(-x_np, 0, np.inf)

        # analytic_signal_up = hilbert(x_np)
        # analytic_signal_down = hilbert(x_np)
        # envelope_max = np.abs(analytic_signal_up)
        # envelope_min = -np.abs(analytic_signal_down)

        up_limit,down_limit = envelope(x_np,1)
        t = np.arange(0,x_np.shape[0])
        envelope_max = up_limit(t).astype(np.float32)
        envelope_min = down_limit(t).astype(np.float32)
        #print(envelope_max.dtype)
        #print(amplitude_envelope.shape)

        # 计算包络线的上下限
        # envelope_max = np.maximum.reduce([amplitude_envelope, np.roll(amplitude_envelope, 1), np.roll(amplitude_envelope, -1)])
        # envelope_min = np.minimum.reduce([amplitude_envelope, np.roll(amplitude_envelope, 1), np.roll(amplitude_envelope, -1)])

        # 将结果转换为torch.Tensor类型
        # up,down = envelope(x_np,1)
        # t = np.arange(0,x_np.shape[0])
        # envelope_max = up(t).astype(np.float32)
        # envelope_min = down(t).astype(np.float32)
        #print(envelope_max.dtype)

        envelope_max = torch.from_numpy(envelope_max).to(x.device)
        envelope_min = torch.from_numpy(envelope_min).to(x.device)

        envelope_max = envelope_max.reshape(B,L,D)
        envelope_min = envelope_min.reshape(B,L,D)

        #print(envelope_max.shape)

        avg_envelope = (envelope_max + envelope_min) / 2

        #print("test!!!!",avg_envelope)

        #print(x.shape,avg_envelope.shape)

        imf = x - avg_envelope
        #norm = nn.InstanceNorm1d(37)
        #norm_imf = norm(imf.permute(0,2,1)).permute(0,2,1)

        #bias = torch.ones(imf.shape)
        
        #std_res = torch.std(imf,dim=1,keepdim =True,unbiased=True)
        #std_res = std_res+0.1*bias.cuda()

        return imf, avg_envelope

class emd_decomp(nn.Module):
    """
    emd decomposition block
    """
    def __init__(self, kernel_size):
        super(emd_decomp, self).__init__()
        self.find_peaks = peaks_pooling(kernel_size, stride=1)
    
    def forward(self,x):
        
        up_limit = self.find_peaks(x)
        down_limit = -self.find_peaks(-x)

        avg_envelope = (up_limit + down_limit) / 2

        #print(x.shape,avg_envelope.shape)
        #print("test!!!!",avg_envelope)

        imf = x - avg_envelope

        return imf, avg_envelope

    # def forward(self, x):
    #     # moving_mean = self.moving_avg(x)
    #     # res = x - moving_mean
    #     fs = 1000
    #     num_signal = x.shape[1]

    #     res = []
    #     avg = []

    #     for x_b in x:
    #         num_interval = 50
    #         serilized_x = semd.concatenate(x_b.T, num_interval).reshape(-1)
    #         tAxis = np.linspace(0,serilized_x.shape[0])

    #         upper_peaks, _ = find_peaks(serilized_x)
    #         lower_peaks, _ = find_peaks(-serilized_x)

    #         f1 = interp1d(upper_peaks/fs,serilized_x[upper_peaks], kind = 'cubic', fill_value = 'extrapolate')
    #         f2 = interp1d(lower_peaks/fs,serilized_x[lower_peaks], kind = 'cubic', fill_value = 'extrapolate')

    #         y1 = f1(tAxis)
    #         y2 = f2(tAxis)

    #         y1[0:5] = 0
    #         y1[-5:] = 0
    #         y2[0:5] = 0
    #         y2[-5:] = 0
    #         avg_envelope = (y1 + y2) / 2

    #         res1 = avg_envelope
    #         imf1 = serilized_x - avg_envelope

    #         imfs = semd.deconcatenate(imf1, num_interval, num_signal)

    #         res.append(imf1)
    #         avg.append(avg_envelope)

    #     res = np.array(res)
    #     avg = np.array(avg)

    #     return res, avg

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class EncoderLayer(nn.Module):
    """
    Autoformer encoder layer with the progressive decomposition architecture
    """
    def __init__(self, attention, d_model, d_ff=None, moving_avg=25, dropout=0.1, activation="relu",id=0):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        # self.decomp1 = emd_decomp(3)
        # self.decomp2 = emd_decomp(3)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.id = id
        self.out = False

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)
        x, _ = self.decomp1(x)
        if self.out:
            self.out = False
            x_0 = x[:,:,0].detach().cpu().numpy()
            x_1 = x[:,:,-1].detach().cpu().numpy()
            x_0 = x_0.flatten()
            x_1 = x_1.flatten()
            x_0 = np.expand_dims(x_0, axis=1)
            x_1 = np.expand_dims(x_1, axis=1)
            out = np.concatenate((x_0,x_1),axis=1)
            np.savetxt('STD_batch_output'+str(self.id)+'.csv',out,delimiter=',')
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        res, _ = self.decomp2(x + y)
        return res, attn

class sigmaEncoderLayer(nn.Module):
    """
    Autoformer encoder layer with the progressive decomposition architecture
    """
    def __init__(self, attention, d_model, d_ff=None, moving_avg=25, dropout=0.1, activation="relu"):
        super(sigmaEncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = standard_decomp(moving_avg,d_model)
        self.decomp2 = standard_decomp(moving_avg,d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)
        x, _ = self.decomp1(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        res, _ = self.decomp2(x + y)
        return res, attn

class sigmaEMDEncoderLayer(nn.Module):
    """
    Autoformer encoder layer with the progressive decomposition architecture
    """
    def __init__(self, attention, d_model, d_ff=None, moving_avg=25, dropout=0.1, activation="relu"):
        super(sigmaEMDEncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = emd_standard_decomp(moving_avg)
        self.decomp2 = emd_standard_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)
        x, _ = self.decomp1(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        res, _ = self.decomp2(x + y)
        return res, attn


class EMDEncoderLayer(nn.Module):
    """
    Autoformer encoder layer with the progressive decomposition architecture
    """
    def __init__(self, attention, d_model, d_ff=None, moving_avg=25, dropout=0.1, activation="relu"):
        super(EMDEncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        # self.decomp1 = series_decomp(moving_avg)
        # self.decomp2 = series_decomp(moving_avg)
        self.decomp1 = emd_decomp(moving_avg)
        self.decomp2 = emd_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)
        x, _ = self.decomp1(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        res, _ = self.decomp2(x + y)
        return res, attn

class sci_EMDEncoderLayer(nn.Module):
    """
    Autoformer encoder layer with the progressive decomposition architecture
    """
    def __init__(self, attention, d_model, d_ff=None, moving_avg=25, dropout=0.1, activation="relu"):
        super(sci_EMDEncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        # self.decomp1 = series_decomp(moving_avg)
        # self.decomp2 = series_decomp(moving_avg)
        self.decomp1 = sci_emd_decomp(moving_avg)
        self.decomp2 = sci_emd_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.out = False

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)
        x, t = self.decomp1(x)

        #print(out.shape)
        if self.out:
            self.out = False
            x_0 = x[:,:,-1].detach().cpu().numpy()
            t_0 = t[:,:,-1].detach().cpu().numpy()
            x_0 = x_0.flatten()
            t_0 = t_0.flatten()
            x_0 = np.expand_dims(x_0, axis=1)
            t_0 = np.expand_dims(t_0, axis=1)
            out = np.concatenate((x_0,t_0),axis=1)
            np.savetxt('batch_output.csv',out,delimiter=',')

        #print(x.shape,t.shape)

        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        res, _ = self.decomp2(x + y)
        return res, attn

class no_decomp_EncoderLayer(nn.Module):
    """
    Autoformer encoder layer with the progressive decomposition architecture
    """
    def __init__(self, attention, d_model, d_ff=None, moving_avg=25, dropout=0.1, activation="relu"):
        super(no_decomp_EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)
        #x, _ = self.decomp1(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        #res, _ = self.decomp2(x + y)
        return (x+y), attn


class Encoder(nn.Module):
    """
    Autoformer encoder
    """
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    """
    Autoformer decoder layer with the progressive decomposition architecture
    """
    def __init__(self, self_attention, cross_attention, d_model, c_out, d_ff=None,
                 moving_avg=25, dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.decomp3 = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1, padding=1,
                                    padding_mode='circular', bias=False)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x, trend1 = self.decomp1(x)
        #print("trend: ",trend1.shape)
        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])
        x, trend2 = self.decomp2(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x, trend3 = self.decomp3(x + y)

        residual_trend = trend1 + trend2 + trend3
        residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)
        #print("residual: ",residual_trend.shape)
        return x, residual_trend

class sigmaDecoderLayer(nn.Module):
    """
    Autoformer decoder layer with the progressive decomposition architecture
    """
    def __init__(self, self_attention, cross_attention, d_model, c_out, d_ff=None,
                 moving_avg=25, dropout=0.1, activation="relu"):
        super(sigmaDecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = standard_decomp(moving_avg,d_model)
        self.decomp2 = standard_decomp(moving_avg,d_model)
        self.decomp3 = standard_decomp(moving_avg,d_model)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1, padding=1,
                                    padding_mode='circular', bias=False)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x, trend1 = self.decomp1(x)
        #print("trend: ",trend1.shape)
        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])
        x, trend2 = self.decomp2(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x, trend3 = self.decomp3(x + y)

        residual_trend = trend1 + trend2 + trend3
        residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)
        #print("residual: ",residual_trend.shape)
        return x, residual_trend

class sigmaEMDDecoderLayer(nn.Module):
    """
    Autoformer decoder layer with the progressive decomposition architecture
    """
    def __init__(self, self_attention, cross_attention, d_model, c_out, d_ff=None,
                 moving_avg=25, dropout=0.1, activation="relu"):
        super(sigmaEMDDecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = emd_standard_decomp(moving_avg)
        self.decomp2 = emd_standard_decomp(moving_avg)
        self.decomp3 = emd_standard_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1, padding=1,
                                    padding_mode='circular', bias=False)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x, trend1 = self.decomp1(x)
        #print("trend: ",trend1.shape)
        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])
        x, trend2 = self.decomp2(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x, trend3 = self.decomp3(x + y)

        residual_trend = trend1 + trend2 + trend3
        residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)
        #print("residual: ",residual_trend.shape)
        return x, residual_trend
    
class SE_DecoderLayer(nn.Module):
    """
    Autoformer decoder layer with the progressive decomposition architecture
    """
    def __init__(self, self_attention, cross_attention, d_model, c_out, d_ff=None,
                 moving_avg=25, dropout=0.1, activation="relu"):
        super(SE_DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.decomp3 = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1, padding=1,
                                    padding_mode='circular', bias=False)
        self.test_projection = nn.Conv1d(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=1,
                                    padding_mode='circular', bias=False)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x, trend1 = self.decomp1(x)
        #print("trend: ",trend1.shape)
        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])
        x, trend2 = self.decomp2(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x, trend3 = self.decomp3(x + y)

        trend = torch.stack([trend1,trend2,trend3],dim=1)
        #trend = trend.permute(0,2,1,3)
        #print(trend.shape)
        #out = self.se(trend)
        #print(out.shape)
        #out = out.reshape(32,3,-1)
        #conv_out = self.test_projection(out)
        #conv_out = torch.sum(out,dim=1)
        #print("conv_: ",conv_out.shape)
        #conv_out = conv_out.reshape(32,1,20,64).permute(0,2,3,1)

        residual_trend = trend1 + trend2 + trend3
        #residual_trend = conv_out
        residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)
        #print("residual: ",residual_trend.shape)
        return x, trend

class SE_EMDDecoderLayer(nn.Module):
    """
    Autoformer decoder layer with the progressive decomposition architecture
    """
    def __init__(self, self_attention, cross_attention, d_model, c_out, d_ff=None,
                 moving_avg=25, dropout=0.1, activation="relu"):
        super(SE_EMDDecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = sci_emd_decomp(moving_avg)
        self.decomp2 = sci_emd_decomp(moving_avg)
        self.decomp3 = sci_emd_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1, padding=1,
                                    padding_mode='circular', bias=False)
        self.test_projection = nn.Conv1d(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=1,
                                    padding_mode='circular', bias=False)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x, trend1 = self.decomp1(x)
        #print("trend: ",trend1.shape)
        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])
        x, trend2 = self.decomp2(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x, trend3 = self.decomp3(x + y)

        trend = torch.stack([trend1,trend2,trend3],dim=1)
        #trend = trend.permute(0,2,1,3)
        #print(trend.shape)
        #out = self.se(trend)
        #print(out.shape)
        #out = out.reshape(32,3,-1)
        #conv_out = self.test_projection(out)
        #conv_out = torch.sum(out,dim=1)
        #print("conv_: ",conv_out.shape)
        #conv_out = conv_out.reshape(32,1,20,64).permute(0,2,3,1)

        residual_trend = trend1 + trend2 + trend3
        #residual_trend = conv_out
        residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)
        #print("residual: ",residual_trend.shape)
        return x, trend

class SE_sigmaDecoderLayer(nn.Module):
    """
    Autoformer decoder layer with the progressive decomposition architecture
    """
    def __init__(self, self_attention, cross_attention, d_model, c_out, d_ff=None,
                 moving_avg=25, dropout=0.1, activation="relu"):
        super(SE_sigmaDecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = standard_decomp(moving_avg,d_model)
        self.decomp2 = standard_decomp(moving_avg,d_model)
        self.decomp3 = standard_decomp(moving_avg,d_model)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1, padding=1,
                                    padding_mode='circular', bias=False)
        self.test_projection = nn.Conv1d(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=1,
                                    padding_mode='circular', bias=False)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x, trend1 = self.decomp1(x)
        #print("trend: ",trend1.shape)
        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])
        x, trend2 = self.decomp2(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x, trend3 = self.decomp3(x + y)

        trend = torch.stack([trend1,trend2,trend3],dim=1)
        #trend = trend.permute(0,2,1,3)
        #print(trend.shape)
        #out = self.se(trend)
        #print(out.shape)
        #out = out.reshape(32,3,-1)
        #conv_out = self.test_projection(out)
        #conv_out = torch.sum(out,dim=1)
        #print("conv_: ",conv_out.shape)
        #conv_out = conv_out.reshape(32,1,20,64).permute(0,2,3,1)

        residual_trend = trend1 + trend2 + trend3
        #residual_trend = conv_out
        residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)
        #print("residual: ",residual_trend.shape)
        return x, trend

class EMDDecoderLayer(nn.Module):
    """
    Autoformer decoder layer with the progressive decomposition architecture
    """
    def __init__(self, self_attention, cross_attention, d_model, c_out, d_ff=None,
                 moving_avg=25, dropout=0.1, activation="relu"):
        super(EMDDecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = emd_decomp(moving_avg)
        self.decomp2 = emd_decomp(moving_avg)
        self.decomp3 = emd_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1, padding=1,
                                    padding_mode='circular', bias=False)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x, trend1 = self.decomp1(x)
        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])
        x, trend2 = self.decomp2(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x, trend3 = self.decomp3(x + y)

        residual_trend = trend1 + trend2 + trend3
        residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)
        return x, residual_trend

class sci_EMDDecoderLayer(nn.Module):
    """
    Autoformer decoder layer with the progressive decomposition architecture
    """
    def __init__(self, self_attention, cross_attention, d_model, c_out, d_ff=None,
                 moving_avg=25, dropout=0.1, activation="relu"):
        super(sci_EMDDecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = sci_emd_decomp(moving_avg)
        self.decomp2 = sci_emd_decomp(moving_avg)
        self.decomp3 = sci_emd_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1, padding=1,
                                    padding_mode='circular', bias=False)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x, trend1 = self.decomp1(x)
        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])
        x, trend2 = self.decomp2(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x, trend3 = self.decomp3(x + y)

        residual_trend = trend1 + trend2 + trend3
        #print("test!!!",residual_trend)
        residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)
        return x, residual_trend


class Decoder(nn.Module):
    """
    Autoformer encoder
    """
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        for layer in self.layers:
            x, residual_trend = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            trend = trend + residual_trend

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x, trend

class SE_Decoder(nn.Module):
    """
    Autoformer encoder
    """
    def __init__(self, layers, norm_layer=None, projection=None,configs=None):
        super(SE_Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection
        print(3*configs.d_layers)
        self.se = SELayer(channel=3*configs.d_layers)
        self.flags = 0

    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        ttrend = None
        for layer in self.layers:
            x, residual_trend = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            if self.flags==0:
                ttrend = residual_trend
                self.flags = 1
            else:
                ttrend = torch.cat([ttrend,residual_trend],dim=1)
        #print("test",ttrend)
        out = self.se(ttrend)
        ttrend = torch.sum(out,dim=1)
        self.flags = 0

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
            ttrend = self.projection(ttrend)
        trend = trend+ttrend
        return x, trend