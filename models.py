import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class Soil2ClassModel(nn.Module):

    def __init__(self, d_model=64, nhead=4, dim_feedforward=256):
        super(Soil2ClassModel, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.expend_dims = nn.Sequential(
            nn.Conv1d(3, d_model, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )
        self.encoder = nn.Sequential(
            PositionalEncoding(d_model),
            nn.TransformerEncoder(encoder_layer, 3)  # seq_len, bs, 3
        )
        self.decoder = nn.Sequential(
            nn.Linear(d_model*2, 2)
        )
    
    def forward(self, sig1, sig2):
        """ sig1/sig2: shape=(bs, ch, len)
        """
        ex_sig1, ex_sig2 = self.expend_dims(sig1), self.expend_dims(sig2)
        # |ex_sig|: (bs, nkernel, len)
        ex_sig1, ex_sig2 = ex_sig1.permute(2, 0, 1), ex_sig2.permute(2, 0, 1)
        # |ex_sig|: (len, bs, nkernel)
        encoded_sig1, encoded_sig2 = self.encoder(ex_sig1)[-1], self.encoder(ex_sig2)[-1]
        # |encoded_sig|: (bs, d_model) nkernel=d_model
        cat_sig = torch.cat([encoded_sig1, encoded_sig2], dim=1)
        # |cat_sig|: (bs, d_model * 2)

        return F.log_softmax(self.decoder(cat_sig), dim=-1)

class Soil2ClassGRUModel(nn.Module):

    def __init__(self, nhiddens=80):
        super(Soil2ClassGRUModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.GRU(3, nhiddens, 2, bidirectional=True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(nhiddens*2, 2)
        )
        self.nhiddens = nhiddens

    def forward(self, sig1, sig2):
        ex_sig1, ex_sig2 = sig1.permute(2, 0, 1), sig2.permute(2, 0, 1)
        # |ex_sig|: (len, bs, nkernel)
        _, hn_sig1 = self.encoder(ex_sig1) # (4, bn, hidden_size)
        _, hn_sig2 = self.encoder(ex_sig2) # (4, bn, hidden_size)
        
        hn_sig1, hn_sig2 = hn_sig1.permute(1,0,2).view(-1, self.nhiddens * 4), \
            hn_sig2.permute(1,0,2).view(-1, self.nhiddens * 4)
        
        # |encoded_sig|: (bs, d_model) nkernel=d_model
        cat_sig = torch.cat([hn_sig1, hn_sig2], dim=1)
        print(cat_sig.size())
        return self.decoder(cat_sig)

class GRUClassifier(nn.Module):

    def __init__(self, n_h=40, n_gru=2, n_ch=3, n_class=3):
        super(GRUClassifier, self).__init__()
        
        self.gru = nn.GRU(n_ch, n_h, n_gru, bidirectional=True)
        self.conv = nn.Sequential(
            self.gen_convblock(n_h*2, 16, 7, 3),
            self.gen_convblock(16, 16, 7, 2),
            self.gen_convblock(16, 8, 7, 2),
            nn.Flatten(1, -1),
            nn.Linear(64, n_class)
        )
    
    def forward(self, sig):
        sig = sig.permute(2, 0, 1)
        # |sig|: (len, bs, ch)
        encoded_sig, _ = self.gru(sig)
        # |encoded_sig|: (len, bs, nhiddens)
        encoded_sig = encoded_sig.permute(1, 2, 0)
        # |encoded_sig|: (bs, nhiddens, len)
        conved_sig = self.conv(encoded_sig)

        return F.log_softmax(conved_sig, dim=-1)
    
    def gen_convblock(self, in_ch, out_ch, k_size, p_size):
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, k_size, 1, (k_size-1)//2),
            nn.AvgPool1d(p_size),
            nn.BatchNorm1d(out_ch),
            nn.ReLU()
        )

if __name__ == '__main__':
    model = GRUClassifier()
    a = torch.randn(2, 3, 96)
    b = torch.randn(2, 3, 96)
    print(model(a).size())
