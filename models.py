import torch
import torch.nn as nn
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
            nn.Linear(d_model*2, 2), 
            nn.Softmax(dim=-1)
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

        return self.decoder(cat_sig)

if __name__ == '__main__':
    model = Soil2ClassModel()
    a = torch.randn(2, 3, 96)
    b = torch.randn(2, 3, 96)
    print(model(a, b).size())
