import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=512):
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


class TransformerModel(nn.Module):

    #     def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
    def __init__(self):
        super(TransformerModel, self).__init__()

        embed_dim = 1
        in_dim = 1
        out_dim = 1

        self.positional_encoding = PositionalEncoding(embed_dim)

        self.encoder_embedding = torch.nn.Linear(
            in_features=in_dim, out_features=embed_dim
        )
        self.decoder_embedding = torch.nn.Linear(
            in_features=out_dim, out_features=embed_dim
        )

        self.ninp = 1
        self.encoder_embedding = nn.Linear(512, 32)
        self.decoder_embedding = nn.Linear(32, 512)

        self.transformer_layers = nn.Transformer(nhead=4, num_encoder_layers=4, d_model=512)

        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, src):
        #         src = self.encoder(src) * math.sqrt(self.ninp)
        # src = self.positional_encoding(src)
        # src = self.encoder_embedding(src)
        #
        # tgt = self.positional_encoding(tgt)
        # tgt = self.encoder_embedding(tgt)
        # src = self.encoder_embedding(src)

        # output = self.transformer_layers(src, tgt)
        output = self.transformer_encoder(src)

        # output = F.log_softmax(output, dim=-1)
        # output = nn.Softmax(dim=0)(output)
        # output = self.decoder_embedding(output)

        # softmax 层
        output = F.softmax(output, dim=2)

        # output = self.unet(src)
        return output



def model_precision(det_bottom_ids, pre_bottom_ids, rel = 0.025):
    nc = 0
    for i in range(len(pre_bottom_ids)):
        if abs(pre_bottom_ids[i] - det_bottom_ids[i]) <= rel * det_bottom_ids[i]:
            nc = nc + 1
    return nc / len(pre_bottom_ids) * 100