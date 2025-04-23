import torch
import torch.nn as nn
import torchvision.models as models


class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, max_len=5000):
        super().__init__()
        positional_encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, model_dim, 2) * (-torch.log(torch.tensor(10000.0)) / model_dim)
        )
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        self.pe = positional_encoding.unsqueeze(0)  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

# encoder class, uses resnet --> ask Dave if that's okay 
# using CNN since good at extracting spatial features from images 
class CNNEncoder(nn.Module):
    def __init__(self, model_dim):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, model_dim)
        self.batch_norm = nn.BatchNorm1d(model_dim, momentum=0.01)

    def forward(self, images):
        with torch.no_grad():
            features = self.resnet(images).squeeze()
        features = self.linear(features)
        return self.batch_norm(features)  # (batch_size, model_dim)

# transformer decoder
# chose transformer because good at sequential info 
class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None):
        tgt_emb = self.embedding(tgt) * (memory.size(-1) ** 0.5)
        tgt_emb = self.pos_encoding(tgt_emb)
        tgt_emb = tgt_emb.transpose(0, 1)  # (T, N, E)
        memory = memory.unsqueeze(0)  # (1, N, E)
        output = self.transformer_decoder(
            tgt_emb, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask
        )
        return self.fc_out(output.transpose(0, 1))  # (N, T, vocab_size)


class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, pad_idx=0):
        super().__init__()
        self.encoder = CNNEncoder(d_model)
        self.decoder = TransformerDecoder(vocab_size, d_model, nhead, num_layers, dim_feedforward, pad_idx)

    def forward(self, images, captions, tgt_mask=None, tgt_key_padding_mask=None):
        memory = self.encoder(images)  # (batch_size, d_model)
        output = self.decoder(captions, memory, tgt_mask, tgt_key_padding_mask)
        return output
