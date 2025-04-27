import torch, math
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0,max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0,d_model,2).float() * (-math.log(10000)/d_model))
        pe[:,0::2] = torch.sin(pos*div)
        pe[:,1::2] = torch.cos(pos*div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:,:x.size(1)]

class TransformerStoryGenerator(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=4, 
                 num_layers=2, dim_ff=512, dropout=0.1, max_len=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_ff,
            dropout=dropout
        )
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None):
        # Embedding + positional encoding
        src_emb = self.pos_enc(self.embed(src) * math.sqrt(self.embed.embedding_dim))
        tgt_emb = self.pos_enc(self.embed(tgt) * math.sqrt(self.embed.embedding_dim))

        # PyTorch Transformer expects (seq, batch, dim)
        memory = self.transformer.encoder(
            src_emb.transpose(0, 1),
            src_key_padding_mask=src_key_padding_mask
        )
        out = self.transformer.decoder(
            tgt_emb.transpose(0, 1),
            memory,
	    tgt_mask=nn.Transformer.generate_square_subsequent_mask(tgt_emb.size(1)).to(tgt.device),
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        # back to (batch, seq, dim) and project to vocab size
        return self.fc_out(out.transpose(0, 1))
