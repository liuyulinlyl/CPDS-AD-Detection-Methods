import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math
import numpy as np
import math
from math import sqrt
import os


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        # Initialize the positional encoding tensor.
        pe.require_grad = False
        # Keep positional encodings fixed during backpropagation.
        position = torch.arange(0, max_len).float().unsqueeze(1)  # Position sequence
        # Shape: (max_len, 1)
        # Convert positions to float and add a singleton dimension.
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        # Frequency term for sinusoidal positional encoding.
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Use sine for even dimensions and cosine for odd dimensions.
        pe = pe.unsqueeze(0)
        # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)
        # Register fixed positional encodings as a buffer.
    def forward(self, x):
        return self.pe[:, :x.size(1)]   # Match the position embedding length to x.
    
'''    
batch_size = 32
win_size = 100
d_model = 512
x = torch.randn(batch_size,win_size,d_model)
PE_model = PositionalEmbedding(d_model=d_model)
pe = PE_model(x)
print(pe.shape)
#torch.Size([1, 100, 512])
''' 

class TokenEmbedding(nn.Module):  # Inherits from nn.Module
    # 1D convolution embedding for each sequence step.
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        # Create a 1D convolution layer with circular padding.
        for m in self.modules(): # Iterate through submodules.
            if isinstance(m, nn.Conv1d):
            # Initialize Conv1d weights.
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        # Apply Conv1d in channel-first format, then restore sequence-first format.
        return x

'''
batch_size = 32
win_size = 100
d_model = 512
c_in = 132
x = torch.randn(batch_size,win_size,c_in)
TE_model = TokenEmbedding(c_in=c_in,d_model=d_model)
te = TE_model(x)
print(te.shape)
#torch.Size([32, 100, 512])
'''

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.0):
        super(DataEmbedding, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)
        # Dropout for regularization.

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)
'''
batch_size = 32
win_size = 100
d_model = 512
c_in = 132
x = torch.randn(batch_size,win_size,c_in)
DE_model = DataEmbedding(c_in=c_in,d_model=d_model)
de = DE_model(x)
print(de.shape)
#torch.Size([32, 100, 512])
'''

class TriangularCausalMask():  # Causal mask for preventing attention to future positions.
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)
    # Upper-triangular boolean mask.
    @property
    def mask(self):
        return self._mask


class Attention_block(nn.Module):
    def __init__(self, mask_flag=True, scale=None, attention_dropout=0.0):
        super(Attention_block, self).__init__() 
        self.scale = scale  # Scale factor
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape  
        # queries shape: (B, L, H, E)
        _, S, _, D = values.shape
        # values shape: (B, S, H, D)
        scale = self.scale or 1. / sqrt(E)
        # Use the default scaled dot-product factor if no scale is provided.

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        # Dot-product attention scores.
        if self.mask_flag:  #mask
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
            # Mask future positions with negative infinity.
        attn = scale * scores  # Scale attention scores

        series = self.dropout(torch.softmax(attn, dim=-1))
        # Softmax attention distribution with dropout.
        V = torch.einsum("bhls,bshd->blhd", series, values) 
        # Weighted sum of values.

        return V.contiguous()  # Return the attention output.


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads) # Default key dimension per head
        d_values = d_values or (d_model // n_heads) # Default value dimension per head
        self.norm = nn.LayerNorm(d_model)
        # Layer normalization over the model dimension.
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model,
                                          d_keys * n_heads)
        # Project queries into attention heads.
        self.key_projection = nn.Linear(d_model,
                                        d_keys * n_heads)
        # Project keys into attention heads.
        self.value_projection = nn.Linear(d_model,
                                          d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        # Project attention output back to model dimension.

        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape # queries shape: (B, L, D)
        _, S, _ = keys.shape # keys shape: (B, S, D)
        H = self.n_heads
        x = queries
        queries = self.query_projection(queries).view(B, L, H, -1)  
        # Reshape projections into attention heads.
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out= self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        # Apply the inner attention mechanism.
        out = out.view(B, L, -1) # Shape: (B, L, D)
        return self.out_projection(out)


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model  # Default feed-forward dimension
        self.attention = attention  
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        # Project from model dimension to feed-forward dimension.
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        # Project back to model dimension.
        self.norm1 = nn.LayerNorm(d_model)
        # Layer normalization for the encoder block.
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        # Dropout for regularization.
        self.activation = F.relu if activation == "relu" else F.gelu
        # Select the activation function.

    def forward(self, x, attn_mask=None): # Inputs: x and optional attention mask
        new_x = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)  # Normalize before the feed-forward block.
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))  
        # Switch to channel-first format for Conv1d.
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y)


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        # Store attention layers as a ModuleList.
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        series_list = []  # Reserved for attention series outputs.
        prior_list = []   # Reserved for attention prior outputs.
        for attn_layer in self.attn_layers: # Iterate through attention layers.
            x= attn_layer(x, attn_mask=attn_mask)
            # Apply the current attention layer.

        if self.norm is not None:
            x = self.norm(x) # Apply final normalization.

        return x


class Transformer(nn.Module):
    def __init__(self,enc_in, c_out, d_model=512, n_heads=8, e_layers=3, d_ff=512,
                 dropout=0.0, activation='gelu'):
        # Transformer encoder configuration.
        super(Transformer, self).__init__()

        self.embedding = DataEmbedding(enc_in, d_model, dropout)
        # Input value and positional embedding.

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        Attention_block(False, attention_dropout=dropout),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.projection = nn.Linear(d_model, c_out, bias=True)  # Project to output dimension.

    def forward(self, x):
        enc_out = self.embedding(x)  # Embed input data.
        enc_out= self.encoder(enc_out)
        # Encode embedded sequence.
        enc_out = self.projection(enc_out)
        # Project encoder output.

        return enc_out  # [B, L, D]
    

# batch_size = 32
# win_size = 100
# d_model = 512
# c_in = 132
# x = torch.randn(batch_size,win_size,c_in)
# model = Transformer(enc_in=c_in, c_out=c_in, d_model=d_model, n_heads=8, e_layers=3, d_ff=512,
#                  dropout=0.0, activation='gelu')
# y = model(x)
# print(y.shape)
# #torch.Size([32, 100, 132])
