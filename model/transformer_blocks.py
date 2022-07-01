import math
import torch
from torch import nn
from typing import Optional, Dict

from fairseq.models import FairseqIncrementalDecoder
from fairseq.modules import TransformerDecoderLayer, TransformerEncoderLayer

# https://github.com/pytorch/examples/blob/master/word_language_model/model.py
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Namespace:
    def __init__(self, argvs):
        for k, v in argvs.items():
            setattr(self, k, v)


class TransformerEncoder(nn.Module):
    def __init__(self, dim, ff_dim, num_head, num_layer):
        super().__init__()

        self.layer = nn.ModuleList([
            TransformerEncoderLayer(Namespace({
                'encoder_embed_dim': dim,
                'encoder_attention_heads': num_head,
                'attention_dropout': 0.1,
                'dropout': 0.1,
                'encoder_normalize_before': True,
                'encoder_ffn_embed_dim': ff_dim,
            })) for i in range(num_layer)
        ])
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape `(tgt_len, src_len)`,
                where `tgt_len` is the length of output and `src_len` is the
                length of input, though here both are equal to `seq_len`.
                `attn_mask[tgt_i, src_j] = 1` means that when calculating the
                embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
                useful for strided self-attention.
        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x, encoder_padding_mask=None):
        # print('my TransformerDecode forward()')
        for layer in self.layer:
            x = layer(x, encoder_padding_mask)
        x = self.layer_norm(x)
        return x  # T x B x C


class TransformerDecoder(nn.Module):
    def __init__(self, dim, ff_dim, num_head, num_layer):
        super().__init__()

        self.layer = nn.ModuleList([
            TransformerDecoderLayer(Namespace({
                'decoder_embed_dim': dim,
                'decoder_attention_heads': num_head,
                'attention_dropout': 0.1,
                'dropout': 0.1,
                'decoder_normalize_before': True,
                'decoder_ffn_embed_dim': ff_dim,
            })) for i in range(num_layer)
        ])
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x, mem, x_mask=None, x_padding_mask=None, mem_padding_mask=None):
        for layer in self.layer:
            x = layer(x, mem,
                      self_attn_mask=x_mask, self_attn_padding_mask=x_padding_mask,
                      encoder_padding_mask=mem_padding_mask)[0]
        x = self.layer_norm(x)
        return x  # T x B x C

    @torch.jit.export
    def forward_one(self,
                    x: torch.Tensor,
                    mem: torch.Tensor,
                    incremental_state: Optional[Dict[str, Dict[str, Optional[torch.Tensor]]]],
                    mem_padding_mask: torch.BoolTensor = None,
                    ) -> torch.Tensor:
        x = x[-1:]
        for layer in self.layer:
            x = layer(x, mem, incremental_state=incremental_state, encoder_padding_mask=mem_padding_mask)[0]
        x = self.layer_norm(x)
        return x
