"""
Transformer module for OED model.

This module implements the cascaded decoders and transformer architecture
for the OED model.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Transformer(nn.Module):
    """Transformer with cascaded decoders for OED.

    :param nn.Module: Base PyTorch module class
    :type nn.Module: class
    """

    def __init__(self, conf):
        """Initialize the transformer.

        :param conf: Configuration object
        :type conf: Config
        :return: None
        :rtype: None
        """
        super().__init__()

        # Model parameters
        self.d_model = getattr(conf, "hidden_dim", 256)
        self.nhead = getattr(conf, "nheads", 8)
        self.num_encoder_layers = getattr(conf, "enc_layers", 6)
        self.num_decoder_layers_hopd = conf.dec_layers_hopd
        self.num_decoder_layers_interaction = conf.dec_layers_interaction
        self.dim_feedforward = getattr(conf, "dim_feedforward", 2048)
        self.dropout = getattr(conf, "dropout", 0.1)
        self.activation = "relu"
        self.pre_norm = getattr(conf, "pre_norm", False)

        # Encoder
        encoder_layer = TransformerEncoderLayer(
            self.d_model,
            self.nhead,
            self.dim_feedforward,
            self.dropout,
            self.activation,
            self.pre_norm,
        )
        encoder_norm = nn.LayerNorm(self.d_model) if self.pre_norm else None
        self.encoder = TransformerEncoder(
            encoder_layer, self.num_encoder_layers, encoder_norm
        )

        # Decoders
        decoder_layer_hopd = TransformerDecoderLayer(
            self.d_model,
            self.nhead,
            self.dim_feedforward,
            self.dropout,
            self.activation,
            self.pre_norm,
        )
        decoder_norm_hopd = nn.LayerNorm(self.d_model) if self.pre_norm else None
        self.decoder_hopd = TransformerDecoder(
            decoder_layer_hopd, self.num_decoder_layers_hopd, decoder_norm_hopd
        )

        decoder_layer_interaction = TransformerDecoderLayer(
            self.d_model,
            self.nhead,
            self.dim_feedforward,
            self.dropout,
            self.activation,
            self.pre_norm,
        )
        decoder_norm_interaction = nn.LayerNorm(self.d_model) if self.pre_norm else None
        self.decoder_interaction = TransformerDecoder(
            decoder_layer_interaction,
            self.num_decoder_layers_interaction,
            decoder_norm_interaction,
        )

        # Initialize parameters
        self._reset_parameters()

    def _reset_parameters(self):
        """Reset transformer parameters.

        :return: None
        :rtype: None
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src,
        mask,
        query_embed,
        pos_embed,
        embed_dict=None,
        targets=None,
        cur_idx=0,
    ):
        """Forward pass through the transformer.

        :param src: Source features
        :type src: torch.Tensor
        :param mask: Attention mask
        :type mask: torch.Tensor
        :param query_embed: Query embeddings
        :type query_embed: torch.Tensor
        :param pos_embed: Position embeddings
        :type pos_embed: torch.Tensor
        :param embed_dict: Dictionary of embedding layers, defaults to None
        :type embed_dict: dict, optional
        :param targets: Ground truth targets, defaults to None
        :type targets: dict, optional
        :param cur_idx: Current frame index, defaults to 0
        :type cur_idx: int, optional
        :return: Tuple of outputs
        :rtype: tuple
        """
        # Handle both image features (4D) and object features (3D)
        if len(src.shape) == 4:
            # Image features: (B, C, H, W)
            bs, c, h, w = src.shape
            src = src.flatten(2).permute(2, 0, 1)  # (H*W, B, C)
            pos_embed = pos_embed.flatten(2).permute(2, 0, 1)  # (H*W, B, C)
        elif len(src.shape) == 3:
            # Object features: (B, N, C) where N is number of objects
            bs, n, c = src.shape
            src = src.permute(1, 0, 2)  # (N, B, C)
            pos_embed = pos_embed.permute(1, 0, 2)  # (N, B, C)
        else:
            raise ValueError(f"Expected 3D or 4D input tensor, got {len(src.shape)}D")

        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)  # (N, B, C)

        # Encoder
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)

        # Decoder for HOPD (Human-Object Pair Detection)
        tgt = torch.zeros_like(query_embed)
        hopd_out = self.decoder_hopd(
            tgt,
            memory,
            memory_key_padding_mask=mask,
            pos=pos_embed,
            query_pos=query_embed,
        )

        # Decoder for interaction
        interaction_out = self.decoder_interaction(
            hopd_out,
            memory,
            memory_key_padding_mask=mask,
            pos=pos_embed,
            query_pos=query_embed,
        )

        # Return outputs
        return hopd_out, interaction_out, None, None


class TransformerEncoder(nn.Module):
    """Transformer encoder.

    :param nn.Module: Base PyTorch module class
    :type nn.Module: class
    """

    def __init__(self, encoder_layer, num_layers, norm=None):
        """Initialize the encoder.

        :param encoder_layer: Encoder layer
        :type encoder_layer: TransformerEncoderLayer
        :param num_layers: Number of layers
        :type num_layers: int
        :param norm: Normalization layer, defaults to None
        :type norm: nn.Module, optional
        :return: None
        :rtype: None
        """
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None, pos=None):
        """Forward pass through encoder.

        :param src: Source features
        :type src: torch.Tensor
        :param mask: Attention mask, defaults to None
        :type mask: torch.Tensor, optional
        :param src_key_padding_mask: Key padding mask, defaults to None
        :type src_key_padding_mask: torch.Tensor, optional
        :param pos: Position embeddings, defaults to None
        :type pos: torch.Tensor, optional
        :return: Encoded features
        :rtype: torch.Tensor
        """
        output = src

        for layer in self.layers:
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                pos=pos,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):
    """Transformer decoder.

    :param nn.Module: Base PyTorch module class
    :type nn.Module: class
    """

    def __init__(self, decoder_layer, num_layers, norm=None):
        """Initialize the decoder.

        :param decoder_layer: Decoder layer
        :type decoder_layer: TransformerDecoderLayer
        :param num_layers: Number of layers
        :type num_layers: int
        :param norm: Normalization layer, defaults to None
        :type norm: nn.Module, optional
        :return: None
        :rtype: None
        """
        super().__init__()
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None,
    ):
        """Forward pass through decoder.

        :param tgt: Target features
        :type tgt: torch.Tensor
        :param memory: Memory features from encoder
        :type memory: torch.Tensor
        :param tgt_mask: Target mask, defaults to None
        :type tgt_mask: torch.Tensor, optional
        :param memory_mask: Memory mask, defaults to None
        :type memory_mask: torch.Tensor, optional
        :param tgt_key_padding_mask: Target key padding mask, defaults to None
        :type tgt_key_padding_mask: torch.Tensor, optional
        :param memory_key_padding_mask: Memory key padding mask, defaults to None
        :type memory_key_padding_mask: torch.Tensor, optional
        :param pos: Position embeddings, defaults to None
        :type pos: torch.Tensor, optional
        :param query_pos: Query position embeddings, defaults to None
        :type query_pos: torch.Tensor, optional
        :return: Decoded features
        :rtype: torch.Tensor
        """
        output = tgt

        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer.

    :param nn.Module: Base PyTorch module class
    :type nn.Module: class
    """

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        pre_norm=False,
    ):
        """Initialize the encoder layer.

        :param d_model: Model dimension
        :type d_model: int
        :param nhead: Number of attention heads
        :type nhead: int
        :param dim_feedforward: Feedforward dimension, defaults to 2048
        :type dim_feedforward: int, optional
        :param dropout: Dropout rate, defaults to 0.1
        :type dropout: float, optional
        :param activation: Activation function, defaults to "relu"
        :type activation: str, optional
        :param pre_norm: Whether to use pre-norm, defaults to False
        :type pre_norm: bool, optional
        :return: None
        :rtype: None
        """
        super().__init__()
        self.pre_norm = pre_norm

        # Self-attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # Feedforward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # Activation
        self.activation = _get_activation_fn(activation)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        """Forward pass through encoder layer.

        :param src: Source features
        :type src: torch.Tensor
        :param src_mask: Source mask, defaults to None
        :type src_mask: torch.Tensor, optional
        :param src_key_padding_mask: Source key padding mask, defaults to None
        :type src_key_padding_mask: torch.Tensor, optional
        :param pos: Position embeddings, defaults to None
        :type pos: torch.Tensor, optional
        :return: Output features
        :rtype: torch.Tensor
        """
        if pos is not None:
            src = src + pos

        if self.pre_norm:
            src2 = self.norm1(src)
            src2 = self.self_attn(
                src2,
                src2,
                src2,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
            )[0]
            src = src + self.dropout1(src2)

            src2 = self.norm2(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
            src = src + self.dropout2(src2)
        else:
            src2 = self.self_attn(
                src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
            )[0]
            src = src + self.dropout1(src2)
            src = self.norm1(src)

            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)

        return src


class TransformerDecoderLayer(nn.Module):
    """Transformer decoder layer.

    :param nn.Module: Base PyTorch module class
    :type nn.Module: class
    """

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        pre_norm=False,
    ):
        """Initialize the decoder layer.

        :param d_model: Model dimension
        :type d_model: int
        :param nhead: Number of attention heads
        :type nhead: int
        :param dim_feedforward: Feedforward dimension, defaults to 2048
        :type dim_feedforward: int, optional
        :param dropout: Dropout rate, defaults to 0.1
        :type dropout: float, optional
        :param activation: Activation function, defaults to "relu"
        :type activation: str, optional
        :param pre_norm: Whether to use pre-norm, defaults to False
        :type pre_norm: bool, optional
        :return: None
        :rtype: None
        """
        super().__init__()
        self.pre_norm = pre_norm

        # Self-attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # Cross-attention
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # Feedforward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        # Activation
        self.activation = _get_activation_fn(activation)

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None,
    ):
        """Forward pass through decoder layer.

        :param tgt: Target features
        :type tgt: torch.Tensor
        :param memory: Memory features
        :type memory: torch.Tensor
        :param tgt_mask: Target mask, defaults to None
        :type tgt_mask: torch.Tensor, optional
        :param memory_mask: Memory mask, defaults to None
        :type memory_mask: torch.Tensor, optional
        :param tgt_key_padding_mask: Target key padding mask, defaults to None
        :type tgt_key_padding_mask: torch.Tensor, optional
        :param memory_key_padding_mask: Memory key padding mask, defaults to None
        :type memory_key_padding_mask: torch.Tensor, optional
        :param pos: Position embeddings, defaults to None
        :type pos: torch.Tensor, optional
        :param query_pos: Query position embeddings, defaults to None
        :type query_pos: torch.Tensor, optional
        :return: Output features
        :rtype: torch.Tensor
        """
        if query_pos is not None:
            tgt = tgt + query_pos

        if pos is not None:
            memory = memory + pos

        if self.pre_norm:
            # Self-attention
            tgt2 = self.norm1(tgt)
            tgt2 = self.self_attn(
                tgt2,
                tgt2,
                tgt2,
                attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask,
            )[0]
            tgt = tgt + self.dropout1(tgt2)

            # Cross-attention
            tgt2 = self.norm2(tgt)
            tgt2 = self.multihead_attn(
                tgt2,
                memory,
                memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
            )[0]
            tgt = tgt + self.dropout2(tgt2)

            # Feedforward
            tgt2 = self.norm3(tgt)
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
            tgt = tgt + self.dropout3(tgt2)
        else:
            # Self-attention
            tgt2 = self.self_attn(
                tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
            )[0]
            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)

            # Cross-attention
            tgt2 = self.multihead_attn(
                tgt,
                memory,
                memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
            )[0]
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)

            # Feedforward
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
            tgt = tgt + self.dropout3(tgt2)
            tgt = self.norm3(tgt)

        return tgt


def _get_activation_fn(activation):
    """Get activation function.

    :param activation: Activation name
    :type activation: str
    :return: Activation function
    :rtype: function
    """
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    else:
        raise RuntimeError(f"activation should be relu/gelu, not {activation}")


def build_transformer(conf):
    """Build transformer network.

    :param conf: Configuration object
    :type conf: Config
    :return: Transformer network
    :rtype: Transformer
    """
    return Transformer(conf)
