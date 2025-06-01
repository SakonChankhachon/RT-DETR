"""
by lyuwenyu
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import get_activation

from src.core import register


__all__ = ['HybridEncoder']


class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size,
            stride,
            padding=(kernel_size - 1) // 2 if padding is None else padding,
            bias=bias
        )
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class RepVggBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu'):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)
        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        return (
            kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1),
            bias3x3 + bias1x1
        )

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            # pad a 1×1 kernel to 3×3 with zeros around
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: ConvNormLayer):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class CSPRepLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_blocks=3,
        expansion=1.0,
        bias=None,
        act="silu"
    ):
        super(CSPRepLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(*[
            RepVggBlock(hidden_channels, hidden_channels, act=act)
            for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)


# transformer
class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False
    ):
        super().__init__()
        self.normalize_before = normalize_before

        # Use batch_first=True so inputs are (batch, seq_len, dim)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        # src is (batch_size, seq_len, d_model)
        residual = src
        if self.normalize_before:
            src = self.norm1(src)

        # Self‐attention with positional embedding
        q = k = self.with_pos_embed(src, pos_embed)
        src2, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)
        src = residual + self.dropout1(src2)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src2)
        if not self.normalize_before:
            src = self.norm2(src)

        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)

        if self.norm is not None:
            output = self.norm(output)
        return output


@register
class HybridEncoder(nn.Module):
    def __init__(
        self,
        in_channels=[512, 1024, 2048],
        feat_strides=[4, 8, 16],
        hidden_dim=256,
        nhead=8,
        dim_feedforward=1024,
        dropout=0.0,
        enc_act='gelu',
        use_encoder_idx=[2],
        num_encoder_layers=1,
        pe_temperature=10000,
        expansion=1.0,
        depth_mult=1.0,
        act='silu',
        eval_spatial_size=None
    ):
        super().__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size

        # All projected feature‐maps will have 'hidden_dim' channels
        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = feat_strides

        # ─── Channel‐projection blocks (1×1 conv + BN) ─────────────────────
        self.input_proj = nn.ModuleList()
        for in_ch in in_channels:
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, hidden_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden_dim)
                )
            )

        # ─── Transformer Encoder (on selected levels) ────────────────────
        encoder_layer = TransformerEncoderLayer(
            hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=enc_act
        )
        self.encoder = nn.ModuleList([
            TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers)
            for _ in range(len(use_encoder_idx))
        ])

        # ─── Top‐down FPN ──────────────────────────────────────────────────
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1, 0, -1):
            # Reduce high‐level to hidden_dim
            self.lateral_convs.append(ConvNormLayer(hidden_dim, hidden_dim, 1, 1, act=act))
            self.fpn_blocks.append(
                CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
            )

        # ─── Bottom‐up PAN ─────────────────────────────────────────────────
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.downsample_convs.append(ConvNormLayer(hidden_dim, hidden_dim, 3, 2, act=act))
            self.pan_blocks.append(
                CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
            )

        # We no longer cache a 'pos_embed' buffer at init. Instead, we
        # will always recompute it “on‐the‐fly” using the actual (h, w)
        # in the forward pass. This ensures no size mismatch in eval.
        # So we do not need _reset_parameters() to register any buffer here.

        self._reset_parameters()

    def _reset_parameters(self):
        # We no longer register any 'pos_embed' buffers. Nothing to do.
        return

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
        """
        Build a (1, w*h, embed_dim) 2D‐sin/cos positional embedding.
        'w' is the width (number of columns), 'h' is the height (number of rows).
        """
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, 'Embed dim must be divisible by 4'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        # out_w: shape (w*h, pos_dim)
        out_w = grid_w.flatten()[..., None] @ omega[None]
        # out_h: shape (w*h, pos_dim)
        out_h = grid_h.flatten()[..., None] @ omega[None]

        # Concatenate sin/cos embeddings: [1, (w*h), embed_dim]
        return torch.cat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    def forward(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Args:
            feats: list of feature maps from backbone, one per in_channels.
                   Each `feat` has shape [B, C_i, H_i, W_i], where (C_i, H_i, W_i)
                   matches `self.in_channels[i]` and `H_i = input_size / feat_strides[i]`.
        Returns:
            outs: list of 3 merged/pan‐joined feature maps, each [B, hidden_dim, H_out_i, W_out_i].
        """
        assert len(feats) == len(self.in_channels)

        # 1) Channel‐project each input feature to `hidden_dim`
        proj_feats = [self.input_proj[i](feats[i]) for i in range(len(feats))]

        # 2) Transformer‐encode *only* the levels in self.use_encoder_idx
        if self.num_encoder_layers > 0:
            for enc_i, enc_ind in enumerate(self.use_encoder_idx):
                # Get the current feature map at that index
                b, c, h, w = proj_feats[enc_ind].shape

                # Flatten to [B, h*w, hidden_dim]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)

                # Always recompute pos_embed using the actual (w, h) from this feature
                pos_embed = self.build_2d_sincos_position_embedding(w, h, self.hidden_dim, self.pe_temperature)
                pos_embed = pos_embed.to(src_flatten.device)

                # Run it through a small TransformerEncoder
                memory = self.encoder[enc_i](src_flatten, pos_embed=pos_embed)

                # Reshape back to [B, hidden_dim, h, w]
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(b, self.hidden_dim, h, w).contiguous()

        # 3) Top‐down FPN: start from highest‐level projection, go downward
        inner_outs = [proj_feats[-1]]  # start with the “deepest” feature (lowest spatial res)
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]                # (the one we just computed)
            feat_low = proj_feats[idx - 1]           # the next shallower feature
            # 1×1 lateral to reduce channels on feat_high
            lateral = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_high)
            inner_outs[0] = lateral
            upsample_feat = F.interpolate(lateral, scale_factor=2.0, mode='nearest')
            fused = torch.cat([upsample_feat, feat_low], dim=1)
            inner = self.fpn_blocks[len(self.in_channels) - 1 - idx](fused)
            inner_outs.insert(0, inner)

        # 4) Bottom‐up PAN: start from the top of inner_outs, go back upward
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)  # reduce spatial by 2
            fused = torch.cat([downsample_feat, feat_high], dim=1)
            out = self.pan_blocks[idx](fused)
            outs.append(out)

        # outs now has exactly len(in_channels) items, each B×hidden_dim×(H_out_i)×(W_out_i)
        return outs
