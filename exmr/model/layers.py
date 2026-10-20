"""Transformer building blocks shared by the body network and the refiners."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Mlp(nn.Module):
    """fc1 -> activation -> dropout -> fc2 -> dropout (same layout and names as timm's Mlp)."""

    def __init__(
        self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0
    ):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        return self.drop2(self.fc2(self.drop1(self.act(self.fc1(x)))))


def zero_module(module):
    """Zero-initialize the parameters of a module (the condition embedders start as no-ops)."""
    for p in module.parameters():
        p.detach().zero_()
    return module


def condition_embedder(in_dim, latent_dim, dropout):
    """Linear -> SiLU -> Dropout -> zero-initialized Linear."""
    return nn.Sequential(
        nn.Linear(in_dim, latent_dim),
        nn.SiLU(),
        nn.Dropout(dropout),
        zero_module(nn.Linear(latent_dim, latent_dim)),
    )


def feature_embedder(in_dim, out_dim):
    """LayerNorm -> zero-initialized Linear, for pretrained image features."""
    return nn.Sequential(nn.LayerNorm(in_dim), zero_module(nn.Linear(in_dim, out_dim)))


# ----- Rotary position embedding ----- #


def _rotate_half(x):
    x = x.unflatten(-1, (-1, 2))  # (..., d, 2)
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def _rope_angles(d_model, max_seq_len):
    """(max_seq_len, d_model) rotation angles, each frequency repeated for its channel pair."""
    t = torch.arange(max_seq_len).float()
    freqs = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
    freqs = torch.einsum("i, j -> i j", t, freqs)
    return freqs.repeat_interleave(2, dim=-1)


class RoPE(nn.Module):
    """Rotary position embedding over the sequence (frame) axis."""

    def __init__(self, d_model, max_seq_len=4096):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.register_buffer("angles", _rope_angles(d_model, max_seq_len), False)
        self._cos_sin_cache = None  # built lazily per device/dtype/length

    def forward(self, x):
        """Rotate queries or keys x: (B, H, L, D)."""
        seq_len, d_model = x.shape[-2:]
        assert d_model == self.d_model

        # cos/sin are computed in fp32 from the angle table (which reaches thousands of radians
        # for long sequences, unrepresentable in fp16) and only the result is cast to x.dtype.
        cache = self._cos_sin_cache
        if (
            cache is None
            or cache[0].shape[0] < seq_len
            or cache[0].device != x.device
            or cache[0].dtype != x.dtype
        ):
            if seq_len <= self.max_seq_len:
                angles = self.angles
            else:  # sequences longer than the table (long videos)
                angles = _rope_angles(d_model, seq_len).to(x.device)
            self._cos_sin_cache = (angles.cos().to(x.dtype), angles.sin().to(x.dtype))
        cos = self._cos_sin_cache[0][:seq_len]
        sin = self._cos_sin_cache[1][:seq_len]
        return (x * cos) + (_rotate_half(x) * sin)


class RoPEAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.rope = RoPE(embed_dim // num_heads, max_seq_len=4096)
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.attn_dropout_p = dropout
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        """
        Args:
            x: (B, L, C)
            attn_mask: (L, L) bool, True where a query may NOT attend to a key
            key_padding_mask: (B, L) bool, True at keys to ignore
        """
        B, L, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.reshape(B, L, self.num_heads, -1).transpose(1, 2)  # (B, H, L, D)
        k = k.reshape(B, L, self.num_heads, -1).transpose(1, 2)
        v = v.reshape(B, L, self.num_heads, -1).transpose(1, 2)
        q, k = self.rope(q), self.rope(k)

        keep_mask = None  # SDPA convention: True = may attend
        if attn_mask is not None:
            keep_mask = (~attn_mask).reshape(1, 1, L, L)
        if key_padding_mask is not None:
            keep_key = (~key_padding_mask).reshape(B, 1, 1, L)
            keep_mask = keep_key if keep_mask is None else (keep_mask & keep_key)

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=keep_mask, dropout_p=self.attn_dropout_p if self.training else 0.0
        )  # (B, H, L, D)
        return self.proj(out.transpose(1, 2).reshape(B, L, -1))


class EncoderRoPEBlock(nn.Module):
    """Pre-norm transformer encoder block with RoPE attention and zero-initialized gates."""

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.attn = RoPEAttention(hidden_size, num_heads, dropout)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=dropout,
        )
        self.gate_msa = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.gate_mlp = nn.Parameter(torch.zeros(1, 1, hidden_size))

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        x = x + self.gate_msa * self.attn(
            self.norm1(x), attn_mask=attn_mask, key_padding_mask=key_padding_mask
        )
        return x + self.gate_mlp * self.mlp(self.norm2(x))


class RoPEEncoder(nn.Module):
    """Stack of EncoderRoPEBlocks over a (B, L, C) sequence with length padding."""

    def __init__(self, latent_dim, num_layers, num_heads, mlp_ratio, dropout, max_len):
        super().__init__()
        self.max_len = max_len
        self.blocks = nn.ModuleList(
            [
                EncoderRoPEBlock(latent_dim, num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, length, frame_valid=None):
        """Returns the encoded sequence and keep (B, L), True at frames that are valid inputs."""
        attn_mask, key_padding_mask, keep = make_encoder_masks(
            length, x.shape[1], self.max_len, frame_valid
        )
        for block in self.blocks:
            x = block(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        return x, keep


# ----- Masks ----- #


def length_to_mask(lengths, max_len):
    """(B,) lengths -> (B, max_len) bool, True at valid frames."""
    return torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < (
        lengths.unsqueeze(1)
    )


def make_encoder_masks(length, L, max_len, frame_valid=None):
    """Attention masks for a RoPE encoder run over (B, L) frames.

    Args:
        length: (B,) valid length of each sequence
        L: sequence length of the tensors fed to the encoder
        max_len: sliding-window width, only applied when L > max_len. The window spans
            [i - max_len//2, i + max_len//2), so an odd max_len is effectively rounded down
            to max_len-1; max_len=1 degenerates to self-attention only. The trained models
            depend on this exact window.
        frame_valid: (B, L) bool or None, False at frames to ignore on top of the length padding

    Returns:
        attn_mask: (L, L) bool or None, True where a query may NOT attend to a key
        key_padding_mask: (B, L) bool, True at keys to ignore, relaxed so that no query
            ends up with an all-masked row (softmax would produce NaN and spread it to valid
            frames through the next block)
        keep: (B, L) bool, True at frames that count as valid observations
    """
    B = length.size(0)
    pmask = ~length_to_mask(length, L)  # (B, L)
    if frame_valid is not None:
        assert frame_valid.shape == (B, L), f"{tuple(frame_valid.shape)} != {(B, L)}"
        pmask = pmask | ~frame_valid
    keep = ~pmask

    if L > max_len:
        # Sliding window: query i attends to [i-half, i+half), clamped to stay inside [0, L)
        half = max_len // 2
        idx = torch.arange(L, device=length.device)
        lo = (idx - half).clamp_min(0).clamp_max(L - max_len)
        hi = (idx + half).clamp_max(L).clamp_min(max_len)
        hi = torch.maximum(hi, lo + 1)  # max_len=1: keep the query itself
        attn_mask = ~((idx[None, :] >= lo[:, None]) & (idx[None, :] < hi[:, None]))
        # valid keys inside each query's window (cumsum trick, avoids a (B, L, L) tensor)
        cumkeep = keep.cumsum(1, dtype=torch.int32)
        cumkeep = torch.cat([cumkeep.new_zeros(B, 1), cumkeep], dim=1)
        window_has_key = (cumkeep[:, hi] - cumkeep[:, lo]) > 0  # (B, L)
    else:
        attn_mask = None
        window_has_key = keep.any(1, keepdim=True)  # (B, 1)

    # Frames with nothing to attend to keep themselves attendable; their output is garbage
    # either way, but this keeps NaN out of the valid frames.
    return attn_mask, pmask & window_has_key, keep
