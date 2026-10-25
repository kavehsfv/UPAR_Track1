"""UPAR 2027 Track 1 submission: CLIP ViT-B/16 fine-tuned for 40-attribute PAR.

Ingestion contract:
    load_model()                    once
    predict_image(sample)           once per image (or predict_batch)

This file is self-contained: the ViT definition and image preprocessing are
inlined (no open_clip/timm dependency - evaluation container is stock torch).
Weights: assets/par_model.pt (visual tower + 40-way head + per-attr bias).
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(HERE, "assets", "par_model.pt")

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

_MODEL: PARModel | None = None
_BIAS: np.ndarray | None = None
_DEV: torch.device | None = None


# ------------------------------------------------------------------ model code

class MultiheadAttention(nn.Module):
    def __init__(self, d: int, n: int):
        super().__init__()
        self.in_proj = nn.Linear(d, 3 * d)
        self.out_proj = nn.Linear(d, d)
        self.n = n

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.in_proj(x).reshape(B, T, 3, self.n, D // self.n).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.out_proj(out)


class MLP(nn.Module):
    def __init__(self, d: int, h: int):
        super().__init__()
        self.c_fc = nn.Linear(d, h)
        self.c_proj = nn.Linear(h, d)

    def forward(self, x):
        h = self.c_fc(x)
        return self.c_proj(h * torch.sigmoid(1.702 * h))  # QuickGELU (OpenAI CLIP)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d: int, n: int, h: int):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d)
        self.attn = MultiheadAttention(d, n)
        self.ln_2 = nn.LayerNorm(d)
        self.mlp = MLP(d, h)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, embed_dim: int = 768, layers: int = 12, heads: int = 12,
                 patch_size: int = 16, image_size: int = 224, width: int = 768,
                 output_dim: int = 512):
        super().__init__()
        self.conv1 = nn.Conv2d(3, width, kernel_size=patch_size, stride=patch_size, bias=False)
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(
            (image_size // patch_size) ** 2 + 1, width))
        self.ln_pre = nn.LayerNorm(width)
        self.blocks = nn.ModuleList([ResidualAttentionBlock(width, heads, width * 4)
                                     for _ in range(layers)])
        self.ln_post = nn.LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        cls = self.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([cls, x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        for blk in self.blocks:  # batch-first blocks; no (T,B,D) permute
            x = blk(x)
        x = self.ln_post(x)
        pooled = x[:, 0]  # class token
        return pooled @ self.proj


class PARModel(nn.Module):
    def __init__(self, num_attributes: int = 40):
        super().__init__()
        self.visual = VisionTransformer()
        self.head = nn.Linear(512, num_attributes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.visual(x))


# ------------------------------------------------------------------ preprocessing

def letterbox(img: Image.Image, size: int = 224) -> Image.Image:
    w, h = img.size
    s = size / max(w, h)
    nw, nh = max(round(w * s), 1), max(round(h * s), 1)
    img = img.convert("RGB").resize((nw, nh), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size), (128, 128, 128))
    canvas.paste(img, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def preprocess(img: Image.Image) -> torch.Tensor:
    img = letterbox(img)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - np.asarray(CLIP_MEAN, dtype=np.float32)) / np.asarray(CLIP_STD, dtype=np.float32)
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


# ------------------------------------------------------------------ API

def load_model() -> None:
    global _MODEL, _BIAS, _DEV
    if _MODEL is not None:
        return
    _DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    model = PARModel()
    model.visual.load_state_dict(ckpt["visual"], strict=True)
    model.head.load_state_dict(ckpt["head"], strict=True)
    model = model.to(_DEV).eval()
    _BIAS = ckpt["t1_bias"].numpy().astype(np.float32)
    _MODEL = model


def _probs(samples: list[dict[str, Any]]) -> np.ndarray:
    if _MODEL is None:
        load_model()
    assert _MODEL is not None and _BIAS is not None and _DEV is not None
    xs = [preprocess(Image.open(s["image_path"]).convert("RGB")) for s in samples]
    out = np.empty((len(xs), 40), dtype=np.float32)
    with torch.no_grad():
        for i0 in range(0, len(xs), 256):
            x = torch.stack(xs[i0:i0 + 256]).to(_DEV)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=_DEV.type == "cuda"):
                logits = _MODEL(x)
            logits = logits.float().cpu().numpy() + _BIAS
            out[i0:i0 + len(x)] = 1.0 / (1.0 + np.exp(-logits))
    return out


def predict_image(sample: dict[str, Any]) -> list[float]:
    # order is fixed by the challenge, but honour sample if present
    p = _probs([sample])[0]
    return [float(v) for v in p]


def predict_batch(samples: list[dict[str, Any]]) -> list[list[float]]:
    P = _probs(samples)
    return [[float(v) for v in row] for row in P]
