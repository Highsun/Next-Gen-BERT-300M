"""Window attention without N x N masks; no monkey-patching of HF internals."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def rotate(x, positions, rotary):
    cos, sin = rotary(x, position_ids=positions[None])
    a, b = x.chunk(2, dim=-1)
    return x * cos[:, None].to(x.dtype) + torch.cat((-b, a), dim=-1) * sin[:, None].to(x.dtype)


def sdpa(q, k, v, mask=None):
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)


def local_attention(q, k, v, radius, tile_size, query_mask=None, reference=False):
    """Single unpadded document. Boolean SDPA mask: True means visible.

    Query rows read all original tokens. Other rows read local union query tokens,
    with duplicates removed. Reference mode is deliberately capped at 1024.
    """
    n = q.shape[-2]
    idx = torch.arange(n, device=q.device)
    is_query = torch.zeros(n, dtype=torch.bool, device=q.device) if query_mask is None else query_mask
    if reference:
        if n > 1024:
            raise ValueError("Dense reference attention is limited to 1024 tokens")
        visible = (idx[:, None] - idx[None, :]).abs() <= radius
        visible = visible | is_query[:, None] | is_query[None, :]
        return sdpa(q, k, v, visible[None, None])
    query_idx = idx[is_query]
    chunks = []
    for start in range(0, n, tile_size):
        end = min(n, start + tile_size)
        keys = idx[max(0, start-radius):min(n, end+radius)]
        keys = torch.unique(torch.cat((keys, query_idx)), sorted=True)
        rows = idx[start:end]
        visible = (rows[:, None] - keys[None, :]).abs() <= radius
        visible = visible | is_query[keys][None, :]
        out = sdpa(q[:, :, start:end], k[:, :, keys], v[:, :, keys], visible[None, None])
        chunks.append(out)
    result = torch.cat(chunks, dim=-2)
    if query_idx.numel():
        query_out = sdpa(q[:, :, query_idx], k, v)
        result = result.index_copy(2, query_idx, query_out)
    return result


class SummaryBranch(nn.Module):
    def __init__(self, width, heads, variant):
        super().__init__()
        self.variant = variant
        if variant in {"M2", "G2", "C2"}:
            self.pool_query = nn.Parameter(torch.empty(width))
            nn.init.normal_(self.pool_query, std=0.02)
        if variant == "G2":
            self.gate = nn.Linear(width, heads)
            nn.init.zeros_(self.gate.weight)
            nn.init.constant_(self.gate.bias, -4.0)
        if variant == "C2":
            self.constant_gate = nn.Parameter(torch.full((heads,), -4.0))

    def summaries(self, x, content_mask, block_size):
        indices = content_mask.nonzero(as_tuple=True)[0]
        if not indices.numel():
            raise ValueError("Document has no poolable content tokens")
        vectors, positions = [], []
        for block in indices.split(block_size):
            values = x[0, block]
            vectors.append(values.mean(dim=0))
            positions.append((block[0] + block[-1]).float() / 2)
            if hasattr(self, "pool_query"):
                weights = torch.softmax((values.float() @ self.pool_query.float()) / math.sqrt(x.shape[-1]), 0)
                vectors.append((weights[:, None].to(values.dtype) * values).sum(0))
                positions.append(positions[-1])
        return torch.stack(vectors)[None], torch.stack(positions)

    def mix(self, x, local, summary):
        if self.variant == "G2":
            gate = torch.sigmoid(self.gate(x)).transpose(1, 2).unsqueeze(-1)
        elif self.variant == "C2":
            gate = self.constant_gate.sigmoid()[None, :, None, None]
        else:
            gate = 0.5
        return (1-gate)*local + gate*summary
