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


def batched_local_attention(q, k, v, radius, tile_size):
    """Batch overlapping K/V windows in one SDPA call; preserve sliding edges."""
    _, heads, n, dim = q.shape
    tiles = math.ceil(n/tile_size)
    padding = tiles*tile_size-n
    width = tile_size+2*radius
    queries = F.pad(q, (0, 0, 0, padding)).reshape(1, heads, tiles, tile_size, dim)
    queries = queries.permute(0, 2, 1, 3, 4).reshape(tiles, heads, tile_size, dim)

    def windows(tensor):
        padded = F.pad(tensor, (0, 0, radius, radius+padding))
        return padded.unfold(2, width, tile_size).permute(0, 2, 1, 4, 3).reshape(tiles, heads, width, dim)

    rows = torch.arange(tile_size, device=q.device)
    columns = torch.arange(width, device=q.device)-radius
    starts = torch.arange(tiles, device=q.device)*tile_size
    keys = starts[:, None]+columns[None, :]
    visible = ((rows[:, None]-columns[None, :]).abs() <= radius)[None]
    visible = visible & (keys[:, None, :] >= 0) & (keys[:, None, :] < n)
    if padding:
        # Padded query rows are discarded; give them valid keys to avoid all-masked-row kernel quirks.
        real_queries = starts[:, None]+rows[None, :] < n
        visible = visible | (~real_queries[..., None] & (keys[:, None, :] >= 0) & (keys[:, None, :] < n))
    out = sdpa(queries, windows(k), windows(v), visible[:, None])
    out = out.reshape(1, tiles, heads, tile_size, dim).permute(0, 2, 1, 3, 4)
    return out.reshape(1, heads, tiles*tile_size, dim)[:, :, :n]


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
    if query_mask is None or not bool(query_mask.any()):
        return batched_local_attention(q, k, v, radius, tile_size)
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
        count = indices.numel()
        blocks = math.ceil(count/block_size)
        padding = blocks*block_size-count
        values = F.pad(x[0, indices], (0, 0, 0, padding)).reshape(blocks, block_size, x.shape[-1])
        valid = torch.arange(blocks*block_size, device=x.device).reshape(blocks, block_size) < count
        means = (values.float().sum(1)/valid.sum(1, keepdim=True)).to(x.dtype)
        first = torch.arange(blocks, device=x.device)*block_size
        last = (first+block_size-1).clamp_max(count-1)
        positions = (indices[first]+indices[last]).float()/2
        if not hasattr(self, 'pool_query'):
            return means[None], positions
        scores = (values.float() @ self.pool_query.float())/math.sqrt(x.shape[-1])
        weights = torch.softmax(scores.masked_fill(~valid, float('-inf')), dim=1)
        weighted = (weights[..., None].to(x.dtype)*values).sum(1)
        return torch.stack((means, weighted), dim=1).reshape(1, -1, x.shape[-1]), positions.repeat_interleave(2)

    def mix(self, x, local, summary):
        if self.variant == "G2":
            gate = torch.sigmoid(self.gate(x)).transpose(1, 2).unsqueeze(-1)
        elif self.variant == "C2":
            gate = self.constant_gate.sigmoid()[None, :, None, None]
        else:
            gate = 0.5
        return (1-gate)*local + gate*summary
