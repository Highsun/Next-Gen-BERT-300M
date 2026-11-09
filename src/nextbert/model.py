"""Weight-compatible mmBERT components with explicit experimental attention."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForMaskedLM
from .attention import SummaryBranch, local_attention, rotate, sdpa


class LoRALinear(nn.Module):
    def __init__(self, base, rank):
        super().__init__()
        self.base = base
        self.a = nn.Parameter(torch.empty(rank, base.in_features))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.scale = 2.0

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.a), self.b) * self.scale


def mean_pool(h, mask):
    return (h * mask[..., None]).sum(-2) / mask.sum(-1, keepdim=True).clamp_min(1)


class Encoder(nn.Module):
    def __init__(self, config, pretrained=None):
        super().__init__()
        self.exp = config
        self.lm = pretrained if pretrained is not None else AutoModelForMaskedLM.from_pretrained(
            config.model_name, revision=config.revision, attn_implementation="sdpa",
            reference_compile=False, trust_remote_code=False)
        c = self.lm.config
        if c.model_type != "modernbert" or c.attention_dropout != 0:
            raise ValueError("Requires ModernBERT with zero attention dropout")
        self.width, self.heads = c.hidden_size, c.num_attention_heads
        self.changed = [i for i in range(c.num_hidden_layers)
                        if i % c.global_attn_every_n_layers == 0 and i not in config.kept_global]
        if config.variant != "B0" and any(i >= c.num_hidden_layers for i in config.kept_global):
            raise ValueError("kept_global contains out-of-range layer indices")
        self.branches = nn.ModuleDict()
        self.projection = nn.Linear(self.width, config.projection_dim, bias=False)
        self.classifier = nn.Linear(self.width, config.num_labels)
        self.qa_head = nn.Linear(self.width, 2)
        self.rerank_head = nn.Linear(self.width, 1)
        self.transition = 1.0
        if config.lora_rank:
            self.lm.requires_grad_(False)
            for layer in self.lm.model.layers:
                for module, name in ((layer.attn, "Wqkv"), (layer.attn, "Wo"),
                                     (layer.mlp, "Wi"), (layer.mlp, "Wo")):
                    setattr(module, name, LoRALinear(getattr(module, name), config.lora_rank))
        # Initialize common heads/adapters before variant-only parameters so paired
        # seeds produce identical common initial weights across architecture arms.
        if config.variant in {"M1", "M2", "G2", "C2"}:
            for i in self.changed:
                self.branches[str(i)] = SummaryBranch(self.width, self.heads, config.variant)
        self.last_forward_flops = 0.0

    def layer(self, hidden, layer, index, content, query):
        x = layer.attn_norm(hidden)
        n = x.shape[1]
        d = self.width // self.heads
        qkv = layer.attn.Wqkv(x).view(1, n, 3, self.heads, d)
        q, k, v = qkv.permute(0, 3, 2, 1, 4).unbind(2)
        positions = torch.arange(n, device=x.device)
        q, k = rotate(q, positions, layer.attn.rotary_emb), rotate(k, positions, layer.attn.rotary_emb)
        original_global = index % self.lm.config.global_attn_every_n_layers == 0
        changed = self.exp.variant != "B0" and index in self.changed
        if original_global and not changed:
            z = sdpa(q, k, v)
        else:
            radius = self.exp.radius if changed else self.lm.config.local_attention // 2
            qmask = query if self.exp.query_global else None
            z = local_attention(q, k, v, radius, self.exp.tile_size, qmask,
                                self.exp.backend == "reference")
            if str(index) in self.branches:
                branch = self.branches[str(index)]
                s, sp = branch.summaries(x, content, self.exp.block_size)
                packed = layer.attn.Wqkv(s).view(1, -1, 3, self.heads, d)
                _, sk, sv = packed.permute(0, 3, 2, 1, 4).unbind(2)
                sk = rotate(sk, sp, layer.attn.rotary_emb)
                summed = sdpa(q, sk, sv)
                mixed = branch.mix(x, z, summed)
                # Query rows retain direct full-token attention in E3.
                z = torch.where(query[None, None, :, None], z, mixed) if self.exp.query_global else mixed
            if changed and self.transition < 1:
                z = (1-self.transition)*sdpa(q, k, v) + self.transition*z
        z = z.transpose(1, 2).reshape(1, n, self.width)
        hidden = hidden + layer.attn.out_drop(layer.attn.Wo(z))
        return hidden + layer.mlp(layer.mlp_norm(hidden))

    def encode(self, input_ids, attention_mask, content_mask, query_mask=None):
        outputs = []
        self.last_forward_flops = 0.0
        for b in range(input_ids.shape[0]):
            n = int(attention_mask[b].sum())
            if n == 0 or not bool(attention_mask[b, :n].all()):
                raise ValueError("Requires nonempty right-padded inputs; packing is not supported")
            ids = input_ids[b:b+1, :n]
            content = content_mask[b, :n].bool()
            query = torch.zeros_like(content) if query_mask is None else query_mask[b, :n].bool()
            h = self.lm.model.embeddings(input_ids=ids)
            for i, layer in enumerate(self.lm.model.layers):
                def fn(value, layer=layer, i=i):
                    return self.layer(value, layer, i, content, query)
                # Bind masks to this document: backward may run after later documents.
                def bound(value, layer=layer, i=i, content=content, query=query):
                    return self.layer(value, layer, i, content, query)
                h = checkpoint(bound, h, use_reentrant=False) if self.exp.checkpointing and self.training else fn(h)
            h = self.lm.model.final_norm(h)
            outputs.append(F.pad(h, (0, 0, 0, input_ids.shape[1]-n)))
            self.last_forward_flops += self.estimate_forward(n, int(query.sum()), int(content.sum()))
        return torch.cat(outputs)

    def estimate_forward(self, n, nq, nc):
        c, e = self.lm.config, self.exp
        pairs, extra_projection = 0, 0
        for i in range(c.num_hidden_layers):
            glob = i % c.global_attn_every_n_layers == 0
            changed = e.variant != "B0" and i in self.changed
            if glob and not changed:
                pairs += n*n
            else:
                r = e.radius if changed else c.local_attention//2
                # Tiles compute a halo superset; count the executed superset, not just logical edges.
                if not e.query_global and e.backend != 'reference':
                    pairs += math.ceil(n/e.tile_size)*e.tile_size*(e.tile_size+2*r)
                else:
                    effective_query = nq if e.query_global else 0
                    pairs += sum(min(e.tile_size, n-start)*min(n, e.tile_size+2*r+effective_query)
                                 for start in range(0, n, e.tile_size))
                if e.query_global:
                    pairs += nq*n
                if str(i) in self.branches:
                    slots = 1 if e.variant == "M1" else 2
                    k = slots*math.ceil(nc/e.block_size)
                    pairs += n*k
                    extra_projection += 6*k*self.width*self.width
                if changed and self.transition < 1:
                    pairs += n*n
        return (c.num_hidden_layers*n*(8*self.width**2+6*self.width*c.intermediate_size)
                + 4*self.width*pairs + extra_projection)

    def forward(self, batch):
        h = self.encode(batch['input_ids'], batch['attention_mask'], batch['content_mask'], batch.get('query_mask'))
        if self.exp.task == "mlm":
            selected = batch['labels'] != -100
            if not bool(selected.any()):
                raise ValueError("MLM batch contains no targets")
            logits = self.lm.decoder(self.lm.head(h[selected]))
            self.last_forward_flops += 2*int(selected.sum())*self.width*self.lm.config.vocab_size
            return F.cross_entropy(logits.float(), batch['labels'][selected]), logits
        if self.exp.task == "classification":
            logits = self.classifier(mean_pool(h, batch['content_mask']))
            return F.cross_entropy(logits.float(), batch['labels']), logits
        if self.exp.task == "ner":
            logits = self.classifier(h)
            return F.cross_entropy(logits.flatten(0, 1).float(), batch['labels'].flatten(), ignore_index=-100), logits
        if self.exp.task == "qa":
            logits = self.qa_head(h).float()
            valid = batch['content_mask'].bool().clone()
            valid[:, 0] = True  # explicit null position, consistently used by the collator and decoder
            logits = logits.masked_fill(~valid[..., None], -1e4)
            loss = (F.cross_entropy(logits[:, :, 0], batch['start']) + F.cross_entropy(logits[:, :, 1], batch['end']))/2
            return loss, logits
        if self.exp.task == "rerank":
            logits = self.rerank_head(mean_pool(h, batch['query_mask'])).squeeze(-1)
            return F.binary_cross_entropy_with_logits(logits.float(), batch['labels'].float()), logits
        raise ValueError("Retrieval uses encode_views and contrastive_loss")

    def encode_views(self, batch):
        h = self.encode(batch['input_ids'], batch['attention_mask'], batch['content_mask'])
        full_flops = self.last_forward_flops
        mask = batch['content_mask'].bool()
        global_vectors = F.normalize(mean_pool(h, mask).float(), dim=-1)
        token_vectors = F.normalize(self.projection(h).float(), dim=-1)
        blocks = []
        for i in range(h.shape[0]):
            positions = mask[i].nonzero(as_tuple=True)[0]
            if self.exp.representation == 'independent':
                vectors = []
                length = int(batch['attention_mask'][i].sum())
                for p in positions.split(self.exp.block_size):
                    ids = torch.cat((batch['input_ids'][i, :1], batch['input_ids'][i, p],
                                     batch['input_ids'][i, length-1:length]))[None]
                    cm = torch.ones_like(ids, dtype=torch.bool)
                    cm[:, 0] = False
                    cm[:, -1] = False
                    isolated = self.encode(ids, torch.ones_like(ids), cm)
                    full_flops += self.last_forward_flops
                    vectors.append(F.normalize(self.projection(isolated[0, 1:-1].mean(0)).float(), dim=-1))
                blocks.append(torch.stack(vectors))
            else:
                blocks.append(torch.stack([F.normalize(self.projection(h[i, p].mean(0)).float(), dim=-1)
                                           for p in positions.split(self.exp.block_size)]))
        self.last_forward_flops = full_flops
        return global_vectors, token_vectors, mask, blocks

    def score_views(self, query, document):
        qg, qt, qm, _ = query
        dg, _, _, db = document
        global_score = qg @ dg.T
        if self.exp.representation == "single":
            return global_score
        columns = []
        for blocks in db:
            maxima = torch.einsum('btd,kd->btk', qt, blocks).max(-1).values
            columns.append((maxima*qm).sum(-1)/qm.sum(-1).clamp_min(1))
        block_score = torch.stack(columns, dim=1)
        return self.exp.alpha*global_score+(1-self.exp.alpha)*block_score
