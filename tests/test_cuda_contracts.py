"""Required AutoDL gates; deliberately NOT executed during local authoring."""
import copy
import pytest
import torch
from transformers import ModernBertConfig, ModernBertForMaskedLM
from nextbert.config import Config
from nextbert.model import Encoder
from nextbert.attention import local_attention

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA validation only')


def tiny(variant='B0', backend='tiled'):
    hc = ModernBertConfig(vocab_size=128, hidden_size=32, num_hidden_layers=7, num_attention_heads=4,
                         intermediate_size=48, max_position_embeddings=512, local_attention=8,
                         global_attn_every_n_layers=3, reference_compile=False,
                         attention_dropout=0., embedding_dropout=0., mlp_dropout=0.)
    hc._attn_implementation = 'sdpa'
    upstream = ModernBertForMaskedLM(hc).cuda().eval()
    cfg = Config(variant=variant, kept_global=[0, 6], radius=4, block_size=8,
                 tile_size=8, lora_rank=0, checkpointing=False, backend=backend)
    return Encoder(cfg, pretrained=upstream).cuda().eval()


def test_b0_matches_hf_valid_tokens():
    model = tiny()
    ids = torch.randint(5, 128, (1, 31), device='cuda')
    mask = torch.ones_like(ids)
    content = mask.bool()
    with torch.no_grad():
        expected = model.lm.model(input_ids=ids, attention_mask=mask).last_hidden_state
        actual = model.encode(ids, mask, content)
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize('query_global', [False, True])
def test_tiled_equals_dense_outputs_and_gradients(query_global):
    tensors = [torch.randn(1, 4, 37, 8, device='cuda', requires_grad=True) for _ in range(3)]
    copies = [x.detach().clone().requires_grad_() for x in tensors]
    mask = torch.zeros(37, device='cuda', dtype=torch.bool)
    if query_global:
        mask[:5] = True
    actual = local_attention(*tensors, 4, 8, mask)
    expected = local_attention(*copies, 4, 8, mask, reference=True)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    actual.square().sum().backward()
    expected.square().sum().backward()
    for a, b in zip(tensors, copies):
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize('variant', ['L3', 'M1', 'M2', 'G2', 'C2'])
def test_variants_backward_and_transition_identity(variant):
    model = tiny(variant)
    ids = torch.randint(5, 128, (1, 31), device='cuda')
    mask = torch.ones_like(ids)
    model.transition = 0.
    expected = model.lm.model(input_ids=ids, attention_mask=mask).last_hidden_state
    actual = model.encode(ids, mask, mask.bool())
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
    model.transition = 1.
    model.train()
    model.exp.checkpointing = True
    model.encode(ids, mask, mask.bool()).square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_documents_are_isolated_and_padding_invariant():
    model = tiny('G2')
    a, b = torch.randint(5, 128, (1, 19), device='cuda'), torch.randint(5, 128, (1, 27), device='cuda')
    batch = torch.cat((torch.nn.functional.pad(a, (0, 8)), b))
    valid = torch.arange(27, device='cuda')[None] < torch.tensor([19, 27], device='cuda')[:, None]
    with torch.no_grad():
        alone = model.encode(a, torch.ones_like(a), torch.ones_like(a, dtype=torch.bool))
        together = model.encode(batch, valid, valid)
    torch.testing.assert_close(alone[0], together[0, :19], rtol=1e-4, atol=1e-5)


def test_compact_checkpoint_restores_trainables_without_replacing_base(tmp_path):
    from nextbert.train import save_checkpoint, restore_state
    model = tiny('G2')
    model.lm.requires_grad_(False)
    model.exp.lora_rank = 16  # Exercise storage format independently of LoRA injection.
    model.exp.revision = 'a'*40
    expected = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    frozen = model.lm.model.embeddings.tok_embeddings.weight.detach().clone()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
    class TokenizerStub:
        def save_pretrained(self, path):
            path.mkdir(parents=True, exist_ok=True)
    save_checkpoint(tmp_path/'compact', model, TokenizerStub(), model.exp, optimizer, {'step': 1})
    payload = torch.load(tmp_path/'compact'/'checkpoint.pt', weights_only=False)
    assert payload['checkpoint_format'] == 'trainable-v1'
    assert not any(n.startswith('lm.') for n in payload['model'])
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in expected:
                p.add_(1)
    restore_state(model, payload)
    for n, p in model.named_parameters():
        if n in expected:
            torch.testing.assert_close(p, expected[n])
    torch.testing.assert_close(model.lm.model.embeddings.tok_embeddings.weight, frozen)
