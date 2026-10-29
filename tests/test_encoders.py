"""The text encoders batch texts of similar length together, so each batch pads only to its own longest text, and
return the embeddings in the caller's order: the same as encoding each text alone."""
from types import SimpleNamespace

import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from jevstiller.encoders._hf import length_batches
from jevstiller.encoders._onnx import OnnxEncoder

WORDS = [f"w{i}" for i in range(40)]


def mixed_texts(n=50, seed=0):
    rng = np.random.default_rng(seed)
    return [" ".join(rng.choice(WORDS, size=int(rng.integers(1, 60)))) for _ in range(n)]


def test_length_batches_cover_every_text_once_shortest_first():
    texts = ["a" * k for k in (5, 1, 9, 3, 7, 2, 8)]
    batches = length_batches(texts, 3)
    assert sorted(i for b in batches for i in b) == list(range(len(texts)))
    assert [[len(texts[i]) for i in b] for b in batches] == [[1, 2, 3], [5, 7, 8], [9]]
    assert length_batches([], 3) == []


class FakeSession:
    """ONNX Runtime's stand-in: hidden states that depend on each text's own tokens and never on the padding, as a
    real model's do under its attention mask. Row 0 (what CLS pooling takes) sums the text's token vectors."""

    def __init__(self, vocab_size: int, dim: int = 8):
        self.table = np.random.default_rng(1).normal(size=(vocab_size, dim)).astype(np.float32)
        self.shapes = []

    def run(self, names, feed):
        ids, mask = feed["input_ids"], feed["attention_mask"]
        self.shapes.append(ids.shape)
        h = self.table[ids] * mask[..., None]
        h[:, 0] = h.sum(1)
        return [h]


def fake_onnx(pooling: str) -> OnnxEncoder:
    vocab = {"[PAD]": 0, "[UNK]": 1} | {w: i + 2 for i, w in enumerate(WORDS)}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok.enable_truncation(256)
    tok.enable_padding()
    enc = OnnxEncoder.__new__(OnnxEncoder)          # no download: the tokenizer is real, the model a stand-in
    enc.tok, enc.sess = tok, FakeSession(len(vocab))
    enc.pooling, enc.batch_size, enc.max_length = pooling, 8, 256
    enc.input_names, enc.output_name = ["input_ids", "attention_mask"], "last_hidden_state"
    return enc


@pytest.mark.parametrize("pooling", ["cls", "mean"])
def test_onnx_encoder_returns_the_callers_order(pooling):
    enc, texts = fake_onnx(pooling), mixed_texts()
    together = enc.encode(texts)
    assert together.shape == (len(texts), 8) and together.dtype == np.float32
    np.testing.assert_allclose(together, np.concatenate([enc.encode([t]) for t in texts]), rtol=0, atol=1e-5)


def test_onnx_encoder_pads_each_batch_to_its_own_longest_text():
    enc = fake_onnx("cls")
    enc.encode([" ".join(["w1"] * (1 if i % 2 else 50)) for i in range(32)])   # short and long, interleaved
    assert enc.sess.shapes == [(8, 1), (8, 1), (8, 50), (8, 50)]                # in arrival order: 4 x (8, 50)


def test_onnx_encoder_takes_an_empty_call():
    enc = fake_onnx("cls")
    enc.dim = 8
    assert enc.encode([]).shape == (0, 8)


def test_torch_encoder_returns_the_callers_order():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from jevstiller.encoders._hf import TorchEncoder
    onnx = fake_onnx("cls")
    table = torch.from_numpy(onnx.sess.table)

    def model(input_ids, attention_mask, **_):
        h = table[input_ids] * attention_mask[..., None]
        h[:, 0] = h.sum(1)
        return SimpleNamespace(last_hidden_state=h)

    enc = TorchEncoder.__new__(TorchEncoder)
    enc.torch, enc.model, enc.device, enc.dim = torch, model, "cpu", 8
    enc.tok = transformers.PreTrainedTokenizerFast(tokenizer_object=onnx.tok, pad_token="[PAD]")
    enc.pooling, enc.batch_size, enc.max_length = "cls", 8, 256
    texts = mixed_texts()
    np.testing.assert_allclose(enc.encode(texts), np.concatenate([enc.encode([t]) for t in texts]), rtol=0, atol=1e-5)
