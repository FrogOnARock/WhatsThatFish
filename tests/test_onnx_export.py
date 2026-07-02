"""Fast ONNX export unit tests (item 2): the artifact is created, accepts the
right input, and produces correctly-named/-shaped outputs.

Uses a TINY synthetic checkpoint so it runs in ordinary CI — no 271 MB weights,
no DB. It still exercises the *real* `Classifier.export()` path (load → build →
torch.onnx.export → optional INT8 quantize), just on a small model.

onnxruntime is imported lazily inside helpers so `--collect-only` works even
where the runtime can't load (e.g. a GPU wheel missing CUDA libs).
"""

import numpy as np
import pytest
import torch

from whatsthatfish.models.classifier import Classifier
from whatsthatfish.models.architecture.custom_resnet import CustomResnet, BasicBlock

TINY_LABELS = [7, 5, 3]  # [species, genus, family]
TINY_ARCH = {"layers": [1, 1, 1, 1], "in_dim": 5, "head_mode": "progressive"}


def _session(path):
    import onnxruntime as ort

    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


@pytest.fixture(scope="module")
def tiny_ckpt(tmp_path_factory):
    """A minimal self-describing checkpoint in the exact shape export() expects."""
    model = CustomResnet(
        block=BasicBlock,
        layers=TINY_ARCH["layers"],
        num_class=TINY_LABELS,
        in_dim=TINY_ARCH["in_dim"],
        pretrained=False,
        head_mode=TINY_ARCH["head_mode"],
    )
    p = tmp_path_factory.mktemp("ckpt") / "tiny.pt"
    torch.save(
        {"model": model.state_dict(), "num_labels": TINY_LABELS, "arch": TINY_ARCH}, p
    )
    return p


@pytest.fixture(scope="module")
def classifier_onnx(tiny_ckpt, tmp_path_factory):
    """Export FP32 + INT8 once for the module (exercises the real export path)."""
    out = tmp_path_factory.mktemp("onnx") / "classifier.onnx"
    return Classifier().export(weights=tiny_ckpt, out=out, int8=True)


class TestClassifierOnnxExport:
    def test_files_created(self, classifier_onnx):
        assert classifier_onnx["fp32"].exists()
        assert classifier_onnx["int8"] is not None and classifier_onnx["int8"].exists()

    def test_input_contract(self, classifier_onnx):
        sess = _session(classifier_onnx["fp32"])
        inp = sess.get_inputs()[0]
        assert inp.name == "input"
        # channels/H/W fixed at 5/320/320; batch is the dynamic axis
        assert list(inp.shape)[1:] == [5, 320, 320]

    def test_output_names_and_dims(self, classifier_onnx):
        sess = _session(classifier_onnx["fp32"])
        assert [o.name for o in sess.get_outputs()] == ["species", "genus", "family"]
        x = np.random.rand(2, 5, 320, 320).astype(np.float32)
        species, genus, family = sess.run(None, {"input": x})
        assert species.shape == (2, TINY_LABELS[0])
        assert genus.shape == (2, TINY_LABELS[1])
        assert family.shape == (2, TINY_LABELS[2])

    def test_batch_axis_is_dynamic(self, classifier_onnx):
        sess = _session(classifier_onnx["fp32"])
        for b in (1, 4):
            x = np.random.rand(b, 5, 320, 320).astype(np.float32)
            out = sess.run(None, {"input": x})
            assert all(o.shape[0] == b for o in out)

    def test_int8_variant_runs_with_same_dims(self, classifier_onnx):
        sess = _session(classifier_onnx["int8"])
        x = np.random.rand(3, 5, 320, 320).astype(np.float32)
        species, genus, family = sess.run(None, {"input": x})
        assert (species.shape, genus.shape, family.shape) == (
            (3, TINY_LABELS[0]),
            (3, TINY_LABELS[1]),
            (3, TINY_LABELS[2]),
        )

    def test_export_without_int8_skips_quant(self, tiny_ckpt, tmp_path):
        paths = Classifier().export(
            weights=tiny_ckpt, out=tmp_path / "c.onnx", int8=False
        )
        assert paths["fp32"].exists()
        assert paths["int8"] is None
