"""Backend-agnostic postprocessing for the classifier's raw logits.

Both inference backends funnel their head logits through here so the output
structure — and the exact top-3 + softmax-over-3 reduction — is single-source:
  * the torch path (`Classifier.predict`) converts its tensors to numpy first;
  * the ONNX path (`OnnxClassInference`) feeds `session.run(...)` output straight in.

Keeping this numpy-only (no torch import) means it stays importable in the slim
serving container AND makes ONNX-vs-torch parity structural rather than two
implementations that happen to agree. Must NOT import classifier.py or
class_inference.py (would create an import cycle).
"""

import numpy as np


def top3(logits: np.ndarray) -> tuple[list[int], list[float]]:
    """Top-3 class indices (descending) + softmax probabilities over those 3.

    Mirrors `torch.topk(logits, 3).values.softmax(0)`: the probabilities are
    relative to the top-3 logits only, not calibrated over the full class set.
    """
    idx = np.argpartition(logits, -3)[-3:]           # top-3, unordered  (O(C), cheap)
    idx = idx[np.argsort(logits[idx])[::-1]]         # sort descending — matches torch.topk order
    v = logits[idx]
    e = np.exp(v - v.max())
    probs = e / e.sum()                              # softmax over the 3 values
    return idx.tolist(), probs.tolist()


def build_predictions(
    species: np.ndarray, genus: np.ndarray, family: np.ndarray
) -> list[dict]:
    """Assemble the per-image prediction dicts from the three heads' logits.

    Each argument is a (B, C) numpy array of raw logits (C differs per head).
    """
    out = []
    for i in range(species.shape[0]):
        s_idx, s_prob = top3(species[i])
        g_idx, g_prob = top3(genus[i])
        f_idx, f_prob = top3(family[i])
        out.append(
            {
                "species": s_idx,
                "species_prob": s_prob,
                "genus": g_idx,
                "genus_prob": g_prob,
                "family": f_idx,
                "family_prob": f_prob,
            }
        )
    return out
