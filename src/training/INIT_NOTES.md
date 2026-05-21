# CustomResnet Weight Initialization

## Strategy

Load pretrained ResNet34 ImageNet weights into the backbone. The first conv layer requires
manual handling because we extend from 3→5 input channels (RGB + Scharr gradient + LCN).

## Steps

### 1. Build the 5-channel first conv weight manually

```python
import torchvision.models as torchvision_models

pretrained = torchvision_models.resnet34(weights=torchvision_models.ResNet34_Weights.IMAGENET1K_V1)

new_w = torch.zeros(64, 5, 7, 7)
new_w[:, :3, :, :] = pretrained.conv1.weight.data                             # RGB — copy verbatim
new_w[:, 3:, :, :] = pretrained.conv1.weight.data.mean(dim=1, keepdim=True)  # Scharr + LCN — mean of RGB
model.conv1.weight.data.copy_(new_w)
```

Channels 3–4 (Scharr gradient and LCN) are edge-enhancement features derived from RGB signal,
so the mean of RGB weights is a semantically informed prior — better than random Kaiming init
(which would start contributing nothing useful) or zeros (near-zero gradients on first pass).

### 2. Pop manually-handled keys before load_state_dict

`load_state_dict` matches by exact parameter name. Pop anything already handled, plus the
pretrained single-head `fc` which has no corresponding key in our three-head model:

```python
state = pretrained.state_dict()
state.pop("conv1.weight")  # handled above — do not overwrite
state.pop("fc.weight")
state.pop("fc.bias")
model.load_state_dict(state, strict=False)
```

`strict=False` silently skips any remaining name mismatches. The backbone layers
(layer1–4, bn1, etc.) all match by name to ResNet34 and load correctly.

### 3. Warmup phase

Freeze all layers except `conv1` for the first N epochs to let channels 3–4 stabilize
before the backbone shifts. Unfreeze via `optimizer.add_param_group` (not optimizer
rebuild) to preserve momentum state accumulated on `conv1` during warmup.

## Why pretrained over scratch?

~450K images across 1,500 classes is not enough to learn low-level visual primitives from
scratch reliably. Early conv layers learn universal features (edges, textures, gradients)
that transfer across domains — underwater images are blue-shifted but edges still exist.
Training from scratch only wins with tens of millions of domain-specific images.
