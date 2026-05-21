"""
Tests for BasicBlock and CustomResnet architecture.

All tests are pure PyTorch — no infrastructure or mocking required.
Validates forward pass shapes, skip connection correctness, and head outputs.
"""

import pytest
import torch
import torch.nn as nn

from whatsthatfish.src.models.c_custom_resnet import BasicBlock, CustomResnet


# ════════════════════════════════════════════════════════════════════════════════
# BasicBlock
# ════════════════════════════════════════════════════════════════════════════════

class TestBasicBlock:

    def test_forward_same_planes_preserves_shape(self):
        block = BasicBlock(planes=64)
        x = torch.randn(2, 64, 32, 32)
        out = block(x)
        assert out.shape == (2, 64, 32, 32)

    def test_forward_with_stride_halves_spatial_dims(self):
        downsample = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=2),
            nn.BatchNorm2d(128),
        )
        block = BasicBlock(planes=128, in_planes=64, stride=2, downsample=downsample)
        x = torch.randn(2, 64, 32, 32)
        out = block(x)
        assert out.shape == (2, 128, 16, 16)

    def test_forward_with_plane_expansion_no_stride(self):
        """Downsample needed when planes change even at stride=1 (e.g. layer1 in_planes=64→64, but check edge)."""
        downsample = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64),
        )
        block = BasicBlock(planes=64, in_planes=32, stride=1, downsample=downsample)
        x = torch.randn(2, 32, 16, 16)
        out = block(x)
        assert out.shape == (2, 64, 16, 16)

    def test_output_is_nonnegative(self):
        """Final ReLU means no negative activations in the output."""
        block = BasicBlock(planes=32)
        x = torch.randn(2, 32, 16, 16)
        out = block(x)
        assert out.min().item() >= 0.0

    def test_no_downsample_uses_identity_skip(self):
        """Without downsample, the skip connection is the raw input — out shape must match x."""
        block = BasicBlock(planes=64, downsample=None)
        x = torch.randn(1, 64, 8, 8)
        out = block(x)
        assert out.shape == x.shape

    def test_in_planes_defaults_to_planes(self):
        """If in_planes is omitted it should default to planes (same-dim block)."""
        block = BasicBlock(planes=128)
        x = torch.randn(1, 128, 4, 4)
        out = block(x)
        assert out.shape == (1, 128, 4, 4)


# ════════════════════════════════════════════════════════════════════════════════
# CustomResnet
# ════════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def small_model():
    """Minimal ResNet18-equivalent with 5-channel input and three heads."""
    return CustomResnet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        num_class=[100, 30, 10],
        in_dim=5,
    )


class TestCustomResnet:

    def test_forward_returns_three_tensors(self, small_model):
        x = torch.randn(2, 5, 224, 224)
        out = small_model(x)
        assert len(out) == 3

    def test_species_head_shape(self, small_model):
        x = torch.randn(2, 5, 224, 224)
        out_species, _, _ = small_model(x)
        assert out_species.shape == (2, 100)

    def test_genus_head_shape(self, small_model):
        x = torch.randn(2, 5, 224, 224)
        _, out_genus, _ = small_model(x)
        assert out_genus.shape == (2, 30)

    def test_subfamily_head_shape(self, small_model):
        x = torch.randn(2, 5, 224, 224)
        _, _, out_subfamily = small_model(x)
        assert out_subfamily.shape == (2, 10)

    def test_five_channel_input_accepted(self, small_model):
        x = torch.randn(1, 5, 224, 224)
        out = small_model(x)
        assert len(out) == 3

    def test_batch_size_one(self, small_model):
        x = torch.randn(1, 5, 224, 224)
        out_species, _, _ = small_model(x)
        assert out_species.shape == (1, 100)

    def test_three_channel_model_accepted(self):
        model = CustomResnet(
            block=BasicBlock,
            layers=[2, 2, 2, 2],
            num_class=[50, 10, 5],
            in_dim=3,
        )
        x = torch.randn(1, 3, 224, 224)
        out = model(x)
        assert len(out) == 3

    def test_outputs_are_logits_not_probs(self, small_model):
        """Heads should output raw logits — values outside [0,1] are expected."""
        x = torch.randn(4, 5, 224, 224)
        out_species, _, _ = small_model(x)
        has_outside_unit = ((out_species > 1.0) | (out_species < 0.0)).any()
        assert has_outside_unit.item()

    def test_layer4_output_512_channels(self, small_model):
        """layer4 should produce 512-channel feature maps before pooling."""
        x = torch.randn(1, 5, 224, 224)
        with torch.no_grad():
            out = small_model.conv1(x)
            out = small_model.bn(out)
            out = small_model.relu(out)
            out = small_model.max_pool(out)
            out = small_model.layer1(out)
            out = small_model.layer2(out)
            out = small_model.layer3(out)
            out = small_model.layer4(out)
        assert out.shape[1] == 512
