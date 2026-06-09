import torch
import torch.nn as nn
from typing import Type
import warnings


class BasicBlock(nn.Module):
    # Block expansion not required given only implementing resnet-34 with basic block.

    def __init__(
        self,
        planes: int,
        in_planes: int | None = None,
        downsample: nn.Sequential | None = None,
        stride: int = 1,
    ):
        super().__init__()
        self.stride = stride

        if in_planes is None:
            in_planes = planes

        self.downsample = downsample
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=self.stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        """
        Pattern is relatively straightforward, convolution -> batch norm -> relu, but on
        second convolution, add back in dimension aligned identity before relu.
        Convolution -> batch norm -> identity addition -> relu -> out
        """
        identity = x  # set identity for all blocks that are not the first block.. dim_identity = dim_tensor

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            identity = self.downsample(
                x
            )  # we down sample our identity if we have to, first block in layer
            # so that it matches the dimensions of the tensor after the convolutions are passed over it with stride = 2
            # we reduce spatial dimensions of the tensor in the new layer, increase num of feature maps, therefore
            # we must apply conv2d + batch norm to identity to perform same operations

        out = out + identity  # Add skip
        out = self.relu(out)

        return out


class CustomResnet(nn.Module):
    def __init__(
        self,
        block: Type[BasicBlock],
        layers: list[int],
        num_class: list[int],
        in_dim: int = 5,
        in_planes: int = 64,
        batch_norm=None,
        pretrained: bool = False,
    ):
        super().__init__()
        self.batch_norm = batch_norm if batch_norm is not None else nn.BatchNorm2d
        self.in_planes = in_planes
        # Variant flag: True -> backbone comes from load_pretrained() and
        # _init_weights must NOT kaiming-clobber it (heads only). For pretrained
        # variants `layers` MUST be [3, 4, 6, 3] so keys map to torchvision r34.
        if pretrained and layers != [3, 4, 6, 3]:
            # Ensure that layers match if pre-trained
            warnings.warn(
                "Layers must match pre-trained backbone of [3, 4, 6, 3] modifying and continuing."
            )
            layers = [3, 4, 6, 3]

        self.in_dim = in_dim
        self.pretrained = pretrained

        self.block = block
        self.conv1 = nn.Conv2d(
            in_dim, self.in_planes, 7, stride=2, padding=3, bias=False
        )  # stem is always from in_dim -> 64 features for resnet
        self.bn1 = self.batch_norm(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)

        self.layer1 = self._make_layer(
            block=block, layer=layers[0], stride=1, planes=64
        )  # make num layers of block type
        self.layer2 = self._make_layer(
            block=block, layer=layers[1], stride=2, planes=128
        )
        self.layer3 = self._make_layer(
            block=block, layer=layers[2], stride=2, planes=256
        )
        self.layer4 = self._make_layer(
            block=block, layer=layers[3], stride=2, planes=512
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_species = nn.Linear(512, num_class[0])
        self.fc_genus = nn.Linear(512, num_class[1])
        self.fc_subfamily = nn.Linear(512, num_class[2])

        self._init_weights()

    # Not required but being explicit about weight, bias initialization for comprehension
    def _init_weights(self):
        if self.pretrained:
            # Backbone is loaded in load_pretrained(); only (re)init the classifier
            # heads here so we don't clobber the pretrained conv/bn weights. The
            # stem (variant A, in_dim=5) is handled by _inflate_stem, not here.
            for head in (self.fc_species, self.fc_genus, self.fc_subfamily):
                nn.init.normal_(head.weight, std=0.01)
                nn.init.constant_(head.bias, 0)
            return
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def load_pretrained(self):
        """Load torchvision ResNet-34 ImageNet weights into this module.

        Preconditions: layers=[3, 4, 6, 3] (block counts must match torchvision)
        and pretrained=True. Custom layers are either popped entirely (fc.*)
        or they are manipulated and reassigned if we're using in_dim = 5

        Everything else (layerX.Y.conv1/bn1/conv2/bn2) maps 1:1.
        """
        from torchvision.models import resnet34, ResNet34_Weights

        src = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1).state_dict()
        src.pop("fc.weight", None)
        src.pop("fc.bias", None)
        if self.in_dim == 5:
            pretrained_conv1 = src.pop("conv1.weight")
            self._inflate_stem(pretrained_conv1=pretrained_conv1)

        keys_r = self.load_state_dict(src, strict=False)

        expected_missing = {
            "fc_species.weight",
            "fc_species.bias",
            "fc_genus.weight",
            "fc_genus.bias",
            "fc_subfamily.weight",
            "fc_subfamily.bias",
        }
        if self.in_dim == 5:
            expected_missing.add("conv1.weight")

        if set(keys_r.missing_keys) != expected_missing:
            raise ValueError(
                f"Unexpected missing_keys — backbone may be over/under-popped. "
                f"Got {sorted(keys_r.missing_keys)}, expected {sorted(expected_missing)}"
            )

        # unexpected_keys MUST be empty: anything here is a src tensor that found no
        # slot in the model.
        if keys_r.unexpected_keys:
            raise ValueError(
                f"Unexpected keys did not map to the model (rename miss?): "
                f"{sorted(keys_r.unexpected_keys)}"
            )

    @torch.no_grad()
    def _inflate_stem(self, pretrained_conv1):
        """Init the (64, in_dim, 7, 7) stem from torchvision's (64, 3, 7, 7) conv1.

        Channels 0-2 take the pretrained RGB filters; the extra channels (Scharr
        gradient, LCN) are grayscale-derived ≈ a weighted RGB average, so the mean
        of the RGB filters is a sensible edge-detector init. Then rescale the whole
        stem by 3/in_dim so expected pre-activation variance matches the pretrained
        regime (sanity check: self.conv1.weight.mean() ≈ pretrained.mean() * 3/in_dim).
        """
        new_w = torch.zeros(64, self.in_dim, 7, 7)
        new_w[:, :3, :, :] = pretrained_conv1
        new_w[:, 3:, :, :] = pretrained_conv1.mean(dim=1, keepdim=True)
        new_w.mul_(3 / self.in_dim)
        self.conv1.weight.data.copy_(new_w)

    def _make_layer(
        self, block: Type[BasicBlock], layer: int, stride: int, planes: int
    ):
        downsample = None
        if stride != 1 or self.in_planes != planes:
            downsample = nn.Sequential(
                # bias=False to match torchvision (the following BN absorbs any bias),
                # and so pretrained downsample keys map 1:1 with no extra params.
                nn.Conv2d(
                    self.in_planes, planes, kernel_size=1, stride=stride, bias=False
                ),
                self.batch_norm(planes),
            )  # required to downsample

        layers = []
        layers.append(
            block(
                in_planes=self.in_planes,
                planes=planes,
                stride=stride,
                downsample=downsample,
            )
        )
        self.in_planes = planes
        for _ in range(1, layer):
            layers.append(block(planes=planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.max_pool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avg_pool(out)
        out = torch.flatten(out, 1)

        out_species = self.fc_species(out)
        out_genus = self.fc_genus(out)
        out_subfamily = self.fc_subfamily(out)

        return out_species, out_genus, out_subfamily
