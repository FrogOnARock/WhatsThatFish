import torch
import torch.nn as nn
from typing import Type
class BasicBlock(nn.Module):

    # Block expansion not required given only implementing resnet-32 with basic block.

    def __init__(self,
                 planes: int,
                 in_planes: int = None,
                 downsample: nn.Sequential = None,
                 stride: int = 1):
        super().__init__()
        self.stride = stride

        if in_planes is None:
            in_planes = planes

        self.down_sample = downsample
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=self.stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        """
        Pattern is relatively straightforward, convolution -> batch norm -> relu, but on
        second convolution, add back in dimension aligned identity before relu.
        Convolution -> batch norm -> identity addition -> relu -> out
        """
        identity = x # set identity for all blocks that are not the first block.. dim_identity = dim_tensor

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.down_sample:
            identity = self.down_sample(x) # we down sample our identity if we have to, first block in layer
            # so that it matches the dimensions of the tensor after the convolutions are passed over it with stride = 2
            # we reduce spatial dimensions of the tensor in the new layer, increase num of feature maps, therefore
            # we must apply conv2d + batch norm to identity to perform same operations

        out = out + identity # Add skip
        out = self.relu(out)

        return out



class CustomResnet(nn.Module):
    def __init__(self,
                 block: Type[BasicBlock],
                 layers: list[int],
                 num_class: list[int],
                 in_dim: int = 5,
                 in_planes: int = 64,
                 batch_norm = None,
                 ):
        super().__init__()
        if batch_norm is None:
            self.batch_norm = nn.BatchNorm2d
        self.in_planes = in_planes

        self.block = block
        self.conv1 = nn.Conv2d(in_dim, self.in_planes, 7, stride=2, padding=3, bias=False) # stem is always from in_dim ->
        # 64 features for resnet
        self.bn = self.batch_norm(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)

        self.layer1 = self._make_layer(block=block, layer=layers[0], stride=1, planes=64) #make num layers of block type
        self.layer2 = self._make_layer(block=block, layer=layers[1], stride=2, planes=128)
        self.layer3 = self._make_layer(block=block, layer=layers[2], stride=2, planes=256)
        self.layer4 = self._make_layer(block=block, layer=layers[3], stride=2, planes=512)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_species = nn.Linear(512, num_class[0])
        self.fc_genus = nn.Linear(512, num_class[1])
        self.fc_subfamily = nn.Linear(512, num_class[2])

        self._init_weights()

    # Not required but being explicit about weight, bias initialization for comprehension
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self,
                    block: Type[BasicBlock],
                    layer: int,
                    stride: int,
                    planes: int
                    ):
        downsample = None
        if stride != 1 or self.in_planes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes, kernel_size=1, stride=stride),
                self.batch_norm(planes)
            ) #required to downsample


        layers = []
        layers.append(
            block(in_planes=self.in_planes, planes=planes, stride=stride, downsample=downsample)
        )
        self.in_planes = planes
        for _ in range(1, layer):
            layers.append(
                block(
                    planes=planes
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn(out)
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




