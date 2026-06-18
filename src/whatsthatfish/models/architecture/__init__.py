"""Network architectures (pure nn.Modules, no training/DB deps)."""

from .custom_resnet import CustomResnet, BasicBlock

__all__ = ["CustomResnet", "BasicBlock"]
