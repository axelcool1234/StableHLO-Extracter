import argparse
import gc
import os
import random
import sys
from typing import Callable

import jax.numpy as jnp
import torch
import torchax as tx
import torchvision
from jax import ShapeDtypeStruct
from jax import jit
from jax.export import export as jax_export
from torchax.export import (
    exported_program_to_stablehlo as jax_exported_program_to_stablehlo,
)
from torch.export import export as torch_export
from torch.nn.modules import Module
from torch_xla.stablehlo import (
    exported_program_to_stablehlo as torch_exported_program_to_stablehlo,
)
from torch_xla.stablehlo import (
    StableHLOGraphModule,
)
from typing_extensions import override

IMAGE_TENSORS: list[tuple[int, int, int, int]] = [
    (2**batch_exponent, 3, size, size)
    for batch_exponent in range(7)
    for size in [224, 384, 518]
]
VIDEO_TENSORS: list[tuple[int, int, int, int, int]] = [
    (2**batch_exponent, 3, 2**frame_exponent, 112, 112)
    for batch_exponent in range(7)
    for frame_exponent in range(6)
]
MVIT_TENSORS = [(16, 3, 16, 224, 224)]
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


class ModelWrapper:
    def __init__(
        self,
        model_name: str,
        constructor: Callable[..., Module],
        default_weight: torchvision.models.WeightsEnum,
        sizes: list[tuple[int, int, int, int]] | list[tuple[int, int, int, int, int]],
    ):
        self.model: None | torch.nn.modules.Module = None
        self.model_name: str = model_name
        self.sizes: (
            list[tuple[int, int, int, int]] | list[tuple[int, int, int, int, int]]
        ) = sizes
        self._constructor: Callable[..., torch.nn.Module] = constructor
        self._default_weight: torchvision.models.WeightsEnum = default_weight

    def construct(self):
        if "quantization" in self.model_name:
            self.model = self._constructor(weights=self._default_weight, quantize=True)
        else:
            self.model = self._constructor(weights=self._default_weight)

    def destruct(self):
        if self.model is not None:
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()  # pyright: ignore[reportUnusedCallResult]

    @override
    def __eq__(self, other: object):
        if isinstance(other, ModelWrapper):
            return self.model_name == other.model_name
        if isinstance(other, str):
            return self.model_name == other
        return NotImplemented

    @override
    def __hash__(self):
        return hash(self.model_name)


models = {
    # AlexNet
    ModelWrapper(
        "alexnet",
        torchvision.models.alexnet,
        torchvision.models.AlexNet_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    # ConvNeXt
    ModelWrapper(
        "convnext_tiny",
        torchvision.models.convnext_tiny,
        torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    ModelWrapper(
        "convnext_small",
        torchvision.models.convnext_small,
        torchvision.models.ConvNeXt_Small_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    ModelWrapper(
        "convnext_base",
        torchvision.models.convnext_base,
        torchvision.models.ConvNeXt_Base_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    ModelWrapper(
        "convnext_large",
        torchvision.models.convnext_large,
        torchvision.models.ConvNeXt_Large_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    # DenseNet
    ModelWrapper(
        "densenet121",
        torchvision.models.densenet121,
        torchvision.models.DenseNet121_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    ModelWrapper(
        "densenet161",
        torchvision.models.densenet161,
        torchvision.models.DenseNet161_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    ModelWrapper(
        "densenet169",
        torchvision.models.densenet169,
        torchvision.models.DenseNet169_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    ModelWrapper(
        "densenet201",
        torchvision.models.densenet201,
        torchvision.models.DenseNet201_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    # EfficientNet
    ModelWrapper(
        "efficientnet_b0",
        torchvision.models.efficientnet_b0,
        torchvision.models.EfficientNet_B0_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    ModelWrapper(
        "efficientnet_b1",
        torchvision.models.efficientnet_b1,
        torchvision.models.EfficientNet_B1_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    ModelWrapper(
        "efficientnet_b2",
        torchvision.models.efficientnet_b2,
        torchvision.models.EfficientNet_B2_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    ModelWrapper(
        "efficientnet_b3",
        torchvision.models.efficientnet_b3,
        torchvision.models.EfficientNet_B3_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    ModelWrapper(
        "efficientnet_b4",
        torchvision.models.efficientnet_b4,
        torchvision.models.EfficientNet_B4_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    ModelWrapper(
        "efficientnet_b5",
        torchvision.models.efficientnet_b5,
        torchvision.models.EfficientNet_B5_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    ModelWrapper(
        "efficientnet_b6",
        torchvision.models.efficientnet_b6,
        torchvision.models.EfficientNet_B6_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    ModelWrapper(
        "efficientnet_b7",
        torchvision.models.efficientnet_b7,
        torchvision.models.EfficientNet_B7_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    ModelWrapper(
        "efficientnet_v2_s",
        torchvision.models.efficientnet_v2_s,
        torchvision.models.EfficientNet_V2_S_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    ModelWrapper(
        "efficientnet_v2_m",
        torchvision.models.efficientnet_v2_m,
        torchvision.models.EfficientNet_V2_M_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    ModelWrapper(
        "efficientnet_v2_l",
        torchvision.models.efficientnet_v2_l,
        torchvision.models.EfficientNet_V2_L_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    # GoogLeNet
    ModelWrapper(
        "googlenet",
        torchvision.models.googlenet,
        torchvision.models.GoogLeNet_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    # Inception
    ModelWrapper(
        "inception_v3",
        torchvision.models.inception_v3,
        torchvision.models.Inception_V3_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    # MNASNet
    ModelWrapper(
        "mnasnet0_5",
        torchvision.models.mnasnet0_5,
        torchvision.models.MNASNet0_5_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    ModelWrapper(
        "mnasnet0_75",
        torchvision.models.mnasnet0_75,
        torchvision.models.MNASNet0_75_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    ModelWrapper(
        "mnasnet1_0",
        torchvision.models.mnasnet1_0,
        torchvision.models.MNASNet1_0_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    ModelWrapper(
        "mnasnet1_3",
        torchvision.models.mnasnet1_3,
        torchvision.models.MNASNet1_3_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    # MobileNetV2
    ModelWrapper(
        "mobilenet_v2",
        torchvision.models.mobilenet_v2,
        torchvision.models.MobileNet_V2_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    # MobileNetV3
    ModelWrapper(
        "mobilenet_v3_large",
        torchvision.models.mobilenet_v3_large,
        torchvision.models.MobileNet_V3_Large_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    ModelWrapper(
        "mobilenet_v3_small",
        torchvision.models.mobilenet_v3_small,
        torchvision.models.MobileNet_V3_Small_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    # RegNet
    ModelWrapper(
        "regnet_y_400mf",
        torchvision.models.regnet_y_400mf,
        torchvision.models.RegNet_Y_400MF_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    ModelWrapper(
        "regnet_y_800mf",
        torchvision.models.regnet_y_800mf,
        torchvision.models.RegNet_Y_800MF_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    ModelWrapper(
        "regnet_y_1_6gf",
        torchvision.models.regnet_y_1_6gf,
        torchvision.models.RegNet_Y_1_6GF_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    ModelWrapper(
        "regnet_y_3_2gf",
        torchvision.models.regnet_y_3_2gf,
        torchvision.models.RegNet_Y_3_2GF_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    ModelWrapper(
        "regnet_y_8gf",
        torchvision.models.regnet_y_8gf,
        torchvision.models.RegNet_Y_8GF_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    ModelWrapper(
        "regnet_y_16gf",
        torchvision.models.regnet_y_16gf,
        torchvision.models.RegNet_Y_16GF_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    ModelWrapper(
        "regnet_y_32gf",
        torchvision.models.regnet_y_32gf,
        torchvision.models.RegNet_Y_32GF_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    ModelWrapper(
        "regnet_y_128gf",
        torchvision.models.regnet_y_128gf,
        torchvision.models.RegNet_Y_128GF_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    ModelWrapper(
        "regnet_x_400mf",
        torchvision.models.regnet_x_400mf,
        torchvision.models.RegNet_X_400MF_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    ModelWrapper(
        "regnet_x_800mf",
        torchvision.models.regnet_x_800mf,
        torchvision.models.RegNet_X_800MF_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    ModelWrapper(
        "regnet_x_1_6gf",
        torchvision.models.regnet_x_1_6gf,
        torchvision.models.RegNet_X_1_6GF_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    ModelWrapper(
        "regnet_x_3_2gf",
        torchvision.models.regnet_x_3_2gf,
        torchvision.models.RegNet_X_3_2GF_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    ModelWrapper(
        "regnet_x_8gf",
        torchvision.models.regnet_x_8gf,
        torchvision.models.RegNet_X_8GF_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    ModelWrapper(
        "regnet_x_16gf",
        torchvision.models.regnet_x_16gf,
        torchvision.models.RegNet_X_16GF_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    ModelWrapper(
        "regnet_x_32gf",
        torchvision.models.regnet_x_32gf,
        torchvision.models.RegNet_X_32GF_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    # ResNet
    ModelWrapper(
        "resnet18",
        torchvision.models.resnet18,
        torchvision.models.ResNet18_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    ModelWrapper(
        "resnet34",
        torchvision.models.resnet34,
        torchvision.models.ResNet34_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    ModelWrapper(
        "resnet50",
        torchvision.models.resnet50,
        torchvision.models.ResNet50_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    ModelWrapper(
        "resnet101",
        torchvision.models.resnet101,
        torchvision.models.ResNet101_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    ModelWrapper(
        "resnet152",
        torchvision.models.resnet152,
        torchvision.models.ResNet152_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    ModelWrapper(
        "resnext50_32x4d",
        torchvision.models.resnext50_32x4d,
        torchvision.models.ResNeXt50_32X4D_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    ModelWrapper(
        "resnext101_32x8d",
        torchvision.models.resnext101_32x8d,
        torchvision.models.ResNeXt101_32X8D_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    ModelWrapper(
        "resnext101_64x4d",
        torchvision.models.resnext101_64x4d,
        torchvision.models.ResNeXt101_64X4D_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    ModelWrapper(
        "wide_resnet50_2",
        torchvision.models.wide_resnet50_2,
        torchvision.models.Wide_ResNet50_2_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    ModelWrapper(
        "wide_resnet101_2",
        torchvision.models.wide_resnet101_2,
        torchvision.models.Wide_ResNet101_2_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    # ShuffleNet
    ModelWrapper(
        "shufflenet_v2_x0_5",
        torchvision.models.shufflenet_v2_x0_5,
        torchvision.models.ShuffleNet_V2_X0_5_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    ModelWrapper(
        "shufflenet_v2_x1_0",
        torchvision.models.shufflenet_v2_x1_0,
        torchvision.models.ShuffleNet_V2_X1_0_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    ModelWrapper(
        "shufflenet_v2_x1_5",
        torchvision.models.shufflenet_v2_x1_5,
        torchvision.models.ShuffleNet_V2_X1_5_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    ModelWrapper(
        "shufflenet_v2_x2_0",
        torchvision.models.shufflenet_v2_x2_0,
        torchvision.models.ShuffleNet_V2_X2_0_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    # SqueezeNet
    ModelWrapper(
        "squeezenet1_0",
        torchvision.models.squeezenet1_0,
        torchvision.models.SqueezeNet1_0_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    ModelWrapper(
        "squeezenet1_1",
        torchvision.models.squeezenet1_1,
        torchvision.models.SqueezeNet1_1_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    # VGG
    ModelWrapper(
        "vgg11",
        torchvision.models.vgg11,
        torchvision.models.VGG11_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    ModelWrapper(
        "vgg11_bn",
        torchvision.models.vgg11_bn,
        torchvision.models.VGG11_BN_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    ModelWrapper(
        "vgg13",
        torchvision.models.vgg13,
        torchvision.models.VGG13_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    ModelWrapper(
        "vgg13_bn",
        torchvision.models.vgg13_bn,
        torchvision.models.VGG13_BN_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    ModelWrapper(
        "vgg16",
        torchvision.models.vgg16,
        torchvision.models.VGG16_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    ModelWrapper(
        "vgg16_bn",
        torchvision.models.vgg16_bn,
        torchvision.models.VGG16_BN_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    ModelWrapper(
        "vgg19",
        torchvision.models.vgg19,
        torchvision.models.VGG19_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    ModelWrapper(
        "vgg19_bn",
        torchvision.models.vgg19_bn,
        torchvision.models.VGG19_BN_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    # VisionTransformer
    ModelWrapper(
        "vit_b_16",
        torchvision.models.vit_b_16,
        torchvision.models.ViT_B_16_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    ModelWrapper(
        "vit_b_32",
        torchvision.models.vit_b_32,
        torchvision.models.ViT_B_32_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    ModelWrapper(
        "vit_l_16",
        torchvision.models.vit_l_16,
        torchvision.models.ViT_L_16_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    ModelWrapper(
        "vit_l_32",
        torchvision.models.vit_l_32,
        torchvision.models.ViT_L_32_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    ModelWrapper(
        "vit_h_14",
        torchvision.models.vit_h_14,
        torchvision.models.ViT_H_14_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    # Swin Transformers
    ModelWrapper(
        "swin_t",
        torchvision.models.swin_t,
        torchvision.models.Swin_T_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    ModelWrapper(
        "swin_s",
        torchvision.models.swin_s,
        torchvision.models.Swin_S_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    ModelWrapper(
        "swin_b",
        torchvision.models.swin_b,
        torchvision.models.Swin_B_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    ModelWrapper(
        "swin_v2_t",
        torchvision.models.swin_v2_t,
        torchvision.models.Swin_V2_T_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    ModelWrapper(
        "swin_v2_s",
        torchvision.models.swin_v2_s,
        torchvision.models.Swin_V2_S_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    ModelWrapper(
        "swin_v2_b",
        torchvision.models.swin_v2_b,
        torchvision.models.Swin_V2_B_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    # MaxVit
    ModelWrapper(
        "maxvit_t",
        torchvision.models.maxvit_t,
        torchvision.models.MaxVit_T_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    # TODO: Label these
    # NOTE: Detection models mutate module attributes during forward/export. Skipping...
    # ModelWrapper(
    #     "detection.fasterrcnn_resnet50_fpn",
    #     torchvision.models.detection.fasterrcnn_resnet50_fpn,
    #     torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
    #     sizes=IMAGE_TENSORS,
    # ),
    # ModelWrapper(
    #     "detection.fasterrcnn_resnet50_fpn_v2",
    #     torchvision.models.detection.fasterrcnn_resnet50_fpn_v2,
    #     torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
    #     sizes=IMAGE_TENSORS,
    # ),
    # ModelWrapper(
    #     "detection.fasterrcnn_mobilenet_v3_large_fpn",
    #     torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn,
    #     torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT,
    #     sizes=IMAGE_TENSORS,
    # ),
    # ModelWrapper(
    #     "detection.fasterrcnn_mobilenet_v3_large_320_fpn",
    #     torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn,
    #     torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT,
    #     sizes=IMAGE_TENSORS,
    # ),
    # ModelWrapper(
    #     "detection.fcos_resnet50_fpn",
    #     torchvision.models.detection.fcos_resnet50_fpn,
    #     torchvision.models.detection.FCOS_ResNet50_FPN_Weights.DEFAULT,
    #     sizes=IMAGE_TENSORS,
    # ),
    # ModelWrapper(
    #     "detection.keypointrcnn_resnet50_fpn",
    #     torchvision.models.detection.keypointrcnn_resnet50_fpn,
    #     torchvision.models.detection.KeypointRCNN_ResNet50_FPN_Weights.DEFAULT,
    #     sizes=IMAGE_TENSORS,
    # ),
    # ModelWrapper(
    #     "detection.maskrcnn_resnet50_fpn",
    #     torchvision.models.detection.maskrcnn_resnet50_fpn,
    #     torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT,
    #     sizes=IMAGE_TENSORS,
    # ),
    # ModelWrapper(
    #     "detection.maskrcnn_resnet50_fpn_v2",
    #     torchvision.models.detection.maskrcnn_resnet50_fpn_v2,
    #     torchvision.models.detection.MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
    #     sizes=IMAGE_TENSORS,
    # ),
    # ModelWrapper(
    #     "detection.retinanet_resnet50_fpn",
    #     torchvision.models.detection.retinanet_resnet50_fpn,
    #     torchvision.models.detection.RetinaNet_ResNet50_FPN_Weights.DEFAULT,
    #     sizes=IMAGE_TENSORS,
    # ),
    # ModelWrapper(
    #     "detection.retinanet_resnet50_fpn_v2",
    #     torchvision.models.detection.retinanet_resnet50_fpn_v2,
    #     torchvision.models.detection.RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT,
    #     sizes=IMAGE_TENSORS,
    # ),
    # ModelWrapper(
    #     "detection.ssd300_vgg16",
    #     torchvision.models.detection.ssd300_vgg16,
    #     torchvision.models.detection.SSD300_VGG16_Weights.DEFAULT,
    #     sizes=IMAGE_TENSORS,
    # ),
    # ModelWrapper(
    #     "detection.ssdlite320_mobilenet_v3_large",
    #     torchvision.models.detection.ssdlite320_mobilenet_v3_large,
    #     torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT,
    #     sizes=IMAGE_TENSORS,
    # ),
    # NOTE: RAFT models mutate during forward. Skipping...
    # ModelWrapper(
    #     "optical_flow.raft_large",
    #     torchvision.models.optical_flow.raft_large,
    #     torchvision.models.optical_flow.Raft_Large_Weights.DEFAULT,
    #     sizes=IMAGE_TENSORS,
    # ),
    # ModelWrapper(
    #     "optical_flow.raft_small",
    #     torchvision.models.optical_flow.raft_small,
    #     torchvision.models.optical_flow.Raft_Small_Weights.DEFAULT,
    #     sizes=IMAGE_TENSORS,
    # ),
    # NOTE: Quantization isn't supported by torch-xla export. Skipping...
    # ModelWrapper(
    #     "quantization.googlenet",
    #     torchvision.models.quantization.googlenet,
    #     torchvision.models.quantization.GoogLeNet_QuantizedWeights.DEFAULT,
    #     sizes=IMAGE_TENSORS,
    # ),
    # ModelWrapper(
    #     "quantization.inception_v3",
    #     torchvision.models.quantization.inception_v3,
    #     torchvision.models.quantization.Inception_V3_QuantizedWeights.DEFAULT,
    #     sizes=IMAGE_TENSORS,
    # ),
    # ModelWrapper(
    #     "quantization.mobilenet_v2",
    #     torchvision.models.quantization.mobilenet_v2,
    #     torchvision.models.quantization.MobileNet_V2_QuantizedWeights.DEFAULT,
    #     sizes=IMAGE_TENSORS,
    # ),
    # ModelWrapper(
    #     "quantization.mobilenet_v3_large",
    #     torchvision.models.quantization.mobilenet_v3_large,
    #     torchvision.models.quantization.MobileNet_V3_Large_QuantizedWeights.DEFAULT,
    #     sizes=IMAGE_TENSORS,
    # ),
    # ModelWrapper(
    #     "quantization.resnet18",
    #     torchvision.models.quantization.resnet18,
    #     torchvision.models.quantization.ResNet18_QuantizedWeights.DEFAULT,
    #     sizes=IMAGE_TENSORS,
    # ),
    # ModelWrapper(
    #     "quantization.resnet50",
    #     torchvision.models.quantization.resnet50,
    #     torchvision.models.quantization.ResNet50_QuantizedWeights.DEFAULT,
    #     sizes=IMAGE_TENSORS,
    # ),
    # ModelWrapper(
    #     "quantization.resnext101_32x8d",
    #     torchvision.models.quantization.resnext101_32x8d,
    #     torchvision.models.quantization.ResNeXt101_32X8D_QuantizedWeights.DEFAULT,
    #     sizes=IMAGE_TENSORS,
    # ),
    # ModelWrapper(
    #     "quantization.resnext101_64x4d",
    #     torchvision.models.quantization.resnext101_64x4d,
    #     torchvision.models.quantization.ResNeXt101_64X4D_QuantizedWeights.DEFAULT,
    #     sizes=IMAGE_TENSORS,
    # ),
    # ModelWrapper(
    #     "quantization.shufflenet_v2_x0_5",
    #     torchvision.models.quantization.shufflenet_v2_x0_5,
    #     torchvision.models.quantization.ShuffleNet_V2_X0_5_QuantizedWeights.DEFAULT,
    #     sizes=IMAGE_TENSORS,
    # ),
    # ModelWrapper(
    #     "quantization.shufflenet_v2_x1_0",
    #     torchvision.models.quantization.shufflenet_v2_x1_0,
    #     torchvision.models.quantization.ShuffleNet_V2_X1_0_QuantizedWeights.DEFAULT,
    #     sizes=IMAGE_TENSORS,
    # ),
    # ModelWrapper(
    #     "quantization.shufflenet_v2_x1_5",
    #     torchvision.models.quantization.shufflenet_v2_x1_5,
    #     torchvision.models.quantization.ShuffleNet_V2_X1_5_QuantizedWeights.DEFAULT,
    #     sizes=IMAGE_TENSORS,
    # ),
    # ModelWrapper(
    #     "quantization.shufflenet_v2_x2_0",
    #     torchvision.models.quantization.shufflenet_v2_x2_0,
    #     torchvision.models.quantization.ShuffleNet_V2_X2_0_QuantizedWeights.DEFAULT,
    #     sizes=IMAGE_TENSORS,
    # ),
    ModelWrapper(
        "segmentation.deeplabv3_mobilenet_v3_large",
        torchvision.models.segmentation.deeplabv3_mobilenet_v3_large,
        torchvision.models.segmentation.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    ModelWrapper(
        "segmentation.deeplabv3_resnet50",
        torchvision.models.segmentation.deeplabv3_resnet50,
        torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    ModelWrapper(
        "segmentation.deeplabv3_resnet101",
        torchvision.models.segmentation.deeplabv3_resnet101,
        torchvision.models.segmentation.DeepLabV3_ResNet101_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    ModelWrapper(
        "segmentation.fcn_resnet50",
        torchvision.models.segmentation.fcn_resnet50,
        torchvision.models.segmentation.FCN_ResNet50_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    ModelWrapper(
        "segmentation.fcn_resnet101",
        torchvision.models.segmentation.fcn_resnet101,
        torchvision.models.segmentation.FCN_ResNet101_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    ModelWrapper(
        "segmentation.lraspp_mobilenet_v3_large",
        torchvision.models.segmentation.lraspp_mobilenet_v3_large,
        torchvision.models.segmentation.LRASPP_MobileNet_V3_Large_Weights.DEFAULT,
        sizes=IMAGE_TENSORS,
    ),
    ModelWrapper(
        "video.mvit_v1_b",
        torchvision.models.video.mvit_v1_b,
        torchvision.models.video.MViT_V1_B_Weights.DEFAULT,
        sizes=MVIT_TENSORS,
    ),
    ModelWrapper(
        "video.mvit_v2_s",
        torchvision.models.video.mvit_v2_s,
        torchvision.models.video.MViT_V2_S_Weights.DEFAULT,
        sizes=MVIT_TENSORS,
    ),
    ModelWrapper(
        "video.r3d_18",
        torchvision.models.video.r3d_18,
        torchvision.models.video.R3D_18_Weights.DEFAULT,
        sizes=VIDEO_TENSORS,
    ),
    ModelWrapper(
        "video.mc3_18",
        torchvision.models.video.mc3_18,
        torchvision.models.video.MC3_18_Weights.DEFAULT,
        sizes=VIDEO_TENSORS,
    ),
    ModelWrapper(
        "video.r2plus1d_18",
        torchvision.models.video.r2plus1d_18,
        torchvision.models.video.R2Plus1D_18_Weights.DEFAULT,
        sizes=VIDEO_TENSORS,
    ),
    ModelWrapper(
        "video.s3d",
        torchvision.models.video.s3d,
        torchvision.models.video.S3D_Weights.DEFAULT,
        sizes=VIDEO_TENSORS,
    ),
    ModelWrapper(
        "video.swin3d_t",
        torchvision.models.video.swin3d_t,
        torchvision.models.video.Swin3D_T_Weights.DEFAULT,
        sizes=VIDEO_TENSORS,
    ),
    ModelWrapper(
        "video.swin3d_s",
        torchvision.models.video.swin3d_s,
        torchvision.models.video.Swin3D_S_Weights.DEFAULT,
        sizes=VIDEO_TENSORS,
    ),
    ModelWrapper(
        "video.swin3d_b",
        torchvision.models.video.swin3d_b,
        torchvision.models.video.Swin3D_B_Weights.DEFAULT,
        sizes=VIDEO_TENSORS,
    ),
}


def main():
    parser = argparse.ArgumentParser(description="StableHLO Extracter")
    _ = parser.add_argument(
        "-r",
        "--random",
        action="store_true",
        help="Use one random tensor size for each model",
    )
    _ = parser.add_argument(
        "-p", "--print", action="store_true", help="Print the StableHLO"
    )
    _ = parser.add_argument(
        "-b",
        "--bytecode",
        action="store_true",
        help="Extract StableHLO bytecode instead of text",
    )
    _ = parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="out",
        help='Directory to save StableHLO outputs (one file per model). Files are saved as "[model name][tensor size].[bin/txt]"',
    )
    _ = parser.add_argument(
        "--override",
        action="store_true",
        help="Override existing files in the output directory",
    )
    _ = parser.add_argument(
        "--no-output",
        action="store_true",
        help="Do not write any files to disk",
    )
    _ = parser.add_argument(
        "--model",
        type=str,
        help="Extract a specific model.",
    )
    _ = parser.add_argument(
        "--torch",
        action="store_true",
        help="Extracts using Torch's extract function instead of Jax's extract function.",
    )
    args = parser.parse_args()
    if args.model:  # pyright: ignore[reportAny]
        extract_and_print_one(
            args.model,  # pyright: ignore[reportAny]
            random_tensor=args.random,  # pyright: ignore[reportAny]
            print_hlo=args.print,  # pyright: ignore[reportAny]
            bytecode=args.bytecode,  # pyright: ignore[reportAny]
            output_dir=None if args.no_output else args.output_dir,  # pyright: ignore[reportAny]
            override=args.override,  # pyright: ignore[reportAny]
            extract_with_torch=args.torch,  # pyright: ignore[reportAny]
        )
    else:
        extract_and_print_all(
            random_tensor=args.random,  # pyright: ignore[reportAny]
            print_hlo=args.print,  # pyright: ignore[reportAny]
            bytecode=args.bytecode,  # pyright: ignore[reportAny]
            output_dir=None if args.no_output else args.output_dir,  # pyright: ignore[reportAny]
            override=args.override,  # pyright: ignore[reportAny]
            extract_with_torch=args.torch,  # pyright: ignore[reportAny]
        )


def extract_and_print_all(**kwargs):  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
    random_tensor: bool = kwargs.pop("random_tensor", False)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    for model in models:
        sizes = [random.choice(model.sizes)] if random_tensor else model.sizes
        for tensor_size in sizes:
            extract_and_print(model, tensor_size, **kwargs)


def extract_and_print_one(model_name: str, **kwargs):  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
    model = _get_model_by_name(model_name)
    random_tensor: bool = kwargs.pop("random_tensor", False)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    sizes = [random.choice(model.sizes)] if random_tensor else model.sizes
    for tensor_size in sizes:
        extract_and_print(model, tensor_size, **kwargs)


def extract_and_print(
    model: ModelWrapper,
    tensor_size: tuple[int, int, int, int] | tuple[int, int, int, int, int],
    print_hlo: bool = False,
    bytecode: bool = False,
    output_dir: str | None = None,
    override: bool = False,
    extract_with_torch: bool = False,
):
    end = "\n" if print_hlo else ""
    status = f" {end}{GREEN}[ok]"
    try:
        print(
            f"{CYAN}extracting {BOLD}{model.model_name}{RESET}{BOLD}{tensor_size}{RESET}...",
            file=sys.stderr,
            end=end,
        )
        # Skip check
        filename = f"{model.model_name}{tensor_size}.{'bin' if bytecode else 'txt'}"
        path = None
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, filename)
            if os.path.exists(path) and not override and not print_hlo:
                status = f" {end}{YELLOW}[skip]"
                return
        elif not print_hlo:
            status = f" {end}{YELLOW}[skip]"
            return

        # Construction (expensive)
        model.construct()
        assert model.model is not None
        model.model = model.model.eval()

        # Generate random filled tensor
        # Export
        if extract_with_torch:
            stablehlo_program = export_from_torch(model.model, tensor_size)
        else:
            stablehlo_program = export_from_jax(model.model, tensor_size)

        # Output to file
        if output_dir:
            assert path is not None
            if override or not os.path.exists(path):
                with open(path, "wb" if bytecode else "w") as f:
                    _ = f.write(_get_content(stablehlo_program, bytecode))

        # Print to stdout
        if print_hlo:
            print(_get_content(stablehlo_program, bytecode))
    except Exception as e:
        status = f" {end}{RED}[failed]{RESET}\n{YELLOW}{e}{RESET}"
        print(e, file=sys.stderr)
    finally:
        print(status, file=sys.stderr)
        model.destruct()


def export_from_torch(
    model: Module,
    tensor_size: tuple[int, int, int, int] | tuple[int, int, int, int, int],
) -> StableHLOGraphModule:
    sample_input = (torch.randn(tensor_size),)
    exported = torch_export(model, sample_input)
    return torch_exported_program_to_stablehlo(exported)


def export_from_jax(
    model: Module,
    tensor_size: tuple[int, int, int, int] | tuple[int, int, int, int, int],
):
    # Get weights + JAX function
    weights, jfunc = tx.extract_jax(model)
    sample_input = (ShapeDtypeStruct(tensor_size, jnp.float32),)
    return jax_export(jit(jfunc))(weights, sample_input)


def _get_model_by_name(model_name: str) -> ModelWrapper:
    if model_name not in models:
        raise Exception("Model does not exist!")
    for model in models:
        if model.model_name == model_name:
            return model
    raise Exception("Model does not exist!")


def _get_content(prog, bytecode: bool):
    if isinstance(prog, StableHLOGraphModule):
        return (
            prog.get_stablehlo_bytecode("forward")
            if bytecode
            else prog.get_stablehlo_text("forward")
        )
    # JAX Exported object
    elif hasattr(prog, "mlir_module"):
        return str(prog.mlir_module())
    else:
        raise TypeError(f"Unsupported export type: {type(prog)}")


if __name__ == "__main__":
    main()
