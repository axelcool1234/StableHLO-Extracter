import torch
import torchvision
from torch.export import export as torch_export
from torch_xla.stablehlo import exported_program_to_stablehlo

models = {
    # AlexNet
    "alexnet": torchvision.models.alexnet(
        weights=torchvision.models.AlexNet_Weights.DEFAULT
    ),
    # ConvNeXt
    "convnext_tiny": torchvision.models.convnext_tiny(
        weights=torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT
    ),
    "convnext_small": torchvision.models.convnext_small(
        weights=torchvision.models.ConvNeXt_Small_Weights.DEFAULT
    ),
    "convnext_base": torchvision.models.convnext_base(
        weights=torchvision.models.ConvNeXt_Base_Weights.DEFAULT
    ),
    "convnext_large": torchvision.models.convnext_large(
        weights=torchvision.models.ConvNeXt_Large_Weights.DEFAULT
    ),
    # DenseNet
    "densenet121": torchvision.models.densenet121(
        weights=torchvision.models.DenseNet121_Weights.DEFAULT
    ),
    "densenet161": torchvision.models.densenet161(
        weights=torchvision.models.DenseNet161_Weights.DEFAULT
    ),
    "densenet169": torchvision.models.densenet169(
        weights=torchvision.models.DenseNet169_Weights.DEFAULT
    ),
    "densenet201": torchvision.models.densenet201(
        weights=torchvision.models.DenseNet201_Weights.DEFAULT
    ),
    # EfficientNet
    "efficientnet_b0": torchvision.models.efficientnet_b0(
        weights=torchvision.models.EfficientNet_B0_Weights.DEFAULT
    ),
    "efficientnet_b1": torchvision.models.efficientnet_b1(
        weights=torchvision.models.EfficientNet_B1_Weights.DEFAULT
    ),
    "efficientnet_b2": torchvision.models.efficientnet_b2(
        weights=torchvision.models.EfficientNet_B2_Weights.DEFAULT
    ),
    "efficientnet_b3": torchvision.models.efficientnet_b3(
        weights=torchvision.models.EfficientNet_B3_Weights.DEFAULT
    ),
    "efficientnet_b4": torchvision.models.efficientnet_b4(
        weights=torchvision.models.EfficientNet_B4_Weights.DEFAULT
    ),
    "efficientnet_b5": torchvision.models.efficientnet_b5(
        weights=torchvision.models.EfficientNet_B5_Weights.DEFAULT
    ),
    "efficientnet_b6": torchvision.models.efficientnet_b6(
        weights=torchvision.models.EfficientNet_B6_Weights.DEFAULT
    ),
    "efficientnet_b7": torchvision.models.efficientnet_b7(
        weights=torchvision.models.EfficientNet_B7_Weights.DEFAULT
    ),
    "efficientnet_v2_s": torchvision.models.efficientnet_v2_s(
        weights=torchvision.models.EfficientNet_V2_S_Weights.DEFAULT
    ),
    "efficientnet_v2_m": torchvision.models.efficientnet_v2_m(
        weights=torchvision.models.EfficientNet_V2_M_Weights.DEFAULT
    ),
    "efficientnet_v2_l": torchvision.models.efficientnet_v2_l(
        weights=torchvision.models.EfficientNet_V2_L_Weights.DEFAULT
    ),
    # GoogLeNet
    "googlenet": torchvision.models.googlenet(
        weights=torchvision.models.GoogLeNet_Weights.DEFAULT
    ),
    # Inception
    "inception_v3": torchvision.models.inception_v3(
        weights=torchvision.models.Inception_V3_Weights.DEFAULT
    ),
    # MNASNet
    "mnasnet0_5": torchvision.models.mnasnet0_5(
        weights=torchvision.models.MNASNet0_5_Weights.DEFAULT
    ),
    "mnasnet0_75": torchvision.models.mnasnet0_75(
        weights=torchvision.models.MNASNet0_75_Weights.DEFAULT
    ),
    "mnasnet1_0": torchvision.models.mnasnet1_0(
        weights=torchvision.models.MNASNet1_0_Weights.DEFAULT
    ),
    "mnasnet1_3": torchvision.models.mnasnet1_3(
        weights=torchvision.models.MNASNet1_3_Weights.DEFAULT
    ),
    # MobileNetV2
    "mobilenet_v2": torchvision.models.mobilenet_v2(
        weights=torchvision.models.MobileNet_V2_Weights.DEFAULT
    ),
    # MobileNetV3
    "mobilenet_v3_large": torchvision.models.mobilenet_v3_large(
        weights=torchvision.models.MobileNet_V3_Large_Weights.DEFAULT
    ),
    "mobilenet_v3_small": torchvision.models.mobilenet_v3_small(
        weights=torchvision.models.MobileNet_V3_Small_Weights.DEFAULT
    ),
    # RegNet
    "regnet_y_400mf": torchvision.models.regnet_y_400mf(
        weights=torchvision.models.RegNet_Y_400MF_Weights.DEFAULT
    ),
    "regnet_y_800mf": torchvision.models.regnet_y_800mf(
        weights=torchvision.models.RegNet_Y_800MF_Weights.DEFAULT
    ),
    "regnet_y_1_6gf": torchvision.models.regnet_y_1_6gf(
        weights=torchvision.models.RegNet_Y_1_6GF_Weights.DEFAULT
    ),
    "regnet_y_3_2gf": torchvision.models.regnet_y_3_2gf(
        weights=torchvision.models.RegNet_Y_3_2GF_Weights.DEFAULT
    ),
    "regnet_y_8gf": torchvision.models.regnet_y_8gf(
        weights=torchvision.models.RegNet_Y_8GF_Weights.DEFAULT
    ),
    "regnet_y_16gf": torchvision.models.regnet_y_16gf(
        weights=torchvision.models.RegNet_Y_16GF_Weights.DEFAULT
    ),
    "regnet_y_32gf": torchvision.models.regnet_y_32gf(
        weights=torchvision.models.RegNet_Y_32GF_Weights.DEFAULT
    ),
    "regnet_y_128gf": torchvision.models.regnet_y_128gf(
        weights=torchvision.models.RegNet_Y_128GF_Weights.DEFAULT
    ),
    "regnet_x_400mf": torchvision.models.regnet_x_400mf(
        weights=torchvision.models.RegNet_X_400MF_Weights.DEFAULT
    ),
    "regnet_x_800mf": torchvision.models.regnet_x_800mf(
        weights=torchvision.models.RegNet_X_800MF_Weights.DEFAULT
    ),
    "regnet_x_1_6gf": torchvision.models.regnet_x_1_6gf(
        weights=torchvision.models.RegNet_X_1_6GF_Weights.DEFAULT
    ),
    "regnet_x_3_2gf": torchvision.models.regnet_x_3_2gf(
        weights=torchvision.models.RegNet_X_3_2GF_Weights.DEFAULT
    ),
    "regnet_x_8gf": torchvision.models.regnet_x_8gf(
        weights=torchvision.models.RegNet_X_8GF_Weights.DEFAULT
    ),
    "regnet_x_16gf": torchvision.models.regnet_x_16gf(
        weights=torchvision.models.RegNet_X_16GF_Weights.DEFAULT
    ),
    "regnet_x_32gf": torchvision.models.regnet_x_32gf(
        weights=torchvision.models.RegNet_X_32GF_Weights.DEFAULT
    ),
    # ResNet
    "resnet18": torchvision.models.resnet18(
        weights=torchvision.models.ResNet18_Weights.DEFAULT
    ),
    "resnet34": torchvision.models.resnet34(
        weights=torchvision.models.ResNet34_Weights.DEFAULT
    ),
    "resnet50": torchvision.models.resnet50(
        weights=torchvision.models.ResNet50_Weights.DEFAULT
    ),
    "resnet101": torchvision.models.resnet101(
        weights=torchvision.models.ResNet101_Weights.DEFAULT
    ),
    "resnet152": torchvision.models.resnet152(
        weights=torchvision.models.ResNet152_Weights.DEFAULT
    ),
    "resnext50_32x4d": torchvision.models.resnext50_32x4d(
        weights=torchvision.models.ResNeXt50_32X4D_Weights.DEFAULT
    ),
    "resnext101_32x8d": torchvision.models.resnext101_32x8d(
        weights=torchvision.models.ResNeXt101_32X8D_Weights.DEFAULT
    ),
    "resnext101_64x4d": torchvision.models.resnext101_64x4d(
        weights=torchvision.models.ResNeXt101_64X4D_Weights.DEFAULT
    ),
    "wide_resnet50_2": torchvision.models.wide_resnet50_2(
        weights=torchvision.models.Wide_ResNet50_2_Weights.DEFAULT
    ),
    "wide_resnet101_2": torchvision.models.wide_resnet101_2(
        weights=torchvision.models.Wide_ResNet101_2_Weights.DEFAULT
    ),
    # ShuffleNet
    "shufflenet_v2_x0_5": torchvision.models.shufflenet_v2_x0_5(
        weights=torchvision.models.ShuffleNet_V2_X0_5_Weights.DEFAULT
    ),
    "shufflenet_v2_x1_0": torchvision.models.shufflenet_v2_x1_0(
        weights=torchvision.models.ShuffleNet_V2_X1_0_Weights.DEFAULT
    ),
    "shufflenet_v2_x1_5": torchvision.models.shufflenet_v2_x1_5(
        weights=torchvision.models.ShuffleNet_V2_X1_5_Weights.DEFAULT
    ),
    "shufflenet_v2_x2_0": torchvision.models.shufflenet_v2_x2_0(
        weights=torchvision.models.ShuffleNet_V2_X2_0_Weights.DEFAULT
    ),
    # SqueezeNet
    "squeezenet1_0": torchvision.models.squeezenet1_0(
        weights=torchvision.models.SqueezeNet1_0_Weights.DEFAULT
    ),
    "squeezenet1_1": torchvision.models.squeezenet1_1(
        weights=torchvision.models.SqueezeNet1_1_Weights.DEFAULT
    ),
    # VGG
    "vgg11": torchvision.models.vgg11(weights=torchvision.models.VGG11_Weights.DEFAULT),
    "vgg11_bn": torchvision.models.vgg11_bn(
        weights=torchvision.models.VGG11_BN_Weights.DEFAULT
    ),
    "vgg13": torchvision.models.vgg13(weights=torchvision.models.VGG13_Weights.DEFAULT),
    "vgg13_bn": torchvision.models.vgg13_bn(
        weights=torchvision.models.VGG13_BN_Weights.DEFAULT
    ),
    "vgg16": torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT),
    "vgg16_bn": torchvision.models.vgg16_bn(
        weights=torchvision.models.VGG16_BN_Weights.DEFAULT
    ),
    "vgg19": torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.DEFAULT),
    "vgg19_bn": torchvision.models.vgg19_bn(
        weights=torchvision.models.VGG19_BN_Weights.DEFAULT
    ),
    # VisionTransformer
    "vit_b_16": torchvision.models.vit_b_16(
        weights=torchvision.models.ViT_B_16_Weights.DEFAULT
    ),
    "vit_b_32": torchvision.models.vit_b_32(
        weights=torchvision.models.ViT_B_32_Weights.DEFAULT
    ),
    "vit_l_16": torchvision.models.vit_l_16(
        weights=torchvision.models.ViT_L_16_Weights.DEFAULT
    ),
    "vit_l_32": torchvision.models.vit_l_32(
        weights=torchvision.models.ViT_L_32_Weights.DEFAULT
    ),
    "vit_h_14": torchvision.models.vit_h_14(
        weights=torchvision.models.ViT_H_14_Weights.DEFAULT
    ),
    # Swin Transformers
    "swin_t": torchvision.models.swin_t(
        weights=torchvision.models.Swin_T_Weights.DEFAULT
    ),
    "swin_s": torchvision.models.swin_s(
        weights=torchvision.models.Swin_S_Weights.DEFAULT
    ),
    "swin_b": torchvision.models.swin_b(
        weights=torchvision.models.Swin_B_Weights.DEFAULT
    ),
    "swin_v2_t": torchvision.models.swin_v2_t(
        weights=torchvision.models.Swin_V2_T_Weights.DEFAULT
    ),
    "swin_v2_s": torchvision.models.swin_v2_s(
        weights=torchvision.models.Swin_V2_S_Weights.DEFAULT
    ),
    "swin_v2_b": torchvision.models.swin_v2_b(
        weights=torchvision.models.Swin_V2_B_Weights.DEFAULT
    ),
    # MaxVit
    "maxvit_t": torchvision.models.maxvit_t(
        weights=torchvision.models.MaxVit_T_Weights.DEFAULT
    ),
    # TODO: Label these
    "detection.fasterrcnn_resnet50_fpn": torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    ),
    "detection.fasterrcnn_resnet50_fpn_v2": torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
        weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    ),
    "detection.fasterrcnn_mobilenet_v3_large_fpn": torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
        weights=torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    ),
    "detection.fasterrcnn_mobilenet_v3_large_320_fpn": torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(
        weights=torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
    ),
    "detection.fcos_resnet50_fpn": torchvision.models.detection.fcos_resnet50_fpn(
        weights=torchvision.models.detection.FCOS_ResNet50_FPN_Weights.DEFAULT
    ),
    "detection.keypointrcnn_resnet50_fpn": torchvision.models.detection.keypointrcnn_resnet50_fpn(
        weights=torchvision.models.detection.KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
    ),
    "detection.maskrcnn_resnet50_fpn": torchvision.models.detection.maskrcnn_resnet50_fpn(
        weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    ),
    "detection.maskrcnn_resnet50_fpn_v2": torchvision.models.detection.maskrcnn_resnet50_fpn_v2(
        weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    ),
    "detection.retinanet_resnet50_fpn": torchvision.models.detection.retinanet_resnet50_fpn(
        weights=torchvision.models.detection.RetinaNet_ResNet50_FPN_Weights.DEFAULT
    ),
    "detection.retinanet_resnet50_fpn_v2": torchvision.models.detection.retinanet_resnet50_fpn_v2(
        weights=torchvision.models.detection.RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
    ),
    "detection.ssd300_vgg16": torchvision.models.detection.ssd300_vgg16(
        weights=torchvision.models.detection.SSD300_VGG16_Weights.DEFAULT
    ),
    "detection.ssdlite320_mobilenet_v3_large": torchvision.models.detection.ssdlite320_mobilenet_v3_large(
        weights=torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    ),
    "optical_flow.raft_large": torchvision.models.optical_flow.raft_large(
        weights=torchvision.models.optical_flow.Raft_Large_Weights.DEFAULT
    ),
    "optical_flow.raft_small": torchvision.models.optical_flow.raft_small(
        weights=torchvision.models.optical_flow.Raft_Small_Weights.DEFAULT
    ),
    "quantization.googlenet": torchvision.models.quantization.googlenet(
        weights=torchvision.models.quantization.GoogLeNet_QuantizedWeights.DEFAULT
    ),
    "quantization.inception_v3": torchvision.models.quantization.inception_v3(
        weights=torchvision.models.quantization.Inception_V3_QuantizedWeights.DEFAULT
    ),
    "quantization.mobilenet_v2": torchvision.models.quantization.mobilenet_v2(
        weights=torchvision.models.quantization.MobileNet_V2_QuantizedWeights.DEFAULT
    ),
    "quantization.mobilenet_v3_large": torchvision.models.quantization.mobilenet_v3_large(
        weights=torchvision.models.quantization.MobileNet_V3_Large_QuantizedWeights.DEFAULT
    ),
    "quantization.resnet18": torchvision.models.quantization.resnet18(
        weights=torchvision.models.quantization.ResNet18_QuantizedWeights.DEFAULT
    ),
    "quantization.resnet50": torchvision.models.quantization.resnet50(
        weights=torchvision.models.quantization.ResNet50_QuantizedWeights.DEFAULT
    ),
    "quantization.resnext101_32x8d": torchvision.models.quantization.resnext101_32x8d(
        weights=torchvision.models.quantization.ResNeXt101_32X8D_QuantizedWeights.DEFAULT
    ),
    "quantization.resnext101_64x4d": torchvision.models.quantization.resnext101_64x4d(
        weights=torchvision.models.quantization.ResNeXt101_64X4D_QuantizedWeights.DEFAULT
    ),
    "quantization.shufflenet_v2_x0_5": torchvision.models.quantization.shufflenet_v2_x0_5(
        weights=torchvision.models.quantization.ShuffleNet_V2_X0_5_QuantizedWeights.DEFAULT
    ),
    "quantization.shufflenet_v2_x1_0": torchvision.models.quantization.shufflenet_v2_x1_0(
        weights=torchvision.models.quantization.ShuffleNet_V2_X1_0_QuantizedWeights.DEFAULT
    ),
    "quantization.shufflenet_v2_x1_5": torchvision.models.quantization.shufflenet_v2_x1_5(
        weights=torchvision.models.quantization.ShuffleNet_V2_X1_5_QuantizedWeights.DEFAULT
    ),
    "quantization.shufflenet_v2_x2_0": torchvision.models.quantization.shufflenet_v2_x2_0(
        weights=torchvision.models.quantization.ShuffleNet_V2_X2_0_QuantizedWeights.DEFAULT
    ),
    "segmentation.deeplabv3_mobilenet_v3_large": torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(
        weights=torchvision.models.segmentation.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
    ),
    "segmentation.deeplabv3_resnet50": torchvision.models.segmentation.deeplabv3_resnet50(
        weights=torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT
    ),
    "segmentation.deeplabv3_resnet101": torchvision.models.segmentation.deeplabv3_resnet101(
        weights=torchvision.models.segmentation.DeepLabV3_ResNet101_Weights.DEFAULT
    ),
    "segmentation.fcn_resnet50": torchvision.models.segmentation.fcn_resnet50(
        weights=torchvision.models.segmentation.FCN_ResNet50_Weights.DEFAULT
    ),
    "segmentation.fcn_resnet101": torchvision.models.segmentation.fcn_resnet101(
        weights=torchvision.models.segmentation.FCN_ResNet101_Weights.DEFAULT
    ),
    "segmentation.lraspp_mobilenet_v3_large": torchvision.models.segmentation.lraspp_mobilenet_v3_large(
        weights=torchvision.models.segmentation.FCN_ResNet101_Weights.DEFAULT
    ),
    "video.mvit_v1_b": torchvision.models.video.mvit_v1_b(
        weights=torchvision.models.video.MViT_V1_B_Weights.DEFAULT
    ),
    "video.mvit_v2_s": torchvision.models.video.mvit_v2_s(
        weights=torchvision.models.video.MViT_V2_S_Weights.DEFAULT
    ),
    "video.r3d_18": torchvision.models.video.r3d_18(
        weights=torchvision.models.video.R3D_18_Weights.DEFAULT
    ),
    "video.mc3_18": torchvision.models.video.mc3_18(
        weights=torchvision.models.video.MC3_18_Weights.DEFAULT
    ),
    "video.r2plus1d_18": torchvision.models.video.r2plus1d_18(
        weights=torchvision.models.video.R2Plus1D_18_Weights.DEFAULT
    ),
    "video.s3d": torchvision.models.video.s3d(
        weights=torchvision.models.video.S3D_Weights.DEFAULT
    ),
    "video.swin3d_t": torchvision.models.video.swin3d_t(
        weights=torchvision.models.video.Swin3D_T_Weights.DEFAULT
    ),
    "video.swin3d_s": torchvision.models.video.swin3d_s(
        weights=torchvision.models.video.Swin3D_S_Weights.DEFAULT
    ),
    "video.swin3d_b": torchvision.models.video.swin3d_b(
        weights=torchvision.models.video.Swin3D_B_Weights.DEFAULT
    ),
}


def main():
    extract_and_print_all()


def extract_and_print_all():
    for model_name in models.keys():
        extract_and_print(model_name)


def extract_and_print(model_name: str):
    model = models[model_name]
    sample_input = (torch.randn(4, 3, 224, 224),)
    exported = torch_export(model, sample_input)
    stablehlo_program = exported_program_to_stablehlo(exported)
    print(stablehlo_program.get_stablehlo_text("forward"))


if __name__ == "__main__":
    main()
