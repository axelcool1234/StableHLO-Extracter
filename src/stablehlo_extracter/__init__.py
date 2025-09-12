import torch
import torchvision
from torch.export import export as torch_export
from torch_xla.stablehlo import exported_program_to_stablehlo

models = {
    # AlexNet
    "alexnet": torchvision.models.alexnet(
        weights=torchvision.models.AlexNet_Weights.DEFAULT
    ),
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
