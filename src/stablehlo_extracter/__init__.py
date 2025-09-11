import torch
import torchvision
from torch.export import export as torch_export
from torch_xla.stablehlo import exported_program_to_stablehlo

models = {
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
    "swin_t": torchvision.models.swin_t(
        weights=torchvision.models.Swin_T_Weights.DEFAULT
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
