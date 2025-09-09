# Torch
import torch

# import torchvision
from torch.export import export as torch_export
from torch_xla.stablehlo import exported_program_to_stablehlo

# Jax
import jax
from jax import export as jax_export
import jax.numpy as jnp
import numpy as np
from jax._src.interpreters import mlir as jax_mlir
from jax._src.lib.mlir import ir


def main():
    # disabled because it requires torchvision
    # torch_example()
    jax_example()


def jax_example():
    # Returns prettyprint of StableHLO module without large constants
    def get_stablehlo_asm(module_str):
        with jax_mlir.make_ir_context():
            stablehlo_module = ir.Module.parse(
                module_str, context=jax_mlir.make_ir_context()
            )
            return stablehlo_module.operation.get_asm(large_elements_limit=20)

    # Disable logging for better tutorial rendering
    import logging

    logging.disable(logging.WARNING)

    # Create a JIT-transformed function
    @jax.jit
    def plus(x, y):
        return jnp.add(x, y)

    # Create abstract input shapes
    inputs = (
        np.int32(1),
        np.int32(1),
    )
    input_shapes = [jax.ShapeDtypeStruct(input.shape, input.dtype) for input in inputs]

    # Export the function to StableHLO
    stablehlo_add = jax_export.export(plus)(*input_shapes).mlir_module()
    print(get_stablehlo_asm(stablehlo_add))


# def torch_example():
#     resnet18 = torchvision.models.resnet18(
#         weights=torchvision.models.ResNet18_Weights.DEFAULT
#     )
#     sample_input = (torch.randn(4, 3, 224, 224),)
#     exported = torch_export(resnet18, sample_input)
#     stablehlo_program = exported_program_to_stablehlo(exported)
#     print(stablehlo_program.get_stablehlo_text("forward")[0:4000], "\n...")


if __name__ == "__main__":
    main()
