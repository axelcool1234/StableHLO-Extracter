#!/usr/bin/env bash

# DESC: A shell script to run stablehlo-opt with a configurable pass pipeline.

# --- Configuration ---
# DESC: Define the input and output filenames and the `stablehlo-opt` path.
INPUT_FILE=""
OUTPUT_FILE="model_optimized.txt"
STABLEHLO_OPT_PATH=$(which stablehlo-opt)

# --- Pass Pipelines ---
# DESC: Define different pass pipelines as variables for easy swapping.

# Pipeline 1: StableHLO-Only Optimizations
# Focuses on high-level graph transformations without lowering to other dialects.
PIPELINE_STABLEHLO_ONLY="builtin.module(inline, stablehlo-refine-shapes, func.func(stablehlo-aggressive-simplification, stablehlo-aggressive-folder, canonicalize, cse), symbol-dce)"

# Pipeline 2: Lowering to Linalg and Loops for Code Generation
# This is a common path for preparing a model for a specific backend.
PIPELINE_LOWERING="builtin.module(stablehlo-legalize-to-linalg, func.func(linalg-generalize-named-ops, linalg-fuse-elementwise-ops, linalg-tiling, convert-linalg-to-loops), canonicalize, cse)"

# --- Main Script ---
# DESC: Runs `stablehlo-opt` with the chosen pipeline
CHOSEN_PIPELINE="${PIPELINE_STABLEHLO_ONLY}"
# CHOSEN_PIPELINE="${PIPELINE_LOWERING}"


# Check if the binary was found.
if [ -z "$STABLEHLO_OPT_PATH" ]; then
    echo "Error: stablehlo-opt not found in your PATH."
    echo "Please ensure the directory containing the binary is in your PATH."
    exit 1
fi

echo "Found stablehlo-opt at: $STABLEHLO_OPT_PATH"
echo "Running stablehlo-opt with the following pipeline:"
echo "$CHOSEN_PIPELINE"
echo "Input file: $INPUT_FILE"
echo "Output file: $OUTPUT_FILE"

# Execute the command.
"$STABLEHLO_OPT_PATH" --pass-pipeline="$CHOSEN_PIPELINE" "$INPUT_FILE" -o "$OUTPUT_FILE"

# Check the exit status of the previous command.
if [ $? -eq 0 ]; then
    echo "Success: Optimization complete. Output saved to $OUTPUT_FILE"
    # Show the differences between the input and output files.
    echo "--- Showing differences between $INPUT_FILE and $OUTPUT_FILE ---"
    diff --color "$INPUT_FILE" "$OUTPUT_FILE"
else
    echo "Error: stablehlo-opt failed."
fi