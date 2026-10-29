"""
Post-training weight compression of a Core ML ML Program (`coremltools.optimize.coreml`).

Takes a converted .mlpackage (for example the FP16 one) and writes a smaller one. No retraining and no data are needed.
Compute stays in FP16: the compressed weights are expanded back to FP16 when the model runs, so this changes the
storage size, not the arithmetic.

Methods:
  linear    -- linear quantization to int8 or int4 (`--dtype`), one scale per output channel by default (`--granularity`).
  palettize -- k-means lookup table with `--nbits` bits per weight (a palette of 2**nbits values and an index per weight).

The accuracy and speed of the result are not assumed: check it with tools/evaluate.py (F1, label differences against
the uncompressed package) and measure it on a device.

Usage:
    python tools/compress_coreml.py --input MealNerRu.mlpackage --output MealNerRu_int8.mlpackage --method linear --dtype int8
    python tools/compress_coreml.py --input MealNerRu.mlpackage --output MealNerRu_pal6.mlpackage --method palettize --nbits 6

Requirements: coremltools (tools/requirements.txt). int4 and per-block granularity need a model with a newer minimum
deployment target than iOS16.
"""

import argparse
import os
import sys

import coremltools as ct
import coremltools.optimize.coreml as cto


def directory_size(path):
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            total += os.path.getsize(os.path.join(root, name))
    return total


def compress(model, args):
    if args.method == "linear":
        config = cto.OptimizationConfig(
            global_config=cto.OpLinearQuantizerConfig(
                mode="linear_symmetric",
                dtype=args.dtype,
                granularity=args.granularity,
                block_size=args.block_size,
            )
        )
        return cto.linear_quantize_weights(model, config)
    config = cto.OptimizationConfig(
        global_config=cto.OpPalettizerConfig(mode="kmeans", nbits=args.nbits)
    )
    return cto.palettize_weights(model, config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="source .mlpackage (ML Program)")
    parser.add_argument("--output", required=True, help="path of the compressed .mlpackage")
    parser.add_argument("--method", choices=["linear", "palettize"], required=True)
    parser.add_argument("--dtype", default="int8", help="linear: int8 or int4")
    parser.add_argument("--granularity", default="per_channel", help="linear: per_tensor, per_channel or per_block")
    parser.add_argument("--block_size", type=int, default=32, help="linear with per_block")
    parser.add_argument("--nbits", type=int, default=6, help="palettize: bits per weight")
    args = parser.parse_args()

    model = ct.models.MLModel(args.input, compute_units=ct.ComputeUnit.CPU_ONLY)
    compressed = compress(model, args)
    compressed.save(args.output)

    before, after = directory_size(args.input), directory_size(args.output)
    print(f"{args.input}: {before / 1e6:.1f} MB")
    print(f"{args.output}: {after / 1e6:.1f} MB ({after / before:.2%} of the source)")


if __name__ == "__main__":
    sys.exit(main())
