"""
Convert fine-tuned RuBERT NER model (HF checkpoint) to CoreML (.mlpackage) via torch tracing. No ONNX.

Colab:
    !pip install -q "torch" "transformers" sentencepiece "coremltools>=8.0"
    !python convert_to_coreml.py [--precision fp16|fp32]

--model_dir may be a local fine-tuned checkpoint directory or a Hugging Face Hub id; both go straight to
from_pretrained (no local path handling). The checkpoint must have a token-classification head: a plain base model
such as cointegrated/rubert-tiny2 loads with a randomly initialised head, so use it only for smoke tests.

After saving, the package is checked against PyTorch with check_parity.py (macOS only; use --skip_parity to skip).

Output contract (matches Packages/MealRecognize ServiceCoreML.swift):
    inputs : input_ids [1, SEQ_LEN] int32, attention_mask [1, SEQ_LEN] int32
    output : logits    [1, SEQ_LEN, num_labels]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import coremltools as ct
from transformers import AutoConfig, AutoModelForTokenClassification

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_parity import check_parity  # noqa: E402

MODEL_DIR = "/content/drive/MyDrive/meal-ner-model"
OUTPUT_PATH = "/content/drive/MyDrive/meal-ner-model/MealNerRu.mlpackage"

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", default=MODEL_DIR)
parser.add_argument("--output_path", default=OUTPUT_PATH)
parser.add_argument("--seq_len", type=int, default=128)
parser.add_argument(
    "--precision",
    choices=["fp16", "fp32"],
    default="fp16",
    help="Weights/compute precision of the .mlpackage. fp16 is half the size (about 54 MB vs 109 MB) and runs on the "
    "Neural Engine; fp32 is for comparison only.",
)
parser.add_argument("--skip_parity", action="store_true", help="skip the PyTorch vs Core ML parity check")
args = parser.parse_args()

SEQ_LEN = args.seq_len
COMPUTE_PRECISION = ct.precision.FLOAT16 if args.precision == "fp16" else ct.precision.FLOAT32

print(f"Loading model from {args.model_dir}...")
config = AutoConfig.from_pretrained(args.model_dir)
model = AutoModelForTokenClassification.from_pretrained(
    args.model_dir,
    attn_implementation="eager",
)
model.eval()


def _fp16_safe_extended_mask(attention_mask, input_shape, device=None, dtype=None):
    """Additive mask with -1e4 instead of finfo(float32).min.

    -3.4e38 does not fit in FP16: it becomes -inf, then 0 * -inf = NaN, and every logit is NaN
    (argmax -> label 0 -> all `O`). -1e4 is already a large negative value for softmax.
    """
    return (1.0 - attention_mask[:, None, None, :].float()) * -1e4


model.bert.get_extended_attention_mask = _fp16_safe_extended_mask


class WrappedModel(torch.nn.Module):
    """Fixes position_ids / token_type_ids so tracing has no dynamic int ops."""

    def __init__(self, model, seq_len):
        super().__init__()
        self.model = model
        self.register_buffer(
            "position_ids",
            torch.arange(seq_len, dtype=torch.long).unsqueeze(0),
        )

    def forward(self, input_ids, attention_mask):
        position_ids = self.position_ids.expand(input_ids.shape[0], -1)
        token_type_ids = torch.zeros_like(input_ids)
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        ).logits


wrapped = WrappedModel(model, SEQ_LEN).eval()

print("Tracing model...")
example_ids = torch.zeros((1, SEQ_LEN), dtype=torch.long)
example_mask = torch.ones((1, SEQ_LEN), dtype=torch.long)

with torch.no_grad():
    traced = torch.jit.trace(wrapped, (example_ids, example_mask), strict=False)

print("Converting to CoreML...")
mlmodel = ct.convert(
    traced,
    convert_to="mlprogram",
    inputs=[
        ct.TensorType(name="input_ids", shape=(1, SEQ_LEN), dtype=np.int32),
        ct.TensorType(name="attention_mask", shape=(1, SEQ_LEN), dtype=np.int32),
    ],
    outputs=[ct.TensorType(name="logits")],
    minimum_deployment_target=ct.target.iOS16,
    # FLOAT16 needs the -1e4 mask patch above; without it the logits are NaN (all-`O` output, parity 0.000).
    # The patch is applied for both precisions so that fp16 and fp32 packages differ only in precision.
    compute_precision=COMPUTE_PRECISION,
)

mlmodel.user_defined_metadata["id2label"] = str(config.id2label)
mlmodel.user_defined_metadata["label2id"] = str(config.label2id)
mlmodel.user_defined_metadata["seq_len"] = str(SEQ_LEN)

mlmodel.save(args.output_path)
print(f"Saved: {args.output_path}")
print(f"Labels ({len(config.id2label)}): {list(config.id2label.values())}")

if not args.skip_parity:
    # Core ML predict() runs only on macOS; check_parity raises RuntimeError elsewhere (e.g. Colab).
    try:
        agreement, max_diff = check_parity(args.model_dir, args.output_path, SEQ_LEN, model=model)
        print(f"Parity (argmax agreement, PyTorch vs Core ML): {agreement:.3f}, max |logit diff| {max_diff:.4f}")
    except Exception as error:
        print(f"Parity check skipped: {error}")
