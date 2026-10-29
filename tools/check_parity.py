"""
PyTorch vs Core ML parity check for the meal NER model.

    python tools/check_parity.py --model_dir X --mlpackage Y [--seq_len 128] [--samples phrases.txt]

--model_dir is a local HF checkpoint directory or a Hugging Face Hub id (it must have a token-classification head).
--samples is a text file with one phrase per line; built-in phrases are used by default.
Only attended (non-padding) positions are compared.

Exit codes: 0 = argmax agreement is exactly 1.000, 1 = mismatch, 2 = cannot run (not macOS / predict unavailable).
"""

import argparse
import platform
import sys

DEFAULT_PHRASES = [
    "на завтрак я съел два яйца и тост с маслом",
    "на обед был борщ со сметаной и кусок чёрного хлеба",
    "выпил стакан молока и съел банан",
    "ужин: гречка с курицей и салат из огурцов и помидоров",
    "200 грамм творога с мёдом",
    "перекусил яблоком и горстью орехов",
    "пицца маргарита и кола",
    "вчера вечером ела пасту с креветками",
]


def check_parity(model_dir, mlpackage, seq_len=128, samples=None, model=None, tokenizer=None):
    """Compare PyTorch and Core ML predictions. Returns (agreement, max_abs_logit_diff).

    Raises RuntimeError if Core ML prediction is unavailable. Pass `model`/`tokenizer` to reuse loaded objects.
    """
    if platform.system() != "Darwin":
        raise RuntimeError("Core ML predict() only runs on macOS")

    import numpy as np
    import torch
    import coremltools as ct
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    if model is None:
        model = AutoModelForTokenClassification.from_pretrained(model_dir, attn_implementation="eager")
    model.eval()
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)

    try:
        mlmodel = ct.models.MLModel(mlpackage)
    except Exception as error:
        raise RuntimeError(f"cannot load {mlpackage} for prediction: {error}") from error

    enc = tokenizer(
        samples or DEFAULT_PHRASES,
        padding="max_length",
        truncation=True,
        max_length=seq_len,
        return_tensors="pt",
    )
    matches, total, max_diff = 0, 0, 0.0
    position_ids = torch.arange(seq_len).unsqueeze(0)
    for ids, mask in zip(enc["input_ids"], enc["attention_mask"]):
        ids, mask = ids.unsqueeze(0), mask.unsqueeze(0)
        with torch.no_grad():
            ref = model(
                input_ids=ids,
                attention_mask=mask,
                position_ids=position_ids,
                token_type_ids=torch.zeros_like(ids),
            ).logits.numpy()
        try:
            out = mlmodel.predict(
                {"input_ids": ids.numpy().astype(np.int32), "attention_mask": mask.numpy().astype(np.int32)}
            )["logits"]
        except Exception as error:
            raise RuntimeError(f"Core ML predict() unavailable: {error}") from error
        keep = mask[0].numpy().astype(bool)
        ref, out = ref[0][keep], np.asarray(out)[0][keep]
        matches += int((ref.argmax(-1) == out.argmax(-1)).sum())
        total += int(keep.sum())
        max_diff = max(max_diff, float(np.abs(ref - out).max()))
    return matches / total, max_diff


def main(argv=None):
    parser = argparse.ArgumentParser(description="PyTorch vs Core ML parity check.")
    parser.add_argument("--model_dir", required=True, help="HF checkpoint dir or Hub id (token-classification head)")
    parser.add_argument("--mlpackage", required=True, help="path to the .mlpackage")
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--samples", help="text file, one phrase per line (default: built-in phrases)")
    args = parser.parse_args(argv)

    samples = None
    if args.samples:
        with open(args.samples, encoding="utf-8") as f:
            samples = [line.strip() for line in f if line.strip()]
        if not samples:
            parser.error("--samples file has no non-empty lines")

    try:
        agreement, max_diff = check_parity(args.model_dir, args.mlpackage, args.seq_len, samples)
    except RuntimeError as error:
        print(f"Parity check cannot run: {error}", file=sys.stderr)
        return 2
    print(f"Parity (argmax agreement, PyTorch vs Core ML): {agreement:.3f}")
    print(f"Max |logit diff|: {max_diff:.4f}")
    return 0 if agreement == 1.0 else 1


if __name__ == "__main__":
    sys.exit(main())
