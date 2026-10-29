"""
Entity-level evaluation (precision / recall / F1 per entity type, seqeval) of a Core ML NER package on the test set.

It runs the .mlpackage itself (macOS only: Core ML `predict()` is not available on Linux), so it measures the converted
model, not the PyTorch checkpoint. Several packages can be compared in one run (for example FP16 and FP32).

How a phrase is scored:
  - the words of `tokens` are lower-cased (the app lower-cases the transcript, the vocabulary is lower-case) and split
    into WordPiece tokens with the same vocabulary the Swift package uses (`vocab.txt`);
  - the label of a word is the model's argmax at the first subword of the word (the model is trained that way, and the
    Swift decoder reads the labels the same way);
  - a `MEAL_TYPE` prediction counts as `MEAL_TITLE` (the Swift package treats them the same; the 13-label training data
    has no `MEAL_TYPE`);
  - the word-level BIO labels are compared with the gold labels with seqeval (an entity is correct only when its span and
    its type both match).

Usage:
    python tools/evaluate.py \\
        --data data/test.json \\
        --vocab Packages/MealRecognize/Sources/MealRecognize/Russian/Resources/vocab.txt \\
        --model fp16=Packages/MealRecognize/Sources/MealRecognize/Russian/Resources/MealNerRu.mlpackage \\
        --model fp32=path/to/MealNerRu_fp32.mlpackage

Requirements: coremltools, numpy, transformers, seqeval (macOS).
"""

import argparse
import ast
import json
import sys
from pathlib import Path

import coremltools as ct
import numpy as np
from seqeval.metrics import accuracy_score, classification_report
from transformers import BertTokenizerFast

SEQ_LEN = 128


def load_samples(path):
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def make_tokenizer(vocab_path):
    # Words are lower-cased by hand (as the Swift package does), so the tokenizer must not lower-case or strip accents.
    return BertTokenizerFast(vocab_file=str(vocab_path), do_lower_case=False, strip_accents=False)


def load_model(path):
    model = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_ONLY)
    id2label = ast.literal_eval(model.user_defined_metadata["id2label"])
    return model, {int(key): value for key, value in id2label.items()}


def normalize(label):
    return label.replace("MEAL_TYPE", "MEAL_TITLE")


def predict_words(model, id2label, tokenizer, words):
    encoding = tokenizer(
        [word.lower() for word in words],
        is_split_into_words=True,
        truncation=True,
        max_length=SEQ_LEN,
        padding="max_length",
        return_tensors="np",
    )
    inputs = {
        "input_ids": encoding["input_ids"].astype(np.int32),
        "attention_mask": encoding["attention_mask"].astype(np.int32),
    }
    logits = model.predict(inputs)["logits"]
    label_ids = np.asarray(logits)[0].argmax(axis=-1)

    labels = ["O"] * len(words)
    seen = set()
    for position, word_id in enumerate(encoding.word_ids()):
        if word_id is None or word_id in seen:
            continue
        seen.add(word_id)
        labels[word_id] = normalize(id2label[int(label_ids[position])])
    return labels


def evaluate(name, model_path, samples, tokenizer):
    model, id2label = load_model(model_path)
    gold, predicted = [], []
    for sample in samples:
        gold.append([normalize(label) for label in sample["labels"]])
        predicted.append(predict_words(model, id2label, tokenizer, sample["tokens"]))

    report = classification_report(gold, predicted, digits=4, output_dict=True, zero_division=0)
    exact = sum(1 for g, p in zip(gold, predicted) if g == p) / len(samples)
    return {
        "name": name,
        "report": report,
        "word_accuracy": accuracy_score(gold, predicted),
        "exact_match": exact,
        "predicted": predicted,
        "gold": gold,
    }


def print_result(result, count):
    print(f"\n=== {result['name']} ({count} samples) ===")
    print(f"{'entity':<14}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>9}")
    for entity, scores in result["report"].items():
        print(
            f"{entity:<14}{scores['precision']:>10.4f}{scores['recall']:>10.4f}"
            f"{scores['f1-score']:>10.4f}{int(scores['support']):>9}"
        )
    print(f"word accuracy: {result['word_accuracy']:.4f}")
    print(f"phrases with every word label correct: {result['exact_match']:.4f}")


def print_markdown(results):
    entities = [key for key in results[0]["report"] if key not in ("micro avg", "macro avg", "weighted avg")]
    names = [result["name"] for result in results]
    print("\nMarkdown (F1 per entity):")
    print("| Entity | Support | " + " | ".join(f"F1 {name}" for name in names) + " |")
    print("|---|---|" + "---|" * len(names))
    for entity in entities + ["micro avg", "macro avg"]:
        support = int(results[0]["report"][entity]["support"])
        cells = " | ".join(f"{result['report'][entity]['f1-score']:.4f}" for result in results)
        print(f"| {entity} | {support} | {cells} |")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="test.json: a list of {tokens, labels} items")
    parser.add_argument("--vocab", required=True, help="vocab.txt used by the Swift package")
    parser.add_argument("--model", action="append", required=True, help="name=path/to/Model.mlpackage (repeatable)")
    parser.add_argument("--json", help="write the full results (reports and word labels) to this file")
    args = parser.parse_args()

    samples = load_samples(args.data)
    tokenizer = make_tokenizer(args.vocab)

    results = []
    for spec in args.model:
        name, _, path = spec.partition("=")
        if not path or not Path(path).exists():
            sys.exit(f"--model expects name=path, got: {spec}")
        result = evaluate(name, path, samples, tokenizer)
        print_result(result, len(samples))
        results.append(result)

    base = results[0]
    total_words = sum(len(labels) for labels in base["predicted"])
    for other in results[1:]:
        phrases = sum(1 for a, b in zip(base["predicted"], other["predicted"]) if a != b)
        words = sum(x != y for a, b in zip(base["predicted"], other["predicted"]) for x, y in zip(a, b))
        print(
            f"\n{other['name']} against {base['name']}: {phrases} of {len(samples)} phrases and "
            f"{words} of {total_words} word labels differ"
        )
    print_markdown(results)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as file:
            json.dump(
                [{key: value for key, value in result.items() if key not in ("predicted", "gold")} for result in results],
                file,
                ensure_ascii=False,
                indent=2,
                default=lambda value: value.item() if hasattr(value, "item") else str(value),
            )


if __name__ == "__main__":
    main()
