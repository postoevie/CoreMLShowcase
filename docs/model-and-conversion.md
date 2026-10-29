# Model and conversion

## What the model does

It turns one spoken Russian phrase (already transcribed to text) into a structured meal: which dishes, how much of each, and optionally the meal title, the date and the time. It is a token classifier (NER), so it tags words and does not generate text.

Input:

```
вчера на ужин курица 150 грамм и гречка 80 грамм
(yesterday for dinner chicken 150 grams and buckwheat 80 grams)
```

Word-level tags (BIO, 15 labels: `O` plus `B-`/`I-` for `MEAL_TITLE`, `MEAL_TYPE`, `FOOD_NAME`, `QUANTITY`, `UNIT`, `DATE`, `TIME`):

| вчера | на | ужин | курица | 150 | грамм | и | гречка | 80 | грамм |
|---|---|---|---|---|---|---|---|---|---|
| B-DATE | O | B-MEAL_TITLE | B-FOOD_NAME | B-QUANTITY | B-UNIT | O | B-FOOD_NAME | B-QUANTITY | B-UNIT |

Result of `parseMeal` (this phrase is a test in `Packages/MealRecognize/Tests`):

```
title:    "ужин"
dateText: "вчера"
timeText: nil
dishes:   [ foodName "курица", quantity 150, unit "грамм" ,
            foodName "гречка", quantity 80,  unit "грамм" ]
```

Each spoken dish becomes its own entry; a quantity and a unit are attached to the right dish whether they come before or after the food name (`150 грамм курицы 80 грамм гречки в семь вечера` works too, with `timeText: "семь вечера"`).

The package returns the recognized text spans; turning "вчера" into a date or "сто пятьдесят" into 150 is the app's job (the package accepts a `WordNumberParsing` implementation for number words).

## The model

- **Base model:** `cointegrated/rubert-tiny2` (a small Russian BERT with hidden size 312; most of its weights are in the embedding table), fine-tuned for token classification. Sequence length is fixed at 128, batch size 1.
- **Labelling:** only the first subword of each word carries a label (the others are ignored in the loss, `-100`). The Swift decoder follows the same rule: it reads the label at the first subword and lets `##` continuation pieces join the active span.
- **Training data:** 6454 labelled phrases, mostly generated from templates (so BIO tags are exact) plus a few hundred phrases written by an LLM with inline tags. Foods are split between train and validation/test, so validation and test contain foods never seen in training. The data generator is not included here because it depends on the private app's food catalog.
- **Fine-tuning:** Google Colab, 10 epochs, learning rate 5e-5, seed 42 (`tools/finetune.py`, run in `notebooks/rubert-tiny2.ipynb`).
- **Why this model:** a compact on-device model (54 MB in FP16). The app already had the task and the groundwork (data, training script, conversion, Swift inference, tests).

## Conversion

`tools/convert_to_coreml.py` traces the model with `torch.jit.trace` and converts it with `coremltools` to an ML Program (`.mlpackage`, iOS 16 target): inputs `input_ids` and `attention_mask` as `int32` `[1, 128]`, output `logits` `[1, 128, num_labels]`. `position_ids` and `token_type_ids` are fixed buffers inside a wrapper module, because their dynamic creation breaks the tracer. `--precision fp16|fp32` selects the precision. After saving, the script runs a parity check (argmax of PyTorch against Core ML) that should print `1.000`. `tools/check_parity.py` is a separate check of an existing `.mlpackage` against the checkpoint.

Parity results: Core ML against PyTorch gave identical entity spans on 200 of 200 test samples; FP16 against FP32 gave 0 differing word labels out of 4522 on 640 test samples (largest logit difference 0.016).

Reproduced on 2026-10-28 in a clean Python 3.11 environment from `tools/requirements.txt` (torch 2.7.0, transformers 4.57.6, coremltools 9.0, macOS): converting the checkpoint to FP16 and running `tools/check_parity.py` on its 8 built-in phrases gave argmax agreement 1.000, largest logit difference 0.0109. The 200- and 640-sample figures above come from the original test set, which is not in this repository.

Errors met along the way and their fixes are in [conversion-log.md](conversion-log.md).

## Weight compression

Two compressed variants of the FP16 package were made with `tools/compress_coreml.py` (`coremltools.optimize.coreml`, post-training, no retraining and no data). The source is the finished FP16 `.mlpackage`, not the checkpoint, so compression is a separate step after conversion. Compute stays in FP16: the compressed weights are expanded back to FP16 when the model runs, so only the stored size changes.

| Variant | Command | Package size |
|---|---|---|
| FP16 (source) | | 57.1 MB |
| INT8 | `--method linear --dtype int8` (symmetric, one scale per output channel) | 28.9 MB (50.5%) |
| Palettized, 6 bits | `--method palettize --nbits 6` (k-means lookup table) | 21.5 MB (37.6%) |

Both were checked with `tools/evaluate.py` on the 640-phrase test set (macOS, `CPU_ONLY`, coremltools 9.0, numpy 2.2.6, transformers 4.57.6, seqeval 1.2.2):

- Entity-level F1 is identical to FP16 for every entity type (micro F1 0.9993; see the table in the README).
- Against FP16: 0 of 640 phrases and 0 of 4522 word labels differ, for both variants.

Read this with care: the test set is made by the same generator as the training data and is easy, so "no difference" shows only that compression broke nothing on these phrases. The largest logit difference was not measured. The Python evaluation does not run the Swift pipeline.

Not tried: int4, per-block granularity and `per_grouped_channel` palettization (they need a newer deployment target than iOS 16). The compressed packages are in `CoreMLShowcaseApp/CoreMLShowcaseAppTests/CompressedModels/` and are used only by the performance tests; the Swift package ships FP16 only.

## Notes from the work

- **FP16 produced all-`O` output (NaN logits).** BERT builds its additive attention mask as `(1 - mask) * finfo(float32).min`. That value does not fit in FP16: it becomes `-inf`, then `0 * -inf = NaN`, and `argmax` over NaN is index 0, which is `O`. The fix is to replace `get_extended_attention_mask` with `(1 - mask) * -1e4` before tracing (in `tools/convert_to_coreml.py`). Always check the parity line after a conversion. Details in [conversion-log.md](conversion-log.md) (entries 11 and 12).
- **The iOS Simulator's GPU path returns zeros for this FP16 model.** With `.all` and `.cpuAndGPU` the logits are exactly 0.0 (no NaN), so every label is `O`. With `.cpuOnly` and `.cpuAndNeuralEngine` the output is correct. The reason is unknown. The package's default `computeUnits` is therefore `.cpuAndNeuralEngine` in the Simulator and `.all` on a device. With that, all package tests pass in the Simulator. The repository contains `SimulatorDiagnosticTests`, the small test that found this by printing the logits for each compute-unit setting.
- **Run performance tests on a real device only.** The Simulator has no Neural Engine and no real GPU, so its numbers mean nothing, and the model-dependent performance tests are skipped there. Tests in a Swift package cannot run on a device without a host app, which is why the performance tests live in the `CoreMLShowcaseApp` project.
- **FP32 and Git LFS.** FP32 (about 114 MB `weight.bin`) is over GitHub's 100 MB file limit; the repository ships FP16 only.
- **Not included:** the training data and its generator, and the private app.

## Tokenizer golden test

`WordPieceTokenizerGoldenTests` compares the Swift `WordPieceTokenizer` with the Hugging Face tokenizer of the checkpoint. `tools/make_golden_tokenizer.py` checks that the repo `vocab.txt` equals the checkpoint vocabulary, then stores `input_ids` and `attention_mask` (length 128) for 38 phrases in `Tests/MealRecognizeTests/Resources/golden_tokenizer.json`. The phrases cover digits, punctuation, mixed case, whitespace, an empty string, Latin words, emoji, rare words that fall back to `[UNK]`, and inputs longer than 128 pieces (truncation). The app lowercases Russian input before tokenizing, and the golden ids are computed the same way.

The first run found one difference: for `№5` Hugging Face gives one token, the Swift tokenizer two. The Swift code treated every Unicode symbol as punctuation, while BERT splits only ASCII non-alphanumeric characters and the Unicode `P*` categories. `isPunctuation` now follows BERT's rule, and the test passes. The golden file does not cover every Unicode edge case (for example Chinese characters or accent stripping), only the listed phrases.
