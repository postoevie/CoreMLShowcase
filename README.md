# CoreMLShowcase

An on-device ML pipeline for iOS: a fine-tuned Russian BERT token classifier, converted from PyTorch to Core ML, with tokenizer and decoding written in Swift, and measured on a real iPhone.

![Demo](docs/demo.gif)

## Highlights

- PyTorch to Core ML in FP16: package **109 MB to 54 MB**.
- **About 0.8 ms per inference** on iPhone 13 (`.cpuAndNeuralEngine`, FP16): 2.36 ms median for the three test phrases run in a row. FP32: 4.20 ms for three, about 1.4 ms each.
- **0 differing word labels out of 4522** between FP16 and FP32 (640 test samples).
- Core ML and PyTorch gave identical entity spans on 200 of 200 test samples.

The model turns a transcribed Russian phrase such as `вчера на ужин курица 150 грамм и гречка 80 грамм` into a title, date, time and a list of dishes with quantities and units. It is extracted from a private meal-tracker app. See [docs/model-and-conversion.md](docs/model-and-conversion.md).

### Input limits

- The input shape is fixed at `[1, 128]`; batch size is 1.
- Layout: `[CLS]`, up to 126 word pieces, `[SEP]`, then `[PAD]`. The attention mask keeps padding from affecting real tokens.
- Text beyond 126 word pieces is cut off silently and not recognized. This is deliberate: spoken meal phrases are short, and a hard failure would be worse than losing a tail. A word may take several word pieces, so the limit in words is lower.
- Empty or whitespace-only input throws `ParsingError.emptyTranscript`.
- The text is lower-cased before tokenizing, because the vocabulary is lower-case.
- The length is fixed at conversion time (`--seq_len`, default 128) and must match `ServiceCoreML.sequenceLength`. Changing it means reconverting the model.
- Wrong array lengths at the inference level throw an internal `InferenceError` instead of writing out of bounds.

## What this shows

- **Model conversion:** `torch.jit.trace` to an ML Program with `coremltools`, FP16 precision, and a parity check against PyTorch.
- **Pre/post-processing in Swift:** WordPiece tokenizer, BIO span decoding and dish grouping, with no Python at runtime. The tokenizer is checked against the Hugging Face tokenizer by a golden test (38 phrases, exact `input_ids` and `attention_mask`); the test found and fixed a punctuation-rule difference (`№5`).
- **Evaluation:** entity-level precision, recall and F1 (seqeval) of the converted Core ML model on a 640-phrase test set, FP16 and FP32 side by side (`tools/evaluate.py`).
- **Inference optimization:** FP16 vs FP32 and all four `computeUnits` settings measured on a device, with Core ML report and Instruments analysis.
- **Debugging:** an FP16 attention-mask overflow (NaN logits, all-`O` output) and a Simulator GPU path that returns zeros; see [docs/conversion-log.md](docs/conversion-log.md).

## Repository layout

| Path | What it is |
|---|---|
| `Packages/MealRecognize` | Swift package: tokenizer, Core ML inference, BIO decoding, dish grouping, model lifecycle. Contains the FP16 model (`MealNerRu.mlpackage`, 54 MB) and its vocabulary. |
| `CoreMLShowcaseApp/CoreMLShowcaseApp.xcodeproj` | Minimal host app with one demo screen. |
| `CoreMLShowcaseApp/CoreMLShowcaseAppTests` | Performance tests (`PerformanceTests`) that run on a real device. |
| `tools/convert_to_coreml.py` | PyTorch (Hugging Face checkpoint) to Core ML conversion with a parity check. |
| `tools/check_parity.py` | Checks a `.mlpackage` against the checkpoint. |
| `tools/make_golden_tokenizer.py` | Generates `golden_tokenizer.json` (expected token ids from the Hugging Face tokenizer) for the Swift tokenizer test. |
| `tools/evaluate.py`, `data/test.json` | Entity-level F1 evaluation of a `.mlpackage` on the 640-phrase test set. |
| `tools/finetune.py`, `notebooks/rubert-tiny2.ipynb` | Fine-tuning script and the Colab notebook. |
| `docs/` | [model-and-conversion](docs/model-and-conversion.md), [package-api](docs/package-api.md), [performance](docs/performance.md), [conversion-log](docs/conversion-log.md). |

## Quick start

```swift
import MealRecognize

let service = try ServiceCoreML(language: .russian)
try await service.prepare()          // optional: load and warm up the model
let meal = try await service.parseMeal(transcript: "вчера на ужин курица 150 грамм и гречка 80 грамм")
// meal.title == "ужин", meal.dateText == "вчера", two dishes with quantity and unit
```

Configuration and the rest of the API: [docs/package-api.md](docs/package-api.md).

Running:

- Open `CoreMLShowcaseApp/CoreMLShowcaseApp.xcodeproj`. The package is referenced locally (`../Packages/MealRecognize`).
- Before running on a device, set your own Team and bundle identifiers in both targets (`CoreMLShowcaseApp` and `CoreMLShowcaseAppTests`): the project ships with an empty Team and the placeholder `com.example.coremlshowcaseapp` (and `.tests`).
- Package tests (also in the Simulator): `cd Packages/MealRecognize && xcodebuild test -scheme MealRecognize -destination 'platform=iOS Simulator,name=<simulator>'`.
- Performance tests: run `CoreMLShowcaseAppTests/PerformanceTests` on a real device, Release configuration, `ENABLE_TESTABILITY=YES`.

Conversion (the Core ML parity check runs only on macOS):

```
pip install -r tools/requirements.txt
python tools/convert_to_coreml.py --model_dir <checkpoint> --output_path <out>.mlpackage --precision fp16
python tools/check_parity.py --model_dir <checkpoint> --mlpackage <out>.mlpackage   # exit 0 on match
python tools/evaluate.py --data data/test.json --vocab <vocab.txt> --model fp16=<out>.mlpackage   # F1 per entity (macOS)
```

Tokenizer golden file (only needed after a vocabulary change): `python tools/make_golden_tokenizer.py --model_dir <checkpoint>`, then run `WordPieceTokenizerGoldenTests` in the package.

## Accuracy

Entity-level scores of the converted FP16 model on `data/test.json` (640 phrases, 2684 entities), computed with `tools/evaluate.py` on macOS by running the `.mlpackage` itself. An entity counts as correct only when its span and its type both match (seqeval).

| Entity | Support | Precision | Recall | F1 FP16 | F1 INT8 | F1 palettized 6-bit |
|---|---|---|---|---|---|---|
| `DATE` | 182 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `FOOD_NAME` | 848 | 0.9988 | 0.9988 | 0.9988 | 0.9988 | 0.9988 |
| `MEAL_TITLE` | 173 | 0.9942 | 0.9942 | 0.9942 | 0.9942 | 0.9942 |
| `QUANTITY` | 692 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `TIME` | 174 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `UNIT` | 615 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **micro avg** | 2684 | 0.9993 | 0.9993 | **0.9993** | **0.9993** | **0.9993** |
| macro avg | 2684 | 0.9988 | 0.9988 | 0.9988 | 0.9988 | 0.9988 |

Precision and recall are given for FP16; the compressed variants have the same values. Word accuracy is 0.9998, and 639 of the 640 phrases have every word label correct. The FP32, INT8 and palettized packages give exactly the same numbers: against FP16, none of the 640 phrases and none of the 4522 word labels differ. How the compressed variants were made: [docs/model-and-conversion.md](docs/model-and-conversion.md#weight-compression).

How a phrase is scored: the label of a word is the model's prediction at its first subword (the model is trained that way and the Swift decoder reads it the same way); words are lower-cased and tokenized with the package's `vocab.txt`; a `MEAL_TYPE` prediction counts as `MEAL_TITLE` (the Swift package treats them the same, and the 13-label data has no `MEAL_TYPE`).

Read these numbers with care: the test set is made by the same generator as the training data (template phrases plus a few hundred phrases written by an LLM). The foods in it were not seen in training, but the shapes of the phrases were. A score this close to 1 shows the model learned the task; it is not a prediction of the accuracy on real speech. The scores measure the model, not the Swift pipeline (tokenizer, span decoding and dish grouping); those are covered by the package tests.

## Performance

iPhone 13, iOS 26.3.1, Release build, 100 iterations, model inference only. Each measured iteration runs three phrases in a row, so the times are per three inferences (divide by 3 for one); median / 95th percentile in ms:

| `computeUnits` | FP32 | FP16 |
|---|---|---|
| `.cpuAndNeuralEngine` | 4.20 / 4.37 | **2.36 / 2.63** |
| `.cpuOnly` | 4.30 / 4.35 | 2.51 / 2.56 |
| `.all` | 21.63 / 22.35 | 6.90 / 9.02 |
| `.cpuAndGPU` | 21.80 / 22.41 | 21.32 / 21.94 |

Package size 109 MB (FP32) / 54 MB (FP16); first load about 1 s in the test process. These numbers come from a custom performance test (XCTest, `measure` over 100 iterations), not from Apple's built-in Core ML Performance Report in Xcode; method in [docs/performance.md](docs/performance.md#method). Tokenizer 0.13 ms, span decoding 0.04 ms, dish grouping 0.02 ms. Method, CPU time, cold start and the GPU analysis: [docs/performance.md](docs/performance.md).

Compressed variants (post-training weight compression, [docs/model-and-conversion.md](docs/model-and-conversion.md#weight-compression)), measured in one run on the same device, same method, median / 95th percentile in ms per three inferences:

| | FP16 | INT8 | Palettized 6-bit |
|---|---|---|---|
| Package size | 57.1 MB | 28.9 MB | 21.5 MB |
| `.cpuAndNeuralEngine` | 2.35 / 2.56 | 2.24 / 2.40 | 1.97 / 2.23 |
| `.cpuOnly` | 2.50 / 2.57 | 2.50 / 2.55 | 2.51 / 2.55 |
| Micro F1 (test set) | 0.9993 | 0.9993 | 0.9993 |

Size halves (INT8) or drops to a third (6-bit) with no change in F1 on this test set. Speed on the CPU path is the same for all three, and on the Neural Engine path the differences are small (about 5% for INT8, 16% for 6-bit; one run, not repeated). Memory: after load the process resident size was about 108 MB (FP16), 124 MB (INT8) and 123 MB (6-bit), so the smaller file did not give a smaller footprint; these are single cold starts and are noisy (see [docs/performance.md](docs/performance.md)).

## Limitations & next steps

- The explanation of the slow GPU path is a reading of Instruments traces; a Metal System Trace has not been taken.
- Energy was not measured. Memory was measured only as the process resident size after load for FP16, INT8 and 6-bit (single cold starts, `.cpuAndNeuralEngine`), not per compute-unit setting.
- Weight compression was tried (INT8 and 6-bit palettization, see Performance). None of the other optimizations below was tried (see "Possible optimizations"). INT4, per-block and grouped-channel variants need a newer deployment target and were not made.
- Only Russian is supported.
- Known failure: the phrase `сегодня 6 утра каша овсяная 200 грамм` is labelled wrongly (checked in Python on macOS with the FP16 and INT8 packages, same result): `6` is tagged `QUANTITY` and `утра` `I-TIME` without a `B-TIME`, and `овсяная` starts a second `FOOD_NAME` instead of continuing `каша`. The cause was not investigated. The test set does not contain such phrases, so the F1 above does not show it.
- The training data and its generator are not included (they depend on the private app's food catalog).
- The iOS Simulator's GPU path returns zeros for this model; the package defaults to `.cpuAndNeuralEngine` there, and performance numbers need a real device.

### Possible optimizations (not done)

These were considered and deliberately left out. None was implemented or measured here; the effects below are expectations, not results. Each step would need the same loop as the FP16 conversion: a PyTorch/Core ML parity check and an entity-level F1 check on the test set, then a measurement on a real device.

The model is already small (FP16, about 54 MB, about 2 ms of inference on the Neural Engine on an iPhone 13). About 52 MB of the size is the embedding table (83,828 tokens by 312 values), and an average phrase uses 18 of the 128 input positions, so the levers are size and sequence length rather than the transformer layers (3 layers, 12 heads).

| Idea | Expected effect | Notes |
|---|---|---|
| **Weight compression with `coremltools.optimize.coreml`** (INT8 per-channel and 6-bit palettization done; int4, per-block and grouped-channel not): `linear_quantize_weights` (int8 or int4, per-channel or per-block) or `palettize_weights` (k-means lookup table, 4 to 6 bits, `per_grouped_channel`) | Size: roughly 27 MB with 8 bits and 14 MB with 4 bits, if the embedding table is compressed | Post-training and data-free; the model must be an ML Program (it is). Latency is expected to stay about the same, because the time is overhead, not memory bandwidth. Grouped-channel palettization is documented to keep accuracy better than scalar 4-bit. |
| **Vocabulary pruning** | Size and load time: keep only the tokens needed for food words and common Russian words (for example 20,000 to 30,000), which shrinks the table more than three times | The Swift WordPiece tokenizer reads `vocab.txt`, so re-indexing is simple. Risk: foods missing from the kept vocabulary split into shorter pieces, so every piece needed for the catalog's words must stay. |
| **Shorter sequence length** (32 or 64, or `EnumeratedShapes` for 32, 64 and 128) | Latency: attention cost grows with the square of the length, and about 86% of the 128 positions are padding for a typical phrase | Ties in with the input limit (126 word pieces); longer phrases would need the larger shape or truncation earlier. |
| **Embedding lookup outside the model** (a row lookup in Swift, the model takes `inputs_embeds`) | Removes the seven GPU operations at the start of the graph and the GPU request preparation seen in cold starts; avoids GPU and Neural Engine transfers in `.all` | Not needed with `.cpuAndNeuralEngine`, which already avoids the GPU. The table still has to be in memory. |
| **Activation quantization (W8A8)** with `linear_quantize_activations` | Possible speed-up on newer chips with int8 paths | The API is marked experimental in the documentation and needs calibration data. No benefit expected on an A15 (iPhone 13). |
| **ANE-oriented layout** (Apple's `ml-ane-transformers`: 1x1 convolutions instead of linear layers) and **distillation** into a smaller model | Small | 75 of 82 operations already run on the Neural Engine and the model has only 3 layers, so the gain would not justify the rewrite. |

Already done: FP16, INT8 and 6-bit palettized weights (measured, not shipped in the package), choosing `.cpuAndNeuralEngine` as the default compute units, and a background `prepare()` that pays the load and first-call cost before the user needs it.

## License

MIT, see [LICENSE](LICENSE).
