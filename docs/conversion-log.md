# Core ML conversion — error log

Errors and fixes encountered during `convert_to_coreml.py` development.

---

## 1. `No module named 'evaluate'`
**Cause:** `evaluate` is a separate HuggingFace package, not included in `transformers`.  
**Fix:** `pip install evaluate`

---

## 2. `TrainingArguments.__init__() got an unexpected keyword argument 'evaluation_strategy'`
**Cause:** Renamed in `transformers` 4.46+.  
**Fix:** Replace `evaluation_strategy="epoch"` → `eval_strategy="epoch"`

---

## 3. `Trainer unexpected keyword argument 'tokenizer'`
**Cause:** Renamed in `transformers` 4.46+.  
**Fix:** Replace `tokenizer=tokenizer` → `processing_class=tokenizer`

---

## 4. `TypeError: only 0-dimensional arrays can be converted to Python scalars` (coremltools torch frontend)
**Cause:** coremltools torch tracer can't handle the `int` op emitted by BERT's dynamic position ID creation.  
**Fix:** Wrap model to pass `position_ids` and `token_type_ids` as explicit fixed buffers, eliminating dynamic int casting during trace.

---

## 5. `ValueError: Couldn't instantiate the backend tokenizer` / `sentencepiece` missing
**Cause:** `rubert-tiny2` uses SentencePiece tokenizer; package not installed.  
**Fix:** `pip install sentencepiece`

---

## 6. `onnxscript` breaks ONNX opset 17 conversion
**Cause:** `onnxscript` intercepts the conversion pipeline and tries to downgrade opset 17 → 14; `LayerNormalization` has no opset 14 version.  
**Fix:** `pip uninstall onnxscript -y` — but torch 2.9+ requires `onnxscript` internally, making this path a dead end.

---

## 7. `AttributeError: module 'coremltools.converters' has no attribute 'onnx'`
**Cause:** `ct.converters.onnx.convert()` was removed in coremltools 7.x (existed only in 6.x).  
**Fix:** Use `ct.convert()` directly with the ONNX file path — but this also fails because coremltools 8.x dropped ONNX frontend entirely.

---

## 8. `np.issubclass_` removed in NumPy 2.0
**Cause:** coremltools 7.2 uses `np.issubclass_` which was removed in NumPy 2.0.  
**Fix:** `pip install "numpy<2.0"` — but this conflicts with other dependencies.

---

## 9. `coremltools 7.2` requires `protobuf<=4.0.0`; `onnx>=1.16` requires `protobuf>=4.25.1`
**Cause:** Incompatible protobuf version constraints between coremltools 7.2 and modern onnx.  
**Fix:** No fix — these two packages cannot coexist. Abandon ONNX path entirely.

---

## 10. `ValueError: Unable to determine the type of the model` (ONNX source not detected)
**Cause:** coremltools 8.x dropped ONNX support; `ct.convert()` no longer accepts ONNX models.  
**Fix:** Return to torch tracing path. Fix the `int` op error (see #4) by wrapping the model with explicit `position_ids` buffer. Use coremltools 8.x + torch 2.9.x + NumPy 2.x.

---

## 11. Converted CoreML model returns `O` for every token (FP16)
**Symptom:** The conversion finishes without errors, but the `.mlpackage` predicts label `0` (`O`) at every position for any input, so `parseDish` returns nil. PyTorch with the same checkpoint tags correctly (e.g. `курица 150 грамм` → `B-FOOD_NAME I-FOOD_NAME B-QUANTITY B-UNIT`). The parity check in `convert_to_coreml.py` printed `0.000`.  
**Environment:** local `scripts/.venv` (torch 2.4.0, coremltools 7.2), `compute_precision=ct.precision.FLOAT16`, fixed shape `[1, 128]`.  
**Cause:** FP16 precision. Re-running the same script with `FLOAT32` gives parity `1.000` on random ids. The exact FP16 failure point (attention scores / LayerNorm overflow) was not investigated.  
**Not the cause:** the all-zeros / all-ones trace example inputs and the fixed `position_ids` / `token_type_ids` in `WrappedModel` were suspected, but FP32 works with both unchanged.  
**Fix:** `convert_to_coreml.py` now uses `ct.precision.FLOAT32`. Cost: the weights are about 116 MB, versus about 57 MB in FP16.  
**Check after every conversion:** compare PyTorch and CoreML argmax (the script prints `Parity`; `predict()` runs only on macOS, so it is skipped on Colab). Expect `1.000`.  
**Side effect:** the `MealNerRu.mlpackage` bundled before this fix was FP16 and most likely hit the same problem.  
**Update:** the cause of the all-`O` FP16 output is found in entry 12 (attention mask), and FP16 turned out to behave differently in the iOS Simulator.

---

## 12. FP16 works on macOS but fails in the iOS Simulator tests

**Cause of entry 11 (found):** BERT builds the additive attention mask as `(1 - mask) * finfo(float32).min` (about -3.4e38). That value does not fit in FP16: it becomes `-inf`, then `0 * -inf = NaN`, and every logit is NaN. `argmax` over NaN returns index 0, which is `O` for every token.  
**Patch:** in `convert_to_coreml.py`, before tracing, replace `model.bert.get_extended_attention_mask` with `(1 - mask[:, None, None, :].float()) * -1e4`. With it, `compute_precision=ct.precision.FLOAT16` converts without NaN.

**What works (macOS, Python `coremltools`):**
- The FP16 package is about 54 MB (`weight.bin` about 57 MB), against about 109 MB for FP32.
- On 640 samples of `test.json` it matches FP32 exactly: 0 of 4522 word labels differ, 0 of 640 samples have different entity spans, no NaN in the logits, max logit difference 0.016.
- The result is the same on all compute units: `ALL`, `CPU_ONLY`, `CPU_AND_GPU`, `CPU_AND_NE`.

**What fails (iOS Simulator, `xcodebuild test`):** with the FP16 package bundled, 12 of 15 tests in `ServiceCoreMLTests` and `ServiceCoreMLMealTests` fail. `parseMeal` / `parseDish` return `nil` for every phrase, i.e. the model returns all `O` (or NaN) there. With the FP32 package the same tests pass.  
**On a real device (2026-10-27): the FP16 model works.** The on-device test succeeded (reported after running the app with the FP16 package). So the failure is specific to the Simulator, not to the model.

**Cause found (2026-10-28), a diagnostic test on the iPhone 17 simulator (iOS 26.3.1).** The logits were inspected per `MLComputeUnits` for the same phrase:

| `computeUnits` in the Simulator | FP16 logits | Result |
|---|---|---|
| `.cpuOnly` | min -2.37, max 7.80 | correct labels |
| `.cpuAndNeuralEngine` | same values | correct labels (there is no Neural Engine in the Simulator, the CPU computes) |
| `.all` | **all exactly 0.0** | every label `O` |
| `.cpuAndGPU` | **all exactly 0.0** | every label `O` |

There is no NaN and no inf in any of them, so the attention-mask patch is not the problem here. The Simulator's GPU path returns an all-zero buffer for this FP16 model; why, is unknown.

**Workaround:** the default `computeUnits` of `ServiceCoreML.Configuration` is `.cpuAndNeuralEngine` in the Simulator and `.all` on a device. With it all package tests with the real model pass in the Simulator (41 tests, 0 failures, 2026-10-28). Run performance tests on a real device only.

**Still unknown:** why the Simulator's GPU path returns zeros.
**FP32** (about 114 MB `weight.bin`) also worked in the Simulator with `.all`, but it is over the 100 MB GitHub file limit and would need Git LFS, so FP16 stays.

---

## Current working stack

```
torch==2.9.0
transformers>=4.46.0
coremltools>=8.0
numpy>=2.0
sentencepiece>=0.2.0
```

Conversion approach: torch tracing via `WrappedModel` with fixed `position_ids` buffer → `ct.convert(traced, ...)`.
