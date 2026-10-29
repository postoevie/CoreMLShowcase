# Performance measurements

Model: `MealNerRu` (fine-tuned `rubert-tiny2`, token classification, fixed shape `[1, 128]`), Core ML ML Program, FP16 (54 MB) and FP32 (109 MB).
Device: **iPhone 13, iOS 26.3.1**. Date of the runs: 2026-10-28.

Run performance tests on a real device only. The Simulator has no Neural Engine and no real GPU; the model-dependent performance tests are skipped there.

## Method

- Tests: `CoreMLShowcaseApp/CoreMLShowcaseAppTests/PerformanceTests.swift`, run with `xcodebuild test` on the device, Release configuration, `ENABLE_TESTABILITY=YES`.
- XCTest `measure` with `XCTClockMetric`, `XCTMemoryMetric` and `XCTCPUMetric`, 100 iterations, one warm-up call before each measured block.
- Three fixed Russian phrases (one dish; several dishes with date, time and title; a long phrase near the 128-token cap).
- Inference-only tests call the model directly on pre-computed encodings, one test per `MLComputeUnits` value.
- The cold start is measured once in a separate test (`ContinuousClock`), because `measure` cannot capture a one-off.
- Not measured: energy; memory per compute-unit setting (see the note under "Memory").

## Inference, per compute-unit setting

100 iterations, median / 95th percentile, milliseconds:

| `computeUnits` | FP32 | FP16 |
|---|---|---|
| `.cpuAndNeuralEngine` | 4.20 / 4.37 | 2.36 / 2.63 |
| `.cpuOnly` | 4.30 / 4.35 | 2.51 / 2.56 |
| `.all` | 21.63 / 22.35 | 6.90 / 9.02 |
| `.cpuAndGPU` | 21.80 / 22.41 | 21.32 / 21.94 |

| | FP32 | FP16 |
|---|---|---|
| Package size | 109 MB | 54 MB |
| CPU time of one inference, `.cpuAndNeuralEngine`, median | 5.36 ms | 0.85 ms |
| CPU time of one inference, `.cpuOnly`, median | 5.51 ms | 3.20 ms |
| Model load (cold start of the test process) | 1.08 s | 0.93 s |

Pre- and post-processing (FP16, median / 95th percentile): tokenizer 0.130 / 0.137 ms, `decodeSpans` 0.039 / 0.040 ms, `DishGrouper` 0.018 / 0.020 ms.

Memory: the model store keeps one loaded model per compute-unit setting in one process, so the peak memory printed by the performance run grows from test to test and is not comparable between settings. A proper memory comparison needs one process per setting.

**Unit of the times.** One measured iteration runs the three fixed phrases, so every wall-clock time in these tables is for three inferences in a row, not one. Divide by 3 for a single inference (for example FP16 `.cpuAndNeuralEngine`: 2.36 ms per iteration, about 0.8 ms per inference). The ratios between rows are unaffected.

## Compressed variants (INT8 and 6-bit palettized)

Date of the run: 2026-10-29, same device (iPhone 13, iOS 26.3.1), Release, `ENABLE_TESTABILITY=YES`, clean derived data, 100 iterations, same three phrases, same method (`CompressedModelPerformanceTests`). All three variants are loaded through one code path (`MLModel(contentsOf:configuration:)` on the compiled model) and are measured in the same run, so the FP16 column here differs slightly from the table above (separate run). How the variants were made and their F1: [model-and-conversion.md](model-and-conversion.md#weight-compression). Bundled `weight.bin` sizes were checked after the build: FP16 57,086,302 B, INT8 28,823,518 B, palettized 21,428,062 B.

Median / 95th percentile in ms per iteration (three inferences), CPU time median in ms:

| Variant | `.cpuAndNeuralEngine` wall | CPU time | `.cpuOnly` wall | CPU time |
|---|---|---|---|---|
| FP16 | 2.347 / 2.556 | 0.893 | 2.504 / 2.572 | 3.199 |
| INT8 | 2.239 / 2.397 | 0.857 | 2.503 / 2.546 | 3.196 |
| Palettized 6-bit | 1.974 / 2.230 | 0.832 | 2.507 / 2.552 | 3.199 |

Cold start, `.cpuAndNeuralEngine`, one fresh test process per variant, single samples:

| Variant | Model load | Resident size after load / after first prediction |
|---|---|---|
| FP16 | 1.56 s | 107.6 / 108.1 MB |
| INT8 | 1.18 s | 123.5 / 124.1 MB |
| Palettized 6-bit | 2.07 s | 123.0 / 123.5 MB |

Reading:

- `.cpuOnly` is identical for all three variants (2.50 ms). That fits the compressed weights being expanded to FP16 before the arithmetic: the compute does not change.
- On the Neural Engine path INT8 is about 5% faster and 6-bit about 16% faster than FP16. It is one run and the differences are small; repeat the run before drawing a conclusion. Why palettization would be faster is not known.
- The smaller file did not give a smaller footprint after load: the resident size of the compressed variants (about 123 MB) is higher than FP16 (108 MB). The simplest reading is that the runtime expands the compressed weights to FP16 when the model is prepared, but this was not verified. The "before" resident size of the three processes differed a lot (58 to 144 MB), so the test host was not equally clean, and these numbers are only roughly comparable.
- Model load times (1.2, 1.6 and 2.1 s) are single samples that probably include Neural Engine compile and cache effects; their order is not reliable.
- The `XCTMemoryMetric` delta was 0 kB in all runs, and the peak physical memory (85 to 91 MB) is noisy, so neither says anything about the variants.
- Not checked in these tests: that the compressed variants return the same predictions as FP16 (this is covered by `tools/evaluate.py`, in Python on macOS, not on the device). Energy was not measured.

## Findings

- FP16 is about twice as fast as FP32 for inference on the CPU (2.51 against 4.30 ms) and nearly twice as fast on the Neural Engine path (2.36 against 4.20 ms), at half the size.
- On FP32, `.cpuAndNeuralEngine` and `.cpuOnly` are equal (4.20 and 4.30 ms). That fits the Neural Engine computing only in FP16: an FP32 model does not run on it.
- On FP16, the Neural Engine barely wins on wall-clock time (2.36 against 2.51 ms, about 6%) but the CPU time drops from 3.20 to 0.85 ms, about 3.8 times. It frees the CPU and should save energy (not measured).
- `.all` on FP16 is worse than `.cpuAndNeuralEngine` (6.90 against 2.36 ms) and noisier (95th percentile 9.02 ms). The default `.all` is not the best choice for this model.

Why `.cpuAndNeuralEngine` is only slightly faster than `.cpuOnly` on FP16:

1. The model is small: hidden size 312, shape `[1, 128]`, batch 1. The computation itself takes about 2 ms. The Neural Engine adds fixed overhead (request setup, buffer transfers) comparable to its gain, and the CPU in FP16 with NEON is fast on matrices this size.
2. Part of the graph does not run on the Neural Engine. In Xcode's report the embedding `gather` (about 1.15 ms by the report's estimate) and six more operations run on the GPU. With `.cpuOnly` there are no transfers between devices.

## Why the GPU is slow

`.cpuAndGPU` takes about 21 ms against 2.4 ms on the Neural Engine path (FP16). In the Instruments trace each `Prediction` has repeating `Prepare GPU Request` and `GPU Request` intervals, so there is a per-call cost. This is an interpretation of the trace and the numbers; the internals of Core ML's GPU backend are not documented, and points 1 to 3 are hypotheses.

1. **The model is too small for the GPU.** Batch 1, 128 tokens, hidden size 312. The work is split into dozens of small operations, each launched as its own GPU kernel with a fixed launch latency, and there is little arithmetic inside.
2. **Per-call overhead.** The CPU encodes commands for the GPU, submits them, waits for completion and reads the result back: CPU time is about 10.7 ms of 21 ms wall-clock for `.cpuAndGPU`. On the Neural Engine the graph is compiled ahead of time (the "prepare and cache" load step) and each prediction runs a ready program (CPU time 0.85 ms).
3. **GPU clock.** On short bursts the GPU may not reach a high clock. Not visible in the trace.
4. **Confirmed by measurement: precision does not matter.** `.cpuAndGPU` is 21.80 ms on FP32 and 21.32 ms on FP16. If arithmetic set the time, FP16 would be much faster, so the time is overhead.
5. **Why `.all` is worse than `.cpuAndNeuralEngine`** (6.90 against 2.36 ms on FP16): in `.all`, seven operations from the start of the graph run on the GPU and the other 75 on the Neural Engine, so every call needs a data transfer and a synchronisation at the boundary. With `.cpuAndNeuralEngine` those seven cheap operations run on the CPU.

Open: a Metal System Trace (or GPU counters) on the `.cpuAndGPU` test would check points 1 to 3.

## Xcode Core ML performance report (FP16, warm medians)

| Metric | Value |
|---|---|
| Prediction | 1.31 ms |
| Load | 44.51 ms |
| Compilation | 42.57 ms |
| Compute units (82 operations) | Neural Engine 75, GPU 7, CPU 0 |

The seven GPU operations are at the start of the graph, before the transformer blocks: the embedding `gather` on `input_ids`, two `add`, a `layer_norm`, two `expand_dims` and a `cast`. This report was taken earlier than the 100-iteration runs, and the device was not recorded for it.

## Instruments, Core ML track (FP16, cold events of one run)

| Event | Time |
|---|---|
| Load (prepare and cache) | 4.09 s |
| Prediction (first) | 1.60 s |
| Prepare GPU Request | 1.40 s |
| Load (cached) | 265.48 ms |
| Compile | 135.10 ms |

A warm prediction is fast (about 1.3 ms), but the first use is expensive: about 4 s the first time the model is prepared and cached on the device, about 265 ms for each later load from the cache, and about 1.6 s for the first prediction, of which 1.4 s is preparing the GPU request. The user only notices this on the first use, not on every phrase. Practical consequence: load the model and run one dummy inference in the background (`ServiceCoreML.prepare()`) before the user needs it.

Hypothesis, not checked: the GPU request preparation is caused by the seven GPU operations. A run with `computeUnits = .cpuAndNeuralEngine` would show whether `Prepare GPU Request` disappears.

The numbers of the performance report and of Instruments differ (for example load 44 ms and 265 ms) because the report gives a warm median and Instruments shows single events.

## Notes

- **Simulator.** The Simulator's GPU path returns all-zero logits for the FP16 model (`.all`, `.cpuAndGPU`); `.cpuOnly` and `.cpuAndNeuralEngine` are correct. The reason is unknown. `SimulatorDiagnosticTests` found this by printing the logits per compute-unit setting; see [conversion-log.md](conversion-log.md), entry 12.
- **Tests on a device.** Tests in a Swift package cannot run on a device without a host app, which is why the performance tests live in the `CoreMLShowcaseApp` project.
- **Status of the explanations.** The GPU reading above comes from the Instruments trace and the numbers; a Metal System Trace has not been taken, so it is not verified.
- **Not measured:** energy, and memory per compute-unit setting.
