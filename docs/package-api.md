# `MealRecognize` package and demo screen

`Packages/MealRecognize` has no dependency on the app:

- a WordPiece tokenizer written in Swift (`WordPieceTokenizer`), so the app does not need a Python tokenizer;
- Core ML inference with `MLModel` and a BIO decoder that produces text spans;
- `DishGrouper`, which splits spans into separate dishes and attaches quantities and units;
- `ModelStore`, an `actor` that owns loaded models (one per language and compute-unit setting), loads lazily, loads once for concurrent callers, does not cache failures and unloads on a memory warning;
- `Language` and the `Russian/` folder: everything specific to Russian (model, vocabulary, lower-casing rule) is in one place, and `ServiceCoreML` does not mention a language;
- `ServiceMock`, a configurable stub of the same `ServiceProtocol` used as a fallback: it recognizes nothing by default (`noEntitiesRecognized`) and returns the `result` it is given, so a broken model is visible instead of hidden behind a simpler parser.

Creating an `MLModel` costs real time (see [performance.md](performance.md)), so the library, not its callers, owns the model's lifetime.

`parseMeal` throws `ParsingError` (`emptyTranscript`, `noEntitiesRecognized`, ...) instead of returning an optional.

## Configuring `ServiceCoreML`

`ServiceCoreML(language:)` uses the defaults. For control, pass a `ServiceCoreML.Configuration`:

```swift
import CoreML
import MealRecognize

let configuration = ServiceCoreML.Configuration(
    language: .russian,
    computeUnits: .cpuAndNeuralEngine,   // default: Configuration.defaultComputeUnits
    loading: .eager,                     // default: .lazy
    unloadsOnMemoryWarning: true         // default: true
)
let service = try ServiceCoreML(configuration: configuration)
```

| Parameter | Default | What it does |
|---|---|---|
| `language` | none (required) | Picks the model, the vocabulary and the language rules. Only `.russian` exists today; everything Russian-specific lives in `Sources/MealRecognize/Russian/`. |
| `computeUnits` | `.all` on a device, `.cpuAndNeuralEngine` in the Simulator | Passed to `MLModelConfiguration`. `.cpuAndNeuralEngine` was fastest on the measured device (see [performance.md](performance.md)). The Simulator's GPU path returns zeros for this model, hence its different default. |
| `loading` | `.lazy` | `.lazy` loads the model on the first `prepare()` or `parseMeal` call; `.eager` starts loading in a background task right after the service is created. |
| `unloadsOnMemoryWarning` | `true` | Drops the cached model when the system sends a memory warning; the next call reloads it (about 265 ms with the system cache, measured on the device). |

Other points of the API:

- Creating a service does not load the model. `init` only checks that the model and the vocabulary resources exist, and throws if they do not, so a caller can fall back to `ServiceMock()`: `(try? ServiceCoreML(language: .russian)) ?? ServiceMock()`.
- `prepare()` loads the model and runs one dummy inference, so the first real call is fast. It is safe to call more than once.
- Loaded models are shared: one per language and `computeUnits` value, no matter how many `ServiceCoreML` instances exist. Concurrent callers share one load, and a failed load is not cached.
- `wordNumberParser` takes a `WordNumberParsing` implementation (`parse(_ text: String) -> Double?`) for quantities written as words. The default understands only digits.
- `ServiceCoreML`, `ServiceMock` and the test fakes all conform to `ServiceProtocol` (`prepare()`, `parseMeal(transcript:)`). `ServiceMock` takes a `Result<MealEntities, ParsingError>` and has no language.

## Demo screen

The `CoreMLShowcaseApp` host app has one SwiftUI screen (`ContentView.swift`, Russian UI): type a meal phrase, press "Распознать", and the result is added as the top row of a table (columns Название, Дата, Время, Блюда; empty cells show `—`; all dishes of a phrase are joined in one cell). It uses `ServiceCoreML` with a `ServiceMock` fallback and calls `prepare()` on appear.
