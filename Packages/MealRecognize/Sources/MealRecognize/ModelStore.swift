//
//  ModelStore.swift
//  MealRecognize
//

import CoreML
import Foundation

/// Loads one model. Injected into `ModelStore` so tests can count loads without a real model.
protocol ModelLoading: Sendable {
    func load(language: Language, computeUnits: MLComputeUnits) async throws -> LoadedModel
}

/// Owns loaded models, one per (language, compute units) pair, shared by every `ServiceCoreML`.
///
/// - Loads lazily on the first request.
/// - Concurrent requests for the same key share one in-flight load (the later callers await the same `Task`).
/// - A failed load is not cached: the next request retries.
/// - `unload` drops the cached model; the next request loads it again.
actor ModelStore {

    struct Key: Hashable, Sendable {
        let language: Language
        let computeUnits: MLComputeUnits
    }

    static let shared = ModelStore(loader: CoreMLModelLoader())

    private let loader: any ModelLoading
    private var loaded: [Key: LoadedModel] = [:]
    private var inFlight: [Key: (id: UUID, task: Task<LoadedModel, Error>)] = [:]

    init(loader: any ModelLoading) {
        self.loader = loader
    }

    func model(for language: Language, computeUnits: MLComputeUnits) async throws -> LoadedModel {
        let key = Key(language: language, computeUnits: computeUnits)
        if let model = loaded[key] {
            return model
        }
        if let pending = inFlight[key] {
            return try await pending.task.value
        }

        let id = UUID()
        let loader = self.loader
        // Unstructured on purpose: cancelling one caller must not cancel the load the others wait for.
        let task = Task { try await loader.load(language: language, computeUnits: computeUnits) }
        inFlight[key] = (id, task)
        do {
            let model = try await task.value
            // `unload` may have run while loading; then the result is returned but not cached.
            if inFlight[key]?.id == id {
                inFlight[key] = nil
                loaded[key] = model
            }
            return model
        } catch {
            if inFlight[key]?.id == id {
                inFlight[key] = nil
            }
            throw error
        }
    }

    func unload(language: Language, computeUnits: MLComputeUnits) {
        let key = Key(language: language, computeUnits: computeUnits)
        loaded[key] = nil
        inFlight[key] = nil
    }

    func unloadAll() {
        loaded.removeAll()
        inFlight.removeAll()
    }
}

// MARK: - Production loader

/// Opens the compiled model from the package resources and builds the tokenizer and label table.
struct CoreMLModelLoader: ModelLoading {

    private static let fallbackLabels = [
        "O",
        "B-MEAL_TITLE", "I-MEAL_TITLE",
        "B-MEAL_TYPE", "I-MEAL_TYPE",
        "B-FOOD_NAME", "I-FOOD_NAME",
        "B-QUANTITY", "I-QUANTITY",
        "B-UNIT", "I-UNIT",
        "B-DATE", "I-DATE",
        "B-TIME", "I-TIME",
    ]

    func load(language: Language, computeUnits: MLComputeUnits) async throws -> LoadedModel {
        guard let modelURL = language.resourceURL(name: language.modelName, withExtension: "mlmodelc") else {
            throw ModelLoadingError.missingModel
        }
        guard let vocabularyURL = language.resourceURL(name: language.vocabularyName, withExtension: "txt") else {
            throw ModelLoadingError.missingVocabulary
        }
        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits
        // Async variant: the load does not block a cooperative-pool thread.
        let model = try await MLModel.load(contentsOf: modelURL, configuration: configuration)
        let tokenizer = try WordPieceTokenizer(vocabularyURL: vocabularyURL)
        let id2label = Self.loadId2Label(from: model) ?? Self.makeFallbackId2Label()
        return LoadedModel(predictor: CoreMLPredictor(model: model), tokenizer: tokenizer, id2label: id2label)
    }

    private static func loadId2Label(from model: MLModel) -> [Int: String]? {
        guard let rawValue = model.modelDescription.metadata
            .first(where: { $0.key.rawValue == "id2label" })?.value as? String
        else {
            return nil
        }
        return parsePythonDictString(rawValue)
    }

    private static func makeFallbackId2Label() -> [Int: String] {
        Dictionary(uniqueKeysWithValues: fallbackLabels.enumerated().map { ($0.offset, $0.element) })
    }

    private static func parsePythonDictString(_ value: String) -> [Int: String]? {
        let pattern = #"(\d+):\s*'([^']*)'"#
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return nil }

        let range = NSRange(value.startIndex..., in: value)
        let matches = regex.matches(in: value, options: [], range: range)
        guard !matches.isEmpty else { return nil }

        var labels: [Int: String] = [:]
        labels.reserveCapacity(matches.count)

        for match in matches {
            guard
                let idRange = Range(match.range(at: 1), in: value),
                let labelRange = Range(match.range(at: 2), in: value),
                let id = Int(value[idRange])
            else {
                continue
            }
            labels[id] = String(value[labelRange])
        }

        return labels.isEmpty ? nil : labels
    }
}
