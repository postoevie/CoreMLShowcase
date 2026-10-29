//
//  ServiceCoreML.swift
//  MealRecognize
//

import CoreML
import Foundation
#if canImport(UIKit)
import UIKit
#endif

// MARK: - ServiceCoreML

/// Light facade over the shared model held by `ModelStore`. Creating it does not load the model.
public final class ServiceCoreML: ServiceProtocol {

    /// Model input length, fixed at conversion time (`--seq_len` in `tools/convert_to_coreml.py`).
    /// Must match the converted model; changing it means reconverting.
    static let sequenceLength = LoadedModel.sequenceLength

    private let configuration: Configuration
    private let store: ModelStore
    private let wordNumberParser: any WordNumberParsing
    private var memoryWarningObserver: (any NSObjectProtocol)?

    /// - Parameters:
    ///   - language: Picks the model, vocabulary and text rules.
    ///   - wordNumberParser: Parses quantity spans written as words.
    ///   Supplied by the host app; defaults to "not understood".
    /// - Throws: if the model or vocabulary resource is missing from the package. Only the URL lookup is done here.
    public convenience init(language: Language, wordNumberParser: any WordNumberParsing = NoWordNumberParser()) throws {
        try self.init(configuration: Configuration(language: language), wordNumberParser: wordNumberParser)
    }

    /// - Throws: if the model or vocabulary resource is missing from the package. Only the URL lookup is done here.
    public convenience init(configuration: Configuration, wordNumberParser: any WordNumberParsing = NoWordNumberParser()) throws {
        try self.init(configuration: configuration, wordNumberParser: wordNumberParser, store: .shared)
    }

    /// Internal: lets performance tests pin the Core ML compute units.
    convenience init(language: Language, computeUnits: MLComputeUnits, wordNumberParser: any WordNumberParsing = NoWordNumberParser()) throws {
        try self.init(configuration: Configuration(language: language, computeUnits: computeUnits), wordNumberParser: wordNumberParser)
    }

    /// Internal: lets tests inject a store with a mock loader.
    init(configuration: Configuration, wordNumberParser: any WordNumberParsing = NoWordNumberParser(), store: ModelStore) throws {
        let language = configuration.language
        guard language.resourceURL(name: language.modelName, withExtension: "mlmodelc") != nil else {
            throw ModelLoadingError.missingModel
        }
        guard language.resourceURL(name: language.vocabularyName, withExtension: "txt") != nil else {
            throw ModelLoadingError.missingVocabulary
        }
        self.configuration = configuration
        self.wordNumberParser = wordNumberParser
        self.store = store

        #if canImport(UIKit)
        if configuration.unloadsOnMemoryWarning {
            let computeUnits = configuration.computeUnits
            // The block captures the store and the key, not `self`, so there is no retain cycle.
            memoryWarningObserver = NotificationCenter.default.addObserver(
                forName: UIApplication.didReceiveMemoryWarningNotification,
                object: nil,
                queue: nil
            ) { _ in
                Task { await store.unload(language: language, computeUnits: computeUnits) }
            }
        }
        #endif

        if configuration.loading == .eager {
            Task(priority: .utility) { [weak self] in
                try? await self?.prepare()
            }
        }
    }

    deinit {
        if let memoryWarningObserver {
            NotificationCenter.default.removeObserver(memoryWarningObserver)
        }
    }

    /// Loads the model (once, shared with other instances) and runs one dummy prediction.
    /// Cheap and idempotent when the model is already loaded and warm.
    public func prepare() async throws {
        let loaded = try await loadedModel()
        try loaded.warmUpIfNeeded()
    }

    public func parseDish(transcript: String) async -> DishEntities? {
        try? await parseMeal(transcript: transcript).dishes.first
    }

    /// Input beyond `sequenceLength - 2` word pieces is not recognized and is dropped silently (deliberate).
    /// Empty or whitespace-only input throws `ParsingError.emptyTranscript`.
    public func parseMeal(transcript: String) async throws -> MealEntities {
        guard !transcript.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw ParsingError.emptyTranscript
        }
        let loaded = try await loadedModel()
        let language = configuration.language
        // Tokenizer and Core ML errors propagate to the caller unchanged.
        let encoding = try loaded.tokenizer.encode(language.lowercasesInput ? transcript.lowercased() : transcript, maxLength: Self.sequenceLength)
        let logits = try loaded.runInference(inputIds: encoding.inputIds, attentionMask: encoding.attentionMask)
        let id2label = loaded.id2label
        let labelIds = argmaxLabelIds(from: logits, sequenceLength: Self.sequenceLength, numLabels: id2label.count)
        let spans = extractSpans(wordTokens: encoding.wordTokens, labelIds: labelIds, id2label: id2label)

        let recognizedLabels: Set<String> = ["FOOD_NAME", "MEAL_TITLE", "MEAL_TYPE", "DATE", "TIME"]
        guard spans.contains(where: { recognizedLabels.contains($0.label) }) else {
            throw ParsingError.noEntitiesRecognized
        }

        func joinedText(for labels: Set<String>) -> String? {
            let parts = spans
                .filter { labels.contains($0.label) }
                .map { $0.text.trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { !$0.isEmpty }
            return parts.isEmpty ? nil : parts.joined(separator: " ")
        }

        return MealEntities(
            title: joinedText(for: ["MEAL_TITLE", "MEAL_TYPE"]),
            dateText: joinedText(for: ["DATE"]),
            timeText: joinedText(for: ["TIME"]),
            dishes: DishGrouper.group(spans, wordNumberParser: wordNumberParser)
        )
    }

    // MARK: - Private

    /// Internal: the shared loaded model for this configuration (loads it if needed).
    func loadedModel() async throws -> LoadedModel {
        try await store.model(for: configuration.language, computeUnits: configuration.computeUnits)
    }

    private func argmaxLabelIds(from logits: MLMultiArray, sequenceLength: Int, numLabels: Int) -> [Int] {
        var labelIds = [Int](repeating: 0, count: sequenceLength)
        for position in 0..<sequenceLength {
            var bestLabel = 0
            var bestScore = -Float.greatestFiniteMagnitude
            for labelIndex in 0..<numLabels {
                let offset = position * numLabels + labelIndex
                let score = logits[offset].floatValue
                if score > bestScore {
                    bestScore = score
                    bestLabel = labelIndex
                }
            }
            labelIds[position] = bestLabel
        }
        return labelIds
    }

    private func extractSpans(wordTokens: [String], labelIds: [Int], id2label: [Int: String]) -> [TextSpan] {
        guard !wordTokens.isEmpty else { return [] }

        let labelCount = min(wordTokens.count, max(0, labelIds.count - 1))
        let labels = (0..<labelCount).map { id2label[labelIds[$0 + 1]] ?? "O" }
        return Self.decodeSpans(wordTokens: Array(wordTokens.prefix(labelCount)), labels: labels)
    }

    /// BIO decoder over word tokens. The label is read only at word-start tokens (the model is trained on the first
    /// subword of each word); a `##` continuation piece joins the active span regardless of its predicted label
    /// and is ignored when no span is active.
    /// - Parameters:
    ///   - wordTokens: WordPiece tokens without special tokens.
    ///   - labels: Label string per token (`B-FOOD_NAME`, `I-UNIT`, `O`), same order as `wordTokens`.
    static func decodeSpans(wordTokens: [String], labels: [String]) -> [TextSpan] {
        var spans: [TextSpan] = []
        var currentType: String?
        var currentTokens: [String] = []
        var currentStart = 0

        func flushCurrentSpan() {
            guard let activeType = currentType, !currentTokens.isEmpty else { return }
            spans.append(TextSpan(label: activeType, text: decodeWordPieceTokens(currentTokens), position: currentStart))
            currentType = nil
            currentTokens = []
        }

        let count = min(wordTokens.count, labels.count)
        for index in 0..<count {
            let token = wordTokens[index]

            if token.hasPrefix("##") {
                if currentType != nil {
                    currentTokens.append(token)
                }
                continue
            }

            let label = labels[index]

            if label == "O" {
                flushCurrentSpan()
                continue
            }

            let parts = label.split(separator: "-", maxSplits: 1, omittingEmptySubsequences: false)
            guard parts.count == 2 else {
                flushCurrentSpan()
                continue
            }

            let prefix = String(parts[0])
            let entityType = String(parts[1])

            switch prefix {
            case "B":
                flushCurrentSpan()
                currentType = entityType
                currentTokens = [token]
                currentStart = index
            case "I":
                if entityType == currentType {
                    currentTokens.append(token)
                } else {
                    flushCurrentSpan()
                    currentType = entityType
                    currentTokens = [token]
                    currentStart = index
                }
            default:
                flushCurrentSpan()
            }
        }

        flushCurrentSpan()

        return spans
    }

    private static func decodeWordPieceTokens(_ tokens: [String]) -> String {
        tokens.reduce(into: "") { result, token in
            if token.hasPrefix("##") {
                result += String(token.dropFirst(2))
            } else if result.isEmpty {
                result = token
            } else {
                result += " \(token)"
            }
        }
    }
}
