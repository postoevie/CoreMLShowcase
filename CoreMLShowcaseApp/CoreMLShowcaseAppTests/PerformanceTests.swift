//
//  PerformanceTests.swift
//  MealRecognizeTests
//

import CoreML
import XCTest
@testable import MealRecognize

private final class ResultBox<T>: @unchecked Sendable {
    var result: Result<T, Error>?
}

// Numbers are only meaningful on a real device: Release configuration, device cooled down,
// not in Low Power Mode. The Xcode Core ML Performance Report (model file > Performance tab)
// complements these tests with per-layer and per-compute-unit data.
//
// Model-dependent tests are skipped on the simulator: the FP16 model produces all-`O` output there.

final class PerformanceTests: XCTestCase {

    // MARK: - Fixtures

    private let shortPhrase = "курица 150 грамм"
    private let multiDishPhrase = "вчера на ужин курица 150 грамм и гречка 80 грамм и яйца 2 штуки в семь вечера"
    private let longPhrase = Array(
        repeating: "курица 150 грамм и гречка 80 грамм и яблоко 2 штуки и молоко 200 миллилитров",
        count: 3
    ).joined(separator: " и ")

    private var phrases: [String] { [shortPhrase, multiDishPhrase, longPhrase] }

    private let metrics: [XCTMetric] = [XCTClockMetric(), XCTMemoryMetric(), XCTCPUMetric()]

    private var options: XCTMeasureOptions {
        let options = XCTMeasureOptions()
        options.iterationCount = 100
        return options
    }

    private func makeService(computeUnits: MLComputeUnits = .all) throws -> ServiceCoreML {
        #if targetEnvironment(simulator)
        throw XCTSkip("FP16 model returns all-`O` output in the iOS Simulator; measure on a device.")
        #else
        return try ServiceCoreML(language: .russian, computeUnits: computeUnits)
        #endif
    }

    /// Runs an async operation to completion from a synchronous test body (`measure` blocks are synchronous).
    /// The operation runs on a detached task, so the calling thread only waits on the semaphore.
    private func blocking<T>(_ operation: @escaping @Sendable () async throws -> T) throws -> T {
        let box = ResultBox<T>()
        let semaphore = DispatchSemaphore(value: 0)
        Task.detached {
            do {
                box.result = .success(try await operation())
            } catch {
                box.result = .failure(error)
            }
            semaphore.signal()
        }
        semaphore.wait()
        return try box.result!.get()
    }

    private func makeTokenizer() throws -> WordPieceTokenizer {
        let url = try XCTUnwrap(Language.russian.resourceURL(name: Language.russian.vocabularyName, withExtension: "txt"))
        return try WordPieceTokenizer(vocabularyURL: url)
    }

    private func makeEncodings(with tokenizer: WordPieceTokenizer) throws -> [WordPieceTokenizer.Encoding] {
        try phrases.map { try tokenizer.encode($0.lowercased(), maxLength: ServiceCoreML.sequenceLength) }
    }

    // MARK: - Cold start (one-off)

    func test_coldStart_initAndFirstParseMeal() async throws {
        _ = try makeService(computeUnits: .all) // skips on the simulator before anything is measured
        let clock = ContinuousClock()
        var service: ServiceCoreML!
        // `init` is now a cheap facade; the model load moved to the first store request.
        let initDuration = try clock.measure {
            service = try ServiceCoreML(language: .russian, computeUnits: .all)
        }
        let start = clock.now
        _ = try await service.loadedModel()
        let loadDuration = clock.now - start
        let firstParseStart = clock.now
        _ = try await service.parseMeal(transcript: shortPhrase)
        let firstParseDuration = clock.now - firstParseStart
        let secondParseStart = clock.now
        _ = try await service.parseMeal(transcript: shortPhrase)
        let secondParseDuration = clock.now - secondParseStart
        let report = "init: \(initDuration), model load: \(loadDuration), first parseMeal: \(firstParseDuration), second parseMeal: \(secondParseDuration)"
        print("[PerformanceTests] cold start -> \(report)")
        let attachment = XCTAttachment(string: report)
        attachment.name = "cold-start"
        attachment.lifetime = .keepAlways
        add(attachment)
    }

    // MARK: - Pure stages (run on the simulator too)

    func test_tokenizer_encode() throws {
        let tokenizer = try makeTokenizer()
        let inputs = phrases.map { $0.lowercased() }
        _ = try inputs.map { try tokenizer.encode($0, maxLength: ServiceCoreML.sequenceLength) }
        measure(metrics: metrics, options: options) {
            for input in inputs {
                _ = try? tokenizer.encode(input, maxLength: ServiceCoreML.sequenceLength)
            }
        }
    }

    func test_decodeSpans_syntheticLabels() {
        let tokens = ["вчера", "на", "ужин", "кури", "##ца", "150", "грамм", "и", "гречка", "80", "грамм"]
        let labels = ["B-DATE", "O", "B-MEAL_TITLE", "B-FOOD_NAME", "I-FOOD_NAME", "B-QUANTITY", "B-UNIT", "O",
                      "B-FOOD_NAME", "B-QUANTITY", "B-UNIT"]
        let long = (0..<10).flatMap { _ in tokens }
        let longLabels = (0..<10).flatMap { _ in labels }
        _ = ServiceCoreML.decodeSpans(wordTokens: long, labels: longLabels)
        measure(metrics: metrics, options: options) {
            _ = ServiceCoreML.decodeSpans(wordTokens: tokens, labels: labels)
            _ = ServiceCoreML.decodeSpans(wordTokens: long, labels: longLabels)
        }
    }

    func test_dishGrouper_group() {
        let dish = [
            TextSpan(label: "FOOD_NAME", text: "курица", position: 0),
            TextSpan(label: "QUANTITY", text: "150", position: 1),
            TextSpan(label: "UNIT", text: "грамм", position: 2),
        ]
        let short = dish
        let long = (0..<12).flatMap { index in
            dish.map { TextSpan(label: $0.label, text: $0.text, position: $0.position + index * 3) }
        }
        let parser = NoWordNumberParser()
        _ = DishGrouper.group(long, wordNumberParser: parser)
        measure(metrics: metrics, options: options) {
            _ = DishGrouper.group(short, wordNumberParser: parser)
            _ = DishGrouper.group(long, wordNumberParser: parser)
        }
    }

    // MARK: - Model inference only, per compute units

    func test_inference_all() throws { try measureInference(.all) }
    func test_inference_cpuOnly() throws { try measureInference(.cpuOnly) }
    func test_inference_cpuAndGPU() throws { try measureInference(.cpuAndGPU) }
    func test_inference_cpuAndNeuralEngine() throws { try measureInference(.cpuAndNeuralEngine) }

    private func measureInference(_ computeUnits: MLComputeUnits) throws {
        let service = try makeService(computeUnits: computeUnits)
        let loaded = try blocking { try await service.loadedModel() }
        let encodings = try makeEncodings(with: try makeTokenizer())
        for encoding in encodings {
            _ = try loaded.runInference(inputIds: encoding.inputIds, attentionMask: encoding.attentionMask)
        }
        measure(metrics: metrics, options: options) {
            for encoding in encodings {
                _ = try? loaded.runInference(inputIds: encoding.inputIds, attentionMask: encoding.attentionMask)
            }
        }
    }

    // MARK: - Full pipeline, per compute units

    func test_parseMeal_all() throws { try measureParseMeal(.all) }
    func test_parseMeal_cpuOnly() throws { try measureParseMeal(.cpuOnly) }
    func test_parseMeal_cpuAndGPU() throws { try measureParseMeal(.cpuAndGPU) }
    func test_parseMeal_cpuAndNeuralEngine() throws { try measureParseMeal(.cpuAndNeuralEngine) }

    private func measureParseMeal(_ computeUnits: MLComputeUnits) throws {
        let service = try makeService(computeUnits: computeUnits)
        let phrases = self.phrases
        for phrase in phrases {
            _ = try blocking { try await service.parseMeal(transcript: phrase) }
        }
        measure(metrics: metrics, options: options) {
            for phrase in phrases {
                _ = try? blocking { try await service.parseMeal(transcript: phrase) }
            }
        }
    }
}
