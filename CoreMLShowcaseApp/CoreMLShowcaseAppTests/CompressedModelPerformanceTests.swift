//
//  CompressedModelPerformanceTests.swift
//

import CoreML
import Darwin
import XCTest
@testable import MealRecognize

// Compares FP16 / INT8 / 6-bit palettized variants of MealNerRu. Device only, Release configuration.
final class CompressedModelPerformanceTests: XCTestCase {

    private enum Variant: String {
        case fp16, int8, pal6
    }

    private let phrases = [
        "курица 150 грамм",
        "вчера на ужин курица 150 грамм и гречка 80 грамм и яйца 2 штуки в семь вечера",
        Array(
            repeating: "курица 150 грамм и гречка 80 грамм и яблоко 2 штуки и молоко 200 миллилитров",
            count: 3
        ).joined(separator: " и "),
    ]

    private let metrics: [XCTMetric] = [XCTClockMetric(), XCTMemoryMetric(), XCTCPUMetric()]

    private var options: XCTMeasureOptions {
        let options = XCTMeasureOptions()
        options.iterationCount = 100
        return options
    }

    // MARK: - Helpers

    private func compiledModelURL(_ variant: Variant) throws -> URL {
        switch variant {
        case .fp16:
            return try XCTUnwrap(Language.russian.resourceURL(name: "MealNerRu", withExtension: "mlmodelc"))
        case .int8, .pal6:
            return try XCTUnwrap(
                Bundle(for: Self.self).url(forResource: "MealNerRu_\(variant.rawValue)", withExtension: "mlmodelc")
            )
        }
    }

    private func makeTokenizer() throws -> WordPieceTokenizer {
        let url = try XCTUnwrap(Language.russian.resourceURL(name: Language.russian.vocabularyName, withExtension: "txt"))
        return try WordPieceTokenizer(vocabularyURL: url)
    }

    private func makeLoadedModel(_ variant: Variant, computeUnits: MLComputeUnits) throws -> LoadedModel {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits
        let model = try MLModel(contentsOf: try compiledModelURL(variant), configuration: configuration)
        return LoadedModel(predictor: CoreMLPredictor(model: model), tokenizer: try makeTokenizer(), id2label: [:])
    }

    private func skipOnSimulator() throws {
        #if targetEnvironment(simulator)
        throw XCTSkip("Measure on a device.")
        #endif
    }

    private func residentMB() -> Double {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size / MemoryLayout<natural_t>.size)
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return -1 }
        return Double(info.resident_size) / 1_048_576
    }

    private func measureInference(_ variant: Variant, _ computeUnits: MLComputeUnits) throws {
        try skipOnSimulator()
        let loaded = try makeLoadedModel(variant, computeUnits: computeUnits)
        let tokenizer = try makeTokenizer()
        let encodings = try phrases.map { try tokenizer.encode($0.lowercased(), maxLength: LoadedModel.sequenceLength) }
        for encoding in encodings {
            _ = try loaded.runInference(inputIds: encoding.inputIds, attentionMask: encoding.attentionMask)
        }
        measure(metrics: metrics, options: options) {
            for encoding in encodings {
                _ = try? loaded.runInference(inputIds: encoding.inputIds, attentionMask: encoding.attentionMask)
            }
        }
    }

    private func coldStart(_ variant: Variant) throws {
        try skipOnSimulator()
        let url = try compiledModelURL(variant)
        let tokenizer = try makeTokenizer()
        let encoding = try tokenizer.encode(phrases[0].lowercased(), maxLength: LoadedModel.sequenceLength)
        let before = residentMB()
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndNeuralEngine
        let clock = ContinuousClock()
        var model: MLModel!
        let duration = try clock.measure {
            model = try MLModel(contentsOf: url, configuration: configuration)
        }
        let afterLoad = residentMB()
        let loaded = LoadedModel(predictor: CoreMLPredictor(model: model), tokenizer: tokenizer, id2label: [:])
        _ = try loaded.runInference(inputIds: encoding.inputIds, attentionMask: encoding.attentionMask)
        let afterFirst = residentMB()
        print("[Compressed] \(variant.rawValue) load: \(duration)")
        print("[Compressed] \(variant.rawValue) rss MB before/afterLoad/afterFirstPrediction: "
              + String(format: "%.1f / %.1f / %.1f", before, afterLoad, afterFirst))
    }

    // MARK: - Inference

    func test_inference_fp16_cpuAndNeuralEngine() throws { try measureInference(.fp16, .cpuAndNeuralEngine) }
    func test_inference_fp16_cpuOnly() throws { try measureInference(.fp16, .cpuOnly) }
    func test_inference_int8_cpuAndNeuralEngine() throws { try measureInference(.int8, .cpuAndNeuralEngine) }
    func test_inference_int8_cpuOnly() throws { try measureInference(.int8, .cpuOnly) }
    func test_inference_pal6_cpuAndNeuralEngine() throws { try measureInference(.pal6, .cpuAndNeuralEngine) }
    func test_inference_pal6_cpuOnly() throws { try measureInference(.pal6, .cpuOnly) }

    // MARK: - Cold start

    func test_coldStart_fp16() throws { try coldStart(.fp16) }
    func test_coldStart_int8() throws { try coldStart(.int8) }
    func test_coldStart_pal6() throws { try coldStart(.pal6) }
}
