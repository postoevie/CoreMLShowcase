//
//  SimulatorDiagnosticTests.swift
//  MealRecognizeTests
//

import CoreML
import XCTest
@testable import MealRecognize

// Diagnostics only: prints raw-logit statistics per compute unit (NaN / inf counts, argmax labels).
// Does not skip on the simulator and asserts no model correctness; fails only on thrown errors.

final class SimulatorDiagnosticTests: XCTestCase {

    private let phrase = "я съел сто пятьдесят грамм курицы и восемьдесят грамм гречки вчера на ужин"

    func test_diag_cpuOnly() async throws { try await diagnose(.cpuOnly, name: "cpuOnly") }
    func test_diag_cpuAndGPU() async throws { try await diagnose(.cpuAndGPU, name: "cpuAndGPU") }
    func test_diag_cpuAndNeuralEngine() async throws { try await diagnose(.cpuAndNeuralEngine, name: "cpuAndNeuralEngine") }
    func test_diag_all() async throws { try await diagnose(.all, name: "all") }

    private func diagnose(_ computeUnits: MLComputeUnits, name: String) async throws {
        #if targetEnvironment(simulator)
        let platform = "simulator"
        #else
        let platform = "device"
        #endif
        let env = ProcessInfo.processInfo.environment
        print("[SimDiag] start platform=\(platform) SIMULATOR_MODEL_IDENTIFIER=\(env["SIMULATOR_MODEL_IDENTIFIER"] ?? "nil") os=\(ProcessInfo.processInfo.operatingSystemVersionString)")

        let service = try ServiceCoreML(language: .russian, computeUnits: computeUnits)
        let loaded = try await service.loadedModel()
        let encoding = try loaded.tokenizer.encode(phrase.lowercased(), maxLength: ServiceCoreML.sequenceLength)
        let realTokenCount = encoding.attentionMask.reduce(0) { $0 + Int($1) }
        let logits = try loaded.runInference(inputIds: encoding.inputIds, attentionMask: encoding.attentionMask)

        let values = Self.floatValues(of: logits)
        var nanCount = 0, infCount = 0
        var minValue = Float.greatestFiniteMagnitude, maxValue = -Float.greatestFiniteMagnitude
        for value in values {
            if value.isNaN { nanCount += 1 } else if value.isInfinite { infCount += 1 } else {
                minValue = min(minValue, value)
                maxValue = max(maxValue, value)
            }
        }

        // Layout: [1, sequence, labels] (or [sequence, labels]); read with strides, not assuming contiguity.
        let shape = logits.shape.map(\.intValue)
        let strides = logits.strides.map(\.intValue)
        let labelCount = shape.last ?? 0
        let tokenStride = shape.count >= 2 ? strides[shape.count - 2] : labelCount
        let labelStride = strides.last ?? 1
        var argmaxIds: [Int] = []
        for position in 0..<min(realTokenCount, ServiceCoreML.sequenceLength) {
            var best = 0
            var bestScore = -Float.infinity
            for label in 0..<labelCount {
                let score = Self.value(of: logits, at: position * tokenStride + label * labelStride)
                if score > bestScore { bestScore = score; best = label }
            }
            argmaxIds.append(best)
        }
        let names = argmaxIds.map { loaded.id2label[$0] ?? "?" }

        print("[SimDiag] \(platform) units=\(name) dtype=\(logits.dataType.rawValue) (\(Self.dataTypeName(logits.dataType))) shape=\(shape) strides=\(strides) NaN=\(nanCount) inf=\(infCount) min=\(minValue) max=\(maxValue) realTokens=\(realTokenCount) argmaxIds=\(argmaxIds) labels=\(names)")

        do {
            let meal = try await service.parseMeal(transcript: phrase)
            print("[SimDiag] \(platform) units=\(name) parseMeal -> \(meal)")
        } catch {
            print("[SimDiag] \(platform) units=\(name) parseMeal threw \(error)")
        }
    }

    private static func dataTypeName(_ type: MLMultiArrayDataType) -> String {
        switch type {
        case .float16: return "float16"
        case .float32: return "float32"
        case .float64: return "float64"
        case .int32: return "int32"
        @unknown default: return "other"
        }
    }

    /// Reads one element by linear memory offset (in elements), honoring the data type.
    private static func value(of array: MLMultiArray, at offset: Int) -> Float {
        switch array.dataType {
        case .float32:
            return array.dataPointer.assumingMemoryBound(to: Float.self)[offset]
        case .float16:
            if #available(iOS 14.0, *) {
                return Float(array.dataPointer.assumingMemoryBound(to: Float16.self)[offset])
            }
            return array[offset].floatValue
        case .float64:
            return Float(array.dataPointer.assumingMemoryBound(to: Double.self)[offset])
        default:
            return array[offset].floatValue
        }
    }

    /// All elements in memory order, up to the extent the strides address.
    private static func floatValues(of array: MLMultiArray) -> [Float] {
        let extent = zip(array.shape, array.strides).map { ($0.intValue - 1) * $1.intValue }.reduce(0, +) + 1
        return (0..<max(0, extent)).map { value(of: array, at: $0) }
    }
}
