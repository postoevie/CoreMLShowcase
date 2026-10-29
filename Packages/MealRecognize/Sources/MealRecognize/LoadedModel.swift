//
//  LoadedModel.swift
//  MealRecognize
//

import CoreML
import Foundation
import os

/// The prediction step of a model. Exists so the store and its tests do not need a real Core ML model.
protocol ModelPredicting: Sendable {
    func prediction(from input: MLFeatureProvider) throws -> MLFeatureProvider
}

/// Holds an `MLModel` behind `ModelPredicting`.
///
/// `MLModel` is a non-final, non-`Sendable` class, so the compiler cannot prove it safe to share across tasks.
/// `@unchecked Sendable` is correct here: the reference is immutable (`let`), and Apple documents
/// `prediction(from:)` as safe to call concurrently on one instance.
struct CoreMLPredictor: ModelPredicting, @unchecked Sendable {
    let model: MLModel

    func prediction(from input: MLFeatureProvider) throws -> MLFeatureProvider {
        try model.prediction(from: input)
    }
}

/// Everything needed to run one model: the predictor, its tokenizer and the label table.
/// All stored properties are immutable after `init`, except the warm-up flag guarded by a lock.
final class LoadedModel: @unchecked Sendable {

    /// Length of both model inputs. The shape `[1, 128]` is fixed at conversion time (`--seq_len` in
    /// `tools/convert_to_coreml.py`), so this constant must match the converted model.
    /// Changing it means reconverting the model.
    static let sequenceLength = 128

    let predictor: any ModelPredicting
    let tokenizer: WordPieceTokenizer
    let id2label: [Int: String]

    private let warmUpFlag = OSAllocatedUnfairLock(initialState: false)

    init(predictor: any ModelPredicting, tokenizer: WordPieceTokenizer, id2label: [Int: String]) {
        self.predictor = predictor
        self.tokenizer = tokenizer
        self.id2label = id2label
    }

    /// The model stage in isolation (between the tokenizer and span decoding). Synchronous.
    /// Both arrays must be exactly `sequenceLength` long, otherwise `InferenceError.invalidInputLength` is thrown
    /// before anything is written to the model input.
    func runInference(inputIds: [Int32], attentionMask: [Int32]) throws -> MLMultiArray {
        for values in [inputIds, attentionMask] where values.count != Self.sequenceLength {
            throw InferenceError.invalidInputLength(expected: Self.sequenceLength, actual: values.count)
        }
        let shape = [1, NSNumber(value: Self.sequenceLength)]
        let idsArray = try Self.makeMultiArray(inputIds, shape: shape)
        let maskArray = try Self.makeMultiArray(attentionMask, shape: shape)
        let input = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": MLFeatureValue(multiArray: idsArray),
            "attention_mask": MLFeatureValue(multiArray: maskArray),
        ])
        let output = try predictor.prediction(from: input)
        guard let logits = output.featureValue(for: "logits")?.multiArrayValue else {
            throw ModelLoadingError.missingOutput
        }
        return logits
    }

    /// Runs one all-padding prediction to pay the first-call cost (compute plan, GPU preparation) up front.
    /// Does nothing if it already succeeded for this instance.
    func warmUpIfNeeded() throws {
        guard !warmUpFlag.withLock({ $0 }) else { return }
        let encoding = try tokenizer.encode("", maxLength: Self.sequenceLength)
        _ = try runInference(inputIds: encoding.inputIds, attentionMask: encoding.attentionMask)
        warmUpFlag.withLock { $0 = true }
    }

    /// Throws `InferenceError.invalidInputLength` unless `values` fills the array exactly.
    static func makeMultiArray(_ values: [Int32], shape: [NSNumber]) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape, dataType: .int32)
        guard values.count == array.count else {
            throw InferenceError.invalidInputLength(expected: array.count, actual: values.count)
        }
        for (index, value) in values.enumerated() {
            array[index] = NSNumber(value: value)
        }
        return array
    }
}

enum InferenceError: Error, Equatable {
    case invalidInputLength(expected: Int, actual: Int)
}

enum ModelLoadingError: Error {
    case missingModel
    case missingVocabulary
    case missingOutput
}
