//
//  InputLimitsTests.swift
//  MealRecognizeTests
//

import CoreML
import XCTest
@testable import MealRecognize

final class InputLimitsTests: XCTestCase {

    // MARK: - Fixtures

    private struct StubPredictor: ModelPredicting {
        func prediction(from input: MLFeatureProvider) throws -> MLFeatureProvider {
            let logits = try MLMultiArray(shape: [1, 128, 1], dataType: .float32)
            return try MLDictionaryFeatureProvider(dictionary: ["logits": MLFeatureValue(multiArray: logits)])
        }
    }

    private func makeStubTokenizer() throws -> WordPieceTokenizer {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("vocab-\(UUID().uuidString).txt")
        try "[PAD]\n[UNK]\n[CLS]\n[SEP]\n".write(to: url, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: url) }
        return try WordPieceTokenizer(vocabularyURL: url)
    }

    private func makeLoadedModel() throws -> LoadedModel {
        LoadedModel(predictor: StubPredictor(), tokenizer: try makeStubTokenizer(), id2label: [0: "O"])
    }

    private func makeRealTokenizer() throws -> WordPieceTokenizer {
        let language = Language.russian
        let url = try XCTUnwrap(language.resourceURL(name: language.vocabularyName, withExtension: "txt"))
        return try WordPieceTokenizer(vocabularyURL: url)
    }

    private let length = LoadedModel.sequenceLength

    // MARK: - runInference

    func test_runInference_throwsForTooLongInputIds() throws {
        let model = try makeLoadedModel()
        XCTAssertThrowsError(
            try model.runInference(
                inputIds: [Int32](repeating: 0, count: length + 1),
                attentionMask: [Int32](repeating: 0, count: length)
            )
        ) { error in
            XCTAssertEqual(error as? InferenceError, .invalidInputLength(expected: length, actual: length + 1))
        }
    }

    func test_runInference_throwsForShortAttentionMask() throws {
        let model = try makeLoadedModel()
        XCTAssertThrowsError(
            try model.runInference(
                inputIds: [Int32](repeating: 0, count: length),
                attentionMask: [Int32](repeating: 0, count: length - 1)
            )
        ) { error in
            XCTAssertEqual(error as? InferenceError, .invalidInputLength(expected: length, actual: length - 1))
        }
    }

    func test_runInference_throwsForEmptyArrays() throws {
        let model = try makeLoadedModel()
        XCTAssertThrowsError(try model.runInference(inputIds: [], attentionMask: [])) { error in
            XCTAssertEqual(error as? InferenceError, .invalidInputLength(expected: length, actual: 0))
        }
    }

    func test_runInference_acceptsExactLength() throws {
        let model = try makeLoadedModel()
        XCTAssertNoThrow(
            try model.runInference(
                inputIds: [Int32](repeating: 0, count: length),
                attentionMask: [Int32](repeating: 0, count: length)
            )
        )
    }

    func test_makeMultiArray_throwsInsteadOfWritingOutOfBounds() {
        let shape = [1, NSNumber(value: length)]
        XCTAssertThrowsError(try LoadedModel.makeMultiArray([Int32](repeating: 1, count: length + 1), shape: shape))
    }

    // MARK: - Tokenizer

    func test_encode_veryLongPhrase_isTruncatedToFixedLength() throws {
        let tokenizer = try makeRealTokenizer()
        let text = Array(repeating: "курица", count: 300).joined(separator: " ")

        let encoding = try tokenizer.encode(text, maxLength: length)

        XCTAssertEqual(encoding.inputIds.count, length)
        XCTAssertEqual(encoding.attentionMask.count, length)
        XCTAssertEqual(encoding.wordTokens.count, length - 2)
        // An empty text encodes to `[CLS] [SEP] [PAD]...`, which gives the `[SEP]` id.
        let sepID = try tokenizer.encode("", maxLength: length).inputIds[1]
        XCTAssertEqual(encoding.inputIds[length - 1], sepID)
        XCTAssertEqual(encoding.attentionMask.reduce(0, +), Int32(length))
    }

    func test_encode_shortPhrase_maskCoversRealTokensOnly() throws {
        let tokenizer = try makeRealTokenizer()

        let encoding = try tokenizer.encode("курица 150 грамм", maxLength: length)

        XCTAssertEqual(encoding.inputIds.count, length)
        XCTAssertEqual(encoding.attentionMask.count, length)
        XCTAssertEqual(encoding.attentionMask.reduce(0, +), Int32(encoding.wordTokens.count + 2))
    }
}
