//
//  ModelStoreTests.swift
//  MealRecognizeTests
//

import CoreML
import XCTest
@testable import MealRecognize

final class ModelStoreTests: XCTestCase {

    // MARK: - Fixtures

    private struct StubPredictor: ModelPredicting {
        func prediction(from input: MLFeatureProvider) throws -> MLFeatureProvider {
            try MLDictionaryFeatureProvider(dictionary: [:])
        }
    }

    private struct LoadFailure: Error {}

    /// Counts loads per key. Suspends each load briefly so concurrent callers overlap in flight.
    private actor CountingLoader: ModelLoading {
        private(set) var loadCount = 0
        private var failuresLeft: Int
        private let tokenizer: WordPieceTokenizer

        init(tokenizer: WordPieceTokenizer, failures: Int = 0) {
            self.tokenizer = tokenizer
            self.failuresLeft = failures
        }

        func load(language: Language, computeUnits: MLComputeUnits) async throws -> LoadedModel {
            loadCount += 1
            try await Task.sleep(nanoseconds: 200_000_000)
            if failuresLeft > 0 {
                failuresLeft -= 1
                throw LoadFailure()
            }
            return LoadedModel(predictor: StubPredictor(), tokenizer: tokenizer, id2label: [0: "O"])
        }
    }

    private func makeTokenizer() throws -> WordPieceTokenizer {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("vocab-\(UUID().uuidString).txt")
        try "[PAD]\n[UNK]\n[CLS]\n[SEP]\n".write(to: url, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: url) }
        return try WordPieceTokenizer(vocabularyURL: url)
    }

    private func makeStore(failures: Int = 0) throws -> (ModelStore, CountingLoader) {
        let loader = CountingLoader(tokenizer: try makeTokenizer(), failures: failures)
        return (ModelStore(loader: loader), loader)
    }

    // MARK: - Tests

    func test_concurrentRequests_loadOnce() async throws {
        let (store, loader) = try makeStore()

        let models = await withTaskGroup(of: LoadedModel?.self) { group in
            for _ in 0..<10 {
                group.addTask { try? await store.model(for: .russian, computeUnits: .all) }
            }
            var result: [LoadedModel] = []
            for await model in group {
                if let model { result.append(model) }
            }
            return result
        }

        XCTAssertEqual(models.count, 10)
        XCTAssertTrue(models.allSatisfy { $0 === models[0] }, "all callers must get the same instance")
        let loadCount = await loader.loadCount
        XCTAssertEqual(loadCount, 1)
    }

    func test_cachedModel_isNotReloaded() async throws {
        let (store, loader) = try makeStore()

        let first = try await store.model(for: .russian, computeUnits: .all)
        let second = try await store.model(for: .russian, computeUnits: .all)

        XCTAssertTrue(first === second)
        let loadCount = await loader.loadCount
        XCTAssertEqual(loadCount, 1)
    }

    func test_differentComputeUnits_loadTwice() async throws {
        let (store, loader) = try makeStore()

        let all = try await store.model(for: .russian, computeUnits: .all)
        let cpu = try await store.model(for: .russian, computeUnits: .cpuOnly)

        XCTAssertFalse(all === cpu)
        let loadCount = await loader.loadCount
        XCTAssertEqual(loadCount, 2)
    }

    func test_afterUnload_nextRequestLoadsAgain() async throws {
        let (store, loader) = try makeStore()

        let first = try await store.model(for: .russian, computeUnits: .all)
        await store.unload(language: .russian, computeUnits: .all)
        let second = try await store.model(for: .russian, computeUnits: .all)

        XCTAssertFalse(first === second)
        let loadCount = await loader.loadCount
        XCTAssertEqual(loadCount, 2)
    }

    func test_failedLoad_isNotCached() async throws {
        let (store, loader) = try makeStore(failures: 1)

        do {
            _ = try await store.model(for: .russian, computeUnits: .all)
            XCTFail("first load should fail")
        } catch {
            XCTAssertTrue(error is LoadFailure)
        }
        _ = try await store.model(for: .russian, computeUnits: .all)

        let loadCount = await loader.loadCount
        XCTAssertEqual(loadCount, 2)
    }

    func test_concurrentRequests_shareFailureOfOneLoad() async throws {
        let (store, loader) = try makeStore(failures: 1)

        let failures = await withTaskGroup(of: Bool.self) { group in
            for _ in 0..<5 {
                group.addTask { (try? await store.model(for: .russian, computeUnits: .all)) == nil }
            }
            var count = 0
            for await failed in group where failed { count += 1 }
            return count
        }

        XCTAssertEqual(failures, 5)
        let loadCount = await loader.loadCount
        XCTAssertEqual(loadCount, 1)
    }
}
