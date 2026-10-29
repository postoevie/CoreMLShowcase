//
//  ServiceMock.swift
//  MealRecognize
//

import Foundation

/// Stub used as the fallback when the Core ML service cannot be created.
/// It recognizes nothing unless a `result` is configured, so a broken model shows up
/// as "nothing recognized" instead of silently degrading to a simpler parser.
public final class ServiceMock: ServiceProtocol {

    public var result: Result<MealEntities, ParsingError>

    public init(result: Result<MealEntities, ParsingError> = .failure(.noEntitiesRecognized)) {
        self.result = result
    }

    public func prepare() async throws {}

    public func parseDish(transcript: String) async -> DishEntities? {
        try? await parseMeal(transcript: transcript).dishes.first
    }

    public func parseMeal(transcript: String) async throws -> MealEntities {
        guard !transcript.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw ParsingError.emptyTranscript
        }
        return try result.get()
    }
}
