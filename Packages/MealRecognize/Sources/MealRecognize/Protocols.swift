//
//  Protocols.swift
//  MealRecognize
//

import Foundation

/// Parses quantity spans written as words. Supplied by the host app.
public protocol WordNumberParsing {
    func parse(_ text: String) -> Double?
}

public protocol ServiceProtocol {
    /// Loads the model and warms it up so the first `parseMeal` is fast. Safe to call repeatedly.
    func prepare() async throws
    func parseDish(transcript: String) async -> DishEntities?
    func parseMeal(transcript: String) async throws -> MealEntities
}
