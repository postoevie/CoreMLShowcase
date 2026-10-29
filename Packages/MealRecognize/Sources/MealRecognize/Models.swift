//
//  Models.swift
//  MealRecognize
//

import Foundation

public enum ParsingError: Error {
    /// The transcript is empty or whitespace only.
    case emptyTranscript
    /// The model ran but found no meal, food, date or time entities in the transcript.
    case noEntitiesRecognized
    /// The transcript was parsed but contains no dish.
    case noDishesFound
}

/// Dish fields extracted from a transcript. `unit` is the raw span text (no conversion).
public struct DishEntities: Equatable {
    public let foodName: String
    public let quantity: Double?
    public let unit: String?

    public init(foodName: String, quantity: Double?, unit: String?) {
        self.foodName = foodName
        self.quantity = quantity
        self.unit = unit
    }
}

/// Whole-meal entities from one phrase. `title`, `dateText` and `timeText` are raw span texts (no normalization).
public struct MealEntities: Equatable {
    public let title: String?
    public let dateText: String?
    public let timeText: String?
    public let dishes: [DishEntities]

    public init(title: String?, dateText: String?, timeText: String?, dishes: [DishEntities]) {
        self.title = title
        self.dateText = dateText
        self.timeText = timeText
        self.dishes = dishes
    }
}

/// Default `WordNumberParsing`: understands nothing, so only digit quantities are recognized.
public struct NoWordNumberParser: WordNumberParsing {
    public init() {}
    public func parse(_ text: String) -> Double? { nil }
}
