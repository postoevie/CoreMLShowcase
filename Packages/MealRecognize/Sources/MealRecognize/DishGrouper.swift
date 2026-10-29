//
//  DishGrouper.swift
//  MealRecognize
//

import Foundation

// MARK: - TextSpan

/// One decoded NER span. `label` has no `B-` / `I-` prefix; `position` is the index of the span's first word token.
struct TextSpan: Equatable {
    let label: String
    let text: String
    let position: Int
}

// MARK: - DishGrouper

enum DishGrouper {


    /// Groups ordered FOOD_NAME / QUANTITY / UNIT spans into separate dishes. Other labels are ignored.
    ///
    /// - Parameter wordNumberParser: Fallback for quantity spans that are not digits (e.g. number words).
    ///   Injected by the host app; the package has no number-word parser of its own.
    static func group(_ spans: [TextSpan], wordNumberParser: any WordNumberParsing = NoWordNumberParser()) -> [DishEntities] {
        struct DishBuilder {
            let name: String
            var quantity: Double?
            var unit: String?
        }
        enum Target {
            case dish(Int)
            case pending
        }

        var dishes: [DishBuilder] = []
        var pendingQuantity: Double?
        var pendingUnit: String?
        var quantityTarget: Target?

        let ordered: [TextSpan] = spans.enumerated()
            .sorted { (lhs: (offset: Int, element: TextSpan), rhs: (offset: Int, element: TextSpan)) -> Bool in
                if lhs.element.position != rhs.element.position {
                    return lhs.element.position < rhs.element.position
                }
                return lhs.offset < rhs.offset
            }
            .map { (item: (offset: Int, element: TextSpan)) -> TextSpan in item.element }

        for span in ordered {
            let text = span.text.trimmingCharacters(in: .whitespacesAndNewlines)

            switch span.label {
            case "FOOD_NAME":
                guard !text.isEmpty else { continue }
                dishes.append(DishBuilder(name: text, quantity: pendingQuantity, unit: pendingUnit))
                pendingQuantity = nil
                pendingUnit = nil
                quantityTarget = nil

            case "QUANTITY":
                guard let value = parseQuantity(text, wordNumberParser: wordNumberParser) else { continue }
                if let last = dishes.indices.last, dishes[last].quantity == nil {
                    dishes[last].quantity = value
                    quantityTarget = .dish(last)
                } else {
                    pendingQuantity = value
                    quantityTarget = .pending
                }

            case "UNIT":
                guard !text.isEmpty else { continue }
                switch quantityTarget {
                case .dish(let index):
                    if dishes[index].unit == nil { dishes[index].unit = text }
                case .pending:
                    if pendingUnit == nil { pendingUnit = text }
                case nil:
                    if let last = dishes.indices.last, dishes[last].unit == nil {
                        dishes[last].unit = text
                    } else if pendingUnit == nil {
                        pendingUnit = text
                    }
                }

            default:
                continue
            }
        }

        return dishes.map { DishEntities(foodName: $0.name, quantity: $0.quantity, unit: $0.unit) }
    }

    /// Digits (comma or dot decimals) first, then number words via the injected parser.
    static func parseQuantity(_ span: String, wordNumberParser: any WordNumberParsing) -> Double? {
        let normalized = span
            .components(separatedBy: .whitespacesAndNewlines).joined()
            .replacingOccurrences(of: ",", with: ".")
        if let digits = Double(normalized) {
            return digits
        }
        return wordNumberParser.parse(span)
    }
}
