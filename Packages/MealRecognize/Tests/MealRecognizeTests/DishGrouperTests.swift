//
//  DishGrouperTests.swift
//  MealRecognizeTests
//

import XCTest
@testable import MealRecognize

private struct StubWordNumberParser: WordNumberParsing {
    func parse(_ text: String) -> Double? { text == "сто пятьдесят" ? 150 : nil }
}

final class DishGrouperTests: XCTestCase {

    private func spans(_ items: [(String, String)]) -> [TextSpan] {
        items.enumerated().map { TextSpan(label: $0.element.0, text: $0.element.1, position: $0.offset) }
    }

    func test_quantityAfterFood_twoDishes() {
        let result = DishGrouper.group(spans([
            ("FOOD_NAME", "курица"), ("QUANTITY", "150"), ("UNIT", "грамм"),
            ("FOOD_NAME", "гречка"), ("QUANTITY", "80"), ("UNIT", "грамм"),
        ]))
        XCTAssertEqual(result, [
            DishEntities(foodName: "курица", quantity: 150, unit: "грамм"),
            DishEntities(foodName: "гречка", quantity: 80, unit: "грамм"),
        ])
    }

    func test_quantityBeforeFood_twoDishes() {
        let result = DishGrouper.group(spans([
            ("QUANTITY", "150"), ("UNIT", "грамм"), ("FOOD_NAME", "курицы"),
            ("QUANTITY", "80"), ("UNIT", "грамм"), ("FOOD_NAME", "гречки"),
        ]))
        XCTAssertEqual(result, [
            DishEntities(foodName: "курицы", quantity: 150, unit: "грамм"),
            DishEntities(foodName: "гречки", quantity: 80, unit: "грамм"),
        ])
    }

    func test_dishWithoutQuantityBetweenTwoWithQuantity() {
        let result = DishGrouper.group(spans([
            ("FOOD_NAME", "курица"), ("QUANTITY", "150"), ("UNIT", "грамм"),
            ("FOOD_NAME", "салат"),
            ("FOOD_NAME", "гречка"), ("QUANTITY", "80"), ("UNIT", "грамм"),
        ]))
        XCTAssertEqual(result.count, 3)
        XCTAssertEqual(result[0].quantity, 150)
        XCTAssertEqual(result[1], DishEntities(foodName: "салат", quantity: nil, unit: nil))
        XCTAssertEqual(result[2].quantity, 80)
    }

    func test_sameFoodTwice_twoDishes() {
        let result = DishGrouper.group(spans([
            ("FOOD_NAME", "яблоко"), ("QUANTITY", "1"),
            ("FOOD_NAME", "яблоко"), ("QUANTITY", "2"),
        ]))
        XCTAssertEqual(result.map(\.foodName), ["яблоко", "яблоко"])
        XCTAssertEqual(result.map(\.quantity), [1, 2])
    }

    func test_quantityWords() {
        // The number-word parser lives in the app; the package only receives it as a closure.
        let result = DishGrouper.group(
            spans([("FOOD_NAME", "курица"), ("QUANTITY", "сто пятьдесят"), ("UNIT", "грамм")]),
            wordNumberParser: StubWordNumberParser()
        )
        XCTAssertEqual(result, [DishEntities(foodName: "курица", quantity: 150, unit: "грамм")])
    }

    func test_decimalComma() {
        let result = DishGrouper.group(spans([("FOOD_NAME", "молоко"), ("QUANTITY", "0,5"), ("UNIT", "литра")]))
        XCTAssertEqual(result.first?.quantity, 0.5)
    }

    func test_quantityWithoutDish_dropped() {
        let result = DishGrouper.group(spans([("QUANTITY", "150"), ("UNIT", "грамм")]))
        XCTAssertEqual(result, [])
    }

    func test_emptyInput() {
        XCTAssertEqual(DishGrouper.group([]), [])
    }

    func test_singleDish_staysSingle() {
        let result = DishGrouper.group(spans([
            ("FOOD_NAME", "гречка"), ("QUANTITY", "200"), ("UNIT", "грамм"),
        ]))
        XCTAssertEqual(result, [DishEntities(foodName: "гречка", quantity: 200, unit: "грамм")])
    }

    func test_nonDishSpansIgnored() {
        let result = DishGrouper.group(spans([
            ("DATE", "вчера"), ("MEAL_TITLE", "ужин"), ("FOOD_NAME", "курица"),
        ]))
        XCTAssertEqual(result, [DishEntities(foodName: "курица", quantity: nil, unit: nil)])
    }
}

// MARK: - TextSpanDecodingTests

final class TextSpanDecodingTests: XCTestCase {

    func test_wordPieceFood_withQuantityAndUnit() {
        let result = ServiceCoreML.decodeSpans(
            wordTokens: ["кури", "##ца", "150", "грамм"],
            labels: ["B-FOOD_NAME", "I-FOOD_NAME", "B-QUANTITY", "B-UNIT"]
        )
        XCTAssertEqual(result, [
            TextSpan(label: "FOOD_NAME", text: "курица", position: 0),
            TextSpan(label: "QUANTITY", text: "150", position: 2),
            TextSpan(label: "UNIT", text: "грамм", position: 3),
        ])
    }

    func test_continuationLabelledB_doesNotSplitWord() {
        let result = ServiceCoreML.decodeSpans(
            wordTokens: ["греч", "##ка"],
            labels: ["B-FOOD_NAME", "B-FOOD_NAME"]
        )
        XCTAssertEqual(result, [TextSpan(label: "FOOD_NAME", text: "гречка", position: 0)])
    }

    func test_continuationLabelledOtherOrEntity_doesNotBreakSpan() {
        let withO = ServiceCoreML.decodeSpans(
            wordTokens: ["греч", "##ка"],
            labels: ["B-FOOD_NAME", "O"]
        )
        XCTAssertEqual(withO, [TextSpan(label: "FOOD_NAME", text: "гречка", position: 0)])

        let withOtherEntity = ServiceCoreML.decodeSpans(
            wordTokens: ["греч", "##ка"],
            labels: ["B-FOOD_NAME", "B-UNIT"]
        )
        XCTAssertEqual(withOtherEntity, [TextSpan(label: "FOOD_NAME", text: "гречка", position: 0)])
    }

    func test_outsideWordWithContinuation_yieldsNoSpan() {
        let result = ServiceCoreML.decodeSpans(
            wordTokens: ["съе", "##л", "греч", "##ка"],
            labels: ["O", "B-FOOD_NAME", "B-FOOD_NAME", "O"]
        )
        XCTAssertEqual(result, [TextSpan(label: "FOOD_NAME", text: "гречка", position: 2)])
    }

    func test_outsideWordWithOnlyContinuation_yieldsNothing() {
        let result = ServiceCoreML.decodeSpans(
            wordTokens: ["съе", "##л"],
            labels: ["O", "B-FOOD_NAME"]
        )
        XCTAssertEqual(result, [])
    }

    func test_twoFoods_separatedByOutside_givePositions() {
        let result = ServiceCoreML.decodeSpans(
            wordTokens: ["кури", "##ца", "и", "греч", "##ка"],
            labels: ["B-FOOD_NAME", "I-FOOD_NAME", "O", "B-FOOD_NAME", "B-FOOD_NAME"]
        )
        XCTAssertEqual(result, [
            TextSpan(label: "FOOD_NAME", text: "курица", position: 0),
            TextSpan(label: "FOOD_NAME", text: "гречка", position: 3),
        ])
    }

    func test_insideOfDifferentType_startsNewSpan() {
        let result = ServiceCoreML.decodeSpans(
            wordTokens: ["борщ", "200"],
            labels: ["B-FOOD_NAME", "I-QUANTITY"]
        )
        XCTAssertEqual(result, [
            TextSpan(label: "FOOD_NAME", text: "борщ", position: 0),
            TextSpan(label: "QUANTITY", text: "200", position: 1),
        ])
    }
}
