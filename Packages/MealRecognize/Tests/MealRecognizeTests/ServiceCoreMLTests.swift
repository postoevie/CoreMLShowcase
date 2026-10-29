//
//  ServiceCoreMLTests.swift
//  MealRecognizeTests
//

import XCTest
@testable import MealRecognize

// NOTE: expectations below were written WITHOUT running the model.
// The exact NER output is unknown, so assertions are tolerant (stems, case-insensitive).
// They may need adjusting after the first run.

final class ServiceCoreMLTests: XCTestCase {

    private var service: ServiceCoreML!

    override func setUpWithError() throws {
        do {
            service = try ServiceCoreML(language: .russian)
        } catch {
            XCTFail("ServiceCoreML init failed (MealNerRu.mlpackage / vocab.txt missing from test host bundle?): \(error)")
            throw error
        }
    }

    override func tearDown() {
        service = nil
        super.tearDown()
    }

    // MARK: - Helpers

    private func assertDish(
        _ transcript: String,
        foodStem: String,
        quantity: Double?,
        unitStem: String?,
        file: StaticString = #filePath,
        line: UInt = #line
    ) async {
        guard let dish = await service.parseDish(transcript: transcript) else {
            XCTFail("parseDish returned nil for \"\(transcript)\"", file: file, line: line)
            return
        }
        XCTAssertTrue(
            dish.foodName.lowercased().contains(foodStem),
            "foodName \"\(dish.foodName)\" should contain \"\(foodStem)\" for \"\(transcript)\"",
            file: file, line: line
        )
        XCTAssertEqual(dish.quantity, quantity, "quantity for \"\(transcript)\"", file: file, line: line)
        if let unitStem {
            XCTAssertTrue(
                dish.unit?.lowercased().contains(unitStem) ?? false,
                "unit \"\(dish.unit ?? "nil")\" should contain \"\(unitStem)\" for \"\(transcript)\"",
                file: file, line: line
            )
        } else {
            XCTAssertNil(dish.unit, "unit for \"\(transcript)\"", file: file, line: line)
        }
    }

    // MARK: - CoreML: food + quantity + unit

    func test_coreML_foodQuantityGrams_chicken150Grams() async {
        await assertDish("курица 150 грамм", foodStem: "кур", quantity: 150, unitStem: "грамм")
    }

    func test_coreML_ignoresMealTypeAndVerb_addChickenForDinner() async {
        await assertDish("добавь курицу 150 грамм на ужин", foodStem: "кур", quantity: 150, unitStem: "грамм")
    }

    func test_coreML_pieces_appleTwoPieces() async {
        await assertDish("яблоко 2 штуки", foodStem: "яблок", quantity: 2, unitStem: "шт")
    }

    func test_coreML_milliliters_milk200() async {
        await assertDish("молоко 200 миллилитров", foodStem: "молок", quantity: 200, unitStem: "мил")
    }

    func test_coreML_shortGramUnit_buckwheat80g() async {
        await assertDish("гречка 80 г", foodStem: "гречк", quantity: 80, unitStem: "г")
    }

    // MARK: - CoreML: missing parts

    func test_coreML_noQuantity_cottageCheese_quantityIsNil() async {
        guard let dish = await service.parseDish(transcript: "творог") else {
            XCTFail("parseDish returned nil for \"творог\"")
            return
        }
        XCTAssertTrue(dish.foodName.lowercased().contains("творог"), dish.foodName)
        XCTAssertNil(dish.quantity)
    }

    func test_coreML_noFood_smallTalk_returnsNil() async {
        let dish = await service.parseDish(transcript: "привет как дела")
        XCTAssertNil(dish)
    }

}

// MARK: - Mock (configurable stub, no model needed)

final class ServiceMockTests: XCTestCase {

    private let chicken = DishEntities(foodName: "курица", quantity: 150, unit: "грамм")
    private let milk = DishEntities(foodName: "молоко", quantity: 0.5, unit: "литра")

    private func meal(_ dishes: [DishEntities]) -> MealEntities {
        MealEntities(title: nil, dateText: nil, timeText: nil, dishes: dishes)
    }

    func test_mock_defaultResult_throwsNoEntitiesRecognized() async {
        do {
            _ = try await ServiceMock().parseMeal(transcript: "курица 150 грамм")
            XCTFail("expected noEntitiesRecognized")
        } catch {
            XCTAssertEqual(error as? ParsingError, .noEntitiesRecognized)
        }
    }

    func test_mock_configuredSuccess_isReturned() async throws {
        let expected = meal([chicken, milk])
        let mock = ServiceMock(result: .success(expected))
        let actual = try await mock.parseMeal(transcript: "что угодно")
        XCTAssertEqual(actual, expected)
    }

    func test_mock_blankTranscript_throwsEmptyTranscriptEvenWithResult() async {
        let mock = ServiceMock(result: .success(meal([chicken])))
        do {
            _ = try await mock.parseMeal(transcript: "  \n ")
            XCTFail("expected emptyTranscript")
        } catch {
            XCTAssertEqual(error as? ParsingError, .emptyTranscript)
        }
    }

    func test_mock_parseDish_returnsFirstDish() async {
        let mock = ServiceMock(result: .success(meal([chicken, milk])))
        let dish = await mock.parseDish(transcript: "что угодно")
        XCTAssertEqual(dish, chicken)
    }

    func test_mock_parseDish_nilWhenNothingConfigured() async {
        let dish = await ServiceMock().parseDish(transcript: "курица 150 грамм")
        XCTAssertNil(dish)
    }
}

// MARK: - CoreML: meal-level parsing (multi-dish, title, date, time)

// NOTE: expectations come from the converted model run in Python; the Swift run may differ slightly,
// so assertions stay tolerant (stems, case-insensitive).

final class ServiceCoreMLMealTests: XCTestCase {

    private var service: ServiceCoreML!

    override func setUpWithError() throws {
        do {
            service = try ServiceCoreML(language: .russian)
        } catch {
            XCTFail("ServiceCoreML init failed (MealNerRu.mlpackage / vocab.txt missing from test host bundle?): \(error)")
            throw error
        }
    }

    override func tearDown() {
        service = nil
        super.tearDown()
    }

    // MARK: - Helpers

    private func assertDish(
        _ dish: DishEntities,
        foodStem: String,
        quantity: Double?,
        unitStem: String?,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertTrue(
            dish.foodName.lowercased().contains(foodStem),
            "foodName \"\(dish.foodName)\" should contain \"\(foodStem)\"",
            file: file, line: line
        )
        XCTAssertEqual(dish.quantity, quantity, "quantity for \"\(dish.foodName)\"", file: file, line: line)
        if let unitStem {
            XCTAssertTrue(
                dish.unit?.lowercased().contains(unitStem) ?? false,
                "unit \"\(dish.unit ?? "nil")\" should contain \"\(unitStem)\" for \"\(dish.foodName)\"",
                file: file, line: line
            )
        } else {
            XCTAssertNil(dish.unit, "unit for \"\(dish.foodName)\"", file: file, line: line)
        }
    }

    // MARK: - Multi-dish

    func test_coreML_meal_twoDishes() async throws {
        let meal = try await service.parseMeal(transcript: "курица 150 грамм и гречка 80 грамм")
        XCTAssertNil(meal.title)
        XCTAssertNil(meal.dateText)
        XCTAssertNil(meal.timeText)
        XCTAssertEqual(meal.dishes.count, 2)
        guard meal.dishes.count == 2 else { return }
        assertDish(meal.dishes[0], foodStem: "кур", quantity: 150, unitStem: "грамм")
        assertDish(meal.dishes[1], foodStem: "гречк", quantity: 80, unitStem: "грамм")
    }

    func test_coreML_meal_dateAndTitle() async throws {
        let meal = try await service.parseMeal(transcript: "вчера на ужин курица 150 грамм и гречка 80 грамм")
        XCTAssertEqual(meal.dateText, "вчера")
        XCTAssertEqual(meal.title, "ужин")
        XCTAssertNil(meal.timeText)
        XCTAssertEqual(meal.dishes.count, 2)
        guard meal.dishes.count == 2 else { return }
        assertDish(meal.dishes[0], foodStem: "кур", quantity: 150, unitStem: "грамм")
        assertDish(meal.dishes[1], foodStem: "гречк", quantity: 80, unitStem: "грамм")
    }

    func test_coreML_meal_quantityBeforeFood_andTime() async throws {
        let meal = try await service.parseMeal(transcript: "150 грамм курицы 80 грамм гречки в семь вечера")
        XCTAssertEqual(meal.timeText, "семь вечера")
        XCTAssertEqual(meal.dishes.count, 2)
        guard meal.dishes.count == 2 else { return }
        assertDish(meal.dishes[0], foodStem: "кур", quantity: 150, unitStem: "грамм")
        assertDish(meal.dishes[1], foodStem: "гречк", quantity: 80, unitStem: "грамм")
    }

    func test_coreML_meal_weekdayTitleAndMissingQuantity() async throws {
        let meal = try await service.parseMeal(transcript: "в понедельник на завтрак овсянка и яйца 2 штуки")
        XCTAssertEqual(meal.dateText, "понедельник")
        XCTAssertEqual(meal.title, "завтрак")
        XCTAssertEqual(meal.dishes.count, 2)
        guard meal.dishes.count == 2 else { return }
        assertDish(meal.dishes[0], foodStem: "овсян", quantity: nil, unitStem: nil)
        assertDish(meal.dishes[1], foodStem: "яйц", quantity: 2, unitStem: "шт")
    }

    func test_coreML_meal_capitalizedFirstLetter() async throws {
        let meal = try await service.parseMeal(transcript: "Курица 150 грамм")
        XCTAssertEqual(meal.dishes.count, 1)
        guard let dish = meal.dishes.first else { return }
        assertDish(dish, foodStem: "кур", quantity: 150, unitStem: "грамм")
    }

    // MARK: - Error cases

    func test_coreML_meal_emptyTranscript_throwsEmptyTranscript() async {
        await assertParseMealThrows("   ", .emptyTranscript)
    }

    private func assertParseMealThrows(
        _ transcript: String,
        _ expected: ParsingError,
        file: StaticString = #filePath,
        line: UInt = #line
    ) async {
        do {
            _ = try await service.parseMeal(transcript: transcript)
            XCTFail("parseMeal should throw \(expected)", file: file, line: line)
        } catch {
            XCTAssertEqual(error as? ParsingError, expected, file: file, line: line)
        }
    }

    func test_coreML_meal_command_throwsNoEntitiesRecognized() async {
        await assertParseMealThrows("на перекус напомни про тренировку", .noEntitiesRecognized)
    }

    func test_coreML_meal_smallTalk_throwsNoEntitiesRecognized() async {
        await assertParseMealThrows("привет как дела", .noEntitiesRecognized)
    }

    // MARK: - Wordpiece regression

    func test_coreML_meal_oneWordFood_notSplitIntoTwoDishes() async throws {
        let meal = try await service.parseMeal(transcript: "гречка 80 грамм")
        XCTAssertEqual(meal.dishes.count, 1)
        guard let dish = meal.dishes.first else { return }
        XCTAssertTrue(dish.foodName.lowercased().contains("гречк"), dish.foodName)
        XCTAssertFalse(dish.foodName.contains(" "), "foodName should be decoded from subwords: \(dish.foodName)")
    }
}
