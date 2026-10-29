//
//  ContentView.swift
//  CoreMLShowcaseApp
//

import SwiftUI
import MealRecognize

struct ContentView: View {

    private struct Row: Identifiable {
        let id = UUID()
        let title: String
        let date: String
        let time: String
        let dishes: String
    }

    @State private var service: any ServiceProtocol = (try? ServiceCoreML(language: .russian))
        ?? ServiceMock()
    @State private var input = ""
    @State private var rows: [Row] = []
    @State private var isLoading = false
    @State private var message = ""
    @FocusState private var isInputFocused: Bool

    private var trimmedInput: String {
        input.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private var isRecognizeDisabled: Bool {
        trimmedInput.isEmpty || isLoading
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 8) {
                TextField("Введите параметры приема пищи", text: $input)
                    .textFieldStyle(.roundedBorder)
                    .submitLabel(.done)
                    .focused($isInputFocused)
                    .onSubmit(recognize)

                Text("Например: вчера на ужин курица 150 грамм и гречка 80 грамм")
                    .font(.caption)
                    .foregroundStyle(.secondary)

                HStack(spacing: 12) {
                    Button(action: recognize) {
                        Text("Распознать")
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(isRecognizeDisabled)

                    if isLoading {
                        ProgressView()
                            .controlSize(.small)
                    }
                }
                .padding(.top, 8)

                if !message.isEmpty {
                    Text(message)
                        .font(.caption)
                        .foregroundStyle(.red)
                }

                ResultsTable(
                    rows: rows.map { ResultsTable.Item(id: $0.id, title: $0.title, date: $0.date, time: $0.time, dishes: $0.dishes) }
                )
                .padding(.top, 16)
            }
            .padding()
        }
        .scrollDismissesKeyboard(.interactively)
        .background(Color(.systemBackground))
        .task {
            try? await service.prepare()
        }
    }

    private func recognize() {
        let transcript = trimmedInput
        guard !transcript.isEmpty, !isLoading else { return }
        isInputFocused = false
        message = ""
        isLoading = true
        let service = service
        Task { @MainActor in
            do {
                let meal = try await service.parseMeal(transcript: transcript)
                rows.insert(Self.makeRow(from: meal), at: 0)
            } catch is ParsingError {
                message = "Ничего не распознано"
            } catch {
                message = "Не удалось распознать"
            }
            isLoading = false
        }
    }

    private static func makeRow(from meal: MealEntities) -> Row {
        Row(
            title: cellText(meal.title),
            date: cellText(meal.dateText),
            time: cellText(meal.timeText),
            dishes: cellText(dishesText(meal.dishes))
        )
    }

    private static func cellText(_ text: String?) -> String {
        guard let text = text?.trimmingCharacters(in: .whitespacesAndNewlines), !text.isEmpty else { return "—" }
        return text
    }

    private static func dishesText(_ dishes: [DishEntities]) -> String {
        dishes.map { dish in
            var parts = [dish.foodName]
            if let quantity = dish.quantity {
                parts.append(quantityText(quantity))
            }
            if let unit = dish.unit, !unit.isEmpty {
                parts.append(unit)
            }
            return parts.joined(separator: " ")
        }
        .joined(separator: ", ")
    }

    private static func quantityText(_ quantity: Double) -> String {
        if quantity == quantity.rounded(), abs(quantity) < 1_000_000_000 {
            return String(Int(quantity))
        }
        return quantity.formatted()
    }
}

private struct ResultsTable: View {

    struct Item: Identifiable {
        let id: UUID
        let title: String
        let date: String
        let time: String
        let dishes: String
    }

    private enum Width {
        static let short: CGFloat = 90
        static let dishes: CGFloat = 220
    }

    let rows: [Item]

    var body: some View {
        ScrollView(.horizontal, showsIndicators: true) {
            Grid(alignment: .topLeading, horizontalSpacing: 12, verticalSpacing: 8) {
                GridRow {
                    headerCell("Название", width: Width.short)
                    headerCell("Дата", width: Width.short)
                    headerCell("Время", width: Width.short)
                    headerCell("Блюда", width: Width.dishes)
                }
                Divider()
                if rows.isEmpty {
                    GridRow {
                        Text("Пока нет результатов")
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                            .gridCellColumns(4)
                    }
                }
                ForEach(rows) { row in
                    GridRow {
                        cell(row.title, width: Width.short)
                        cell(row.date, width: Width.short)
                        cell(row.time, width: Width.short)
                        cell(row.dishes, width: Width.dishes)
                    }
                    Divider()
                }
            }
        }
    }

    private func headerCell(_ text: String, width: CGFloat) -> some View {
        Text(text)
            .font(.subheadline.bold())
            .foregroundStyle(.secondary)
            .frame(width: width, alignment: .leading)
    }

    private func cell(_ text: String, width: CGFloat) -> some View {
        Text(text)
            .font(.body)
            .foregroundStyle(.primary)
            .fixedSize(horizontal: false, vertical: true)
            .frame(width: width, alignment: .leading)
    }
}

#Preview {
    ContentView()
}
