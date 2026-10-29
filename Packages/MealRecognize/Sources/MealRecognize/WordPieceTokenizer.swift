//
//  WordPieceTokenizer.swift
//  MealRecognize
//

import Foundation

/// BERT-style WordPiece tokenizer. Language-agnostic: the vocabulary comes from the caller.
final class WordPieceTokenizer {

    struct Encoding {
        let inputIds: [Int32]
        let attentionMask: [Int32]
        let wordTokens: [String]
    }

    private static let unkToken = "[UNK]"
    private static let padToken = "[PAD]"

    private let vocab: [String: Int32]
    private let unkTokenID: Int32
    private let clsTokenID: Int32
    private let sepTokenID: Int32
    private let padTokenID: Int32

    init(vocabularyURL: URL) throws {
        let contents = try String(contentsOf: vocabularyURL, encoding: .utf8)
        var loadedVocab: [String: Int32] = [:]
        loadedVocab.reserveCapacity(32_768)

        for (index, line) in contents.split(whereSeparator: \.isNewline).enumerated() {
            loadedVocab[String(line)] = Int32(index)
        }

        guard
            let unkTokenID = loadedVocab[Self.unkToken],
            let clsTokenID = loadedVocab["[CLS]"],
            let sepTokenID = loadedVocab["[SEP]"],
            let padTokenID = loadedVocab[Self.padToken]
        else {
            throw TokenizerError.invalidVocabulary
        }

        self.vocab = loadedVocab
        self.unkTokenID = unkTokenID
        self.clsTokenID = clsTokenID
        self.sepTokenID = sepTokenID
        self.padTokenID = padTokenID
    }

    /// Always returns exactly `maxLength` ids and mask entries: `[CLS]`, at most `maxLength - 2` word pieces,
    /// `[SEP]`, then `[PAD]`. Longer text is truncated silently, on purpose: a spoken meal phrase is short,
    /// and losing a tail is better than failing. A word can take several word pieces, so the limit is in
    /// pieces, not words.
    func encode(_ text: String, maxLength: Int) throws -> Encoding {
        let basicTokens = basicTokenize(text)
        var allWordTokens: [String] = []
        allWordTokens.reserveCapacity(basicTokens.count)

        for basicToken in basicTokens {
            allWordTokens.append(contentsOf: wordPieceTokenize(basicToken))
        }

        let maxWordTokens = max(0, maxLength - 2)
        let wordTokens = Array(allWordTokens.prefix(maxWordTokens))

        var inputIds: [Int32] = [clsTokenID]
        inputIds.reserveCapacity(maxLength)
        inputIds.append(contentsOf: wordTokens.map { vocab[$0] ?? unkTokenID })
        inputIds.append(sepTokenID)

        if inputIds.count < maxLength {
            inputIds.append(contentsOf: Array(repeating: padTokenID, count: maxLength - inputIds.count))
        }

        let realTokenCount = min(wordTokens.count + 2, maxLength)
        var attentionMask = Array(repeating: Int32(0), count: maxLength)
        for index in 0..<realTokenCount {
            attentionMask[index] = 1
        }

        return Encoding(
            inputIds: inputIds,
            attentionMask: attentionMask,
            wordTokens: wordTokens
        )
    }

    private func basicTokenize(_ text: String) -> [String] {
        let cleaned = text
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)

        var tokens: [String] = []
        var current = ""

        for character in cleaned {
            if character.isWhitespace {
                if !current.isEmpty {
                    tokens.append(current)
                    current = ""
                }
                continue
            }

            if isPunctuation(character) {
                if !current.isEmpty {
                    tokens.append(current)
                    current = ""
                }
                tokens.append(String(character))
            } else {
                current.append(character)
            }
        }

        if !current.isEmpty {
            tokens.append(current)
        }

        return tokens
    }

    /// Same rule as BERT's `_is_punctuation`: every ASCII character that is not a letter, digit or space
    /// (this includes symbols such as `$`, `+`, `=`), plus the Unicode `P*` categories. Other Unicode symbols
    /// (`№`, emoji) are not punctuation and stay inside the word.
    private func isPunctuation(_ character: Character) -> Bool {
        character.unicodeScalars.allSatisfy { scalar in
            switch scalar.value {
            case 33...47, 58...64, 91...96, 123...126:
                return true
            default:
                switch scalar.properties.generalCategory {
                case .connectorPunctuation, .dashPunctuation, .openPunctuation, .closePunctuation,
                     .initialPunctuation, .finalPunctuation, .otherPunctuation:
                    return true
                default:
                    return false
                }
            }
        }
    }

    private func wordPieceTokenize(_ token: String) -> [String] {
        if vocab[token] != nil {
            return [token]
        }

        var pieces: [String] = []
        var start = token.startIndex

        while start < token.endIndex {
            var end = token.endIndex
            var found: String?

            while start < end {
                var piece = String(token[start..<end])
                if start != token.startIndex {
                    piece = "##\(piece)"
                }
                if vocab[piece] != nil {
                    found = piece
                    break
                }
                end = token.index(before: end)
            }

            if let found {
                pieces.append(found)
                start = end
            } else {
                pieces.append(Self.unkToken)
                break
            }
        }

        return pieces
    }

    enum TokenizerError: Error {
        case invalidVocabulary
    }
}
