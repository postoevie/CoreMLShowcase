//
//  RussianLanguage.swift
//  MealRecognize
//

import Foundation

/// Russian-specific values. Everything language-dependent that is not data lives here.
enum RussianLanguage {

    /// Resource folder of the language, relative to the package resource bundle.
    static let resourceSubdirectory = "Russian/Resources"

    /// Core ML model name (`MealNerRu.mlpackage`, compiled to `MealNerRu.mlmodelc`).
    static let modelName = "MealNerRu"

    /// WordPiece vocabulary file name (`vocab.txt`).
    static let vocabularyName = "vocab"

    /// The vocabulary is lower-case, so input must be lower-cased before tokenization.
    static let lowercasesInput = true
}
