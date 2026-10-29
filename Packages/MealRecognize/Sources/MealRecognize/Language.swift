//
//  Language.swift
//  MealRecognize
//

import Foundation

/// Language the recognizer works in. Picks the model, vocabulary and text rules.
public enum Language: Sendable {
    case russian

    var resourceSubdirectory: String {
        switch self {
        case .russian: return RussianLanguage.resourceSubdirectory
        }
    }

    var modelName: String {
        switch self {
        case .russian: return RussianLanguage.modelName
        }
    }

    var vocabularyName: String {
        switch self {
        case .russian: return RussianLanguage.vocabularyName
        }
    }

    var lowercasesInput: Bool {
        switch self {
        case .russian: return RussianLanguage.lowercasesInput
        }
    }

    /// Looks the resource up in the language folder first, then at the root of the resource bundle
    /// (SwiftPM may or may not keep the folder structure of processed resources).
    func resourceURL(name: String, withExtension fileExtension: String) -> URL? {
        Bundle.module.url(forResource: name, withExtension: fileExtension, subdirectory: resourceSubdirectory)
            ?? Bundle.module.url(forResource: name, withExtension: fileExtension)
    }
}
