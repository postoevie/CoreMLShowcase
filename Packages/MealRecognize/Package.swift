// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "MealRecognize",
    platforms: [.iOS(.v16)],
    products: [
        .library(name: "MealRecognize", targets: ["MealRecognize"]),
    ],
    targets: [
        .target(
            name: "MealRecognize",
            resources: [
                // `.process` compiles MealNerRu.mlpackage to MealNerRu.mlmodelc inside the package resource bundle.
                .process("Russian/Resources"),
            ],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .testTarget(
            name: "MealRecognizeTests",
            dependencies: ["MealRecognize"],
            resources: [.copy("Resources/golden_tokenizer.json")],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
    ]
)
