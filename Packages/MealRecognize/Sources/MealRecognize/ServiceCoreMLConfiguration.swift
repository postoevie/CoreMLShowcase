//
//  ServiceCoreMLConfiguration.swift
//  MealRecognize
//

import CoreML

extension ServiceCoreML {

    /// When the model is loaded.
    public enum LoadingPolicy: Sendable {
        /// On the first `prepare()` or `parseMeal` call.
        case lazy
        /// In a background task right after the service is created.
        case eager
    }

    public struct Configuration: Sendable {
        /// `.all` on a device. In the iOS Simulator the GPU path (`.all`, `.cpuAndGPU`) returns all-zero logits for this
        /// FP16 model (no NaN or inf, every label is `O`; cause unknown). `.cpuOnly` and `.cpuAndNeuralEngine` give correct
        /// output there, so the simulator uses `.cpuAndNeuralEngine`. Checked on 2026-10-28: the package tests with the
        /// real model pass in the simulator with this default. Run performance tests on a real device only.
        public static var defaultComputeUnits: MLComputeUnits {
            #if targetEnvironment(simulator)
            .cpuAndNeuralEngine
            #else
            .all
            #endif
        }

        public var language: Language
        public var computeUnits: MLComputeUnits
        public var loading: LoadingPolicy
        /// Drops the cached model on a memory warning. The next call reloads it (fast with the system cache).
        public var unloadsOnMemoryWarning: Bool

        public init(
            language: Language,
            computeUnits: MLComputeUnits = Configuration.defaultComputeUnits,
            loading: LoadingPolicy = .lazy,
            unloadsOnMemoryWarning: Bool = true
        ) {
            self.language = language
            self.computeUnits = computeUnits
            self.loading = loading
            self.unloadsOnMemoryWarning = unloadsOnMemoryWarning
        }
    }
}
