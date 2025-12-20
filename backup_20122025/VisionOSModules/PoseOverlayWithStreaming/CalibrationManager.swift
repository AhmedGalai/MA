import Foundation
import simd
import SwiftUI

@MainActor
final class CalibrationManager: ObservableObject {
    @Published private(set) var calibrationTransform: simd_float4x4?

    private let storageKey = "poseoverlay.calibrationTransform"

    init() {
        loadFromDefaults()
    }

    func saveCalibration(transform: simd_float4x4) {
        calibrationTransform = transform
        persist()
    }

    func clearCalibration() {
        calibrationTransform = nil
        persist()
    }

    private func persist() {
        guard let calibrationTransform else {
            UserDefaults.standard.removeObject(forKey: storageKey)
            return
        }
        let values: [Float] = [
            calibrationTransform.columns.0.x, calibrationTransform.columns.0.y, calibrationTransform.columns.0.z, calibrationTransform.columns.0.w,
            calibrationTransform.columns.1.x, calibrationTransform.columns.1.y, calibrationTransform.columns.1.z, calibrationTransform.columns.1.w,
            calibrationTransform.columns.2.x, calibrationTransform.columns.2.y, calibrationTransform.columns.2.z, calibrationTransform.columns.2.w,
            calibrationTransform.columns.3.x, calibrationTransform.columns.3.y, calibrationTransform.columns.3.z, calibrationTransform.columns.3.w
        ]
        UserDefaults.standard.set(values, forKey: storageKey)
    }

    private func loadFromDefaults() {
        guard let values = UserDefaults.standard.array(forKey: storageKey) as? [Float],
              values.count == 16 else {
            return
        }
        let m = simd_float4x4(columns: (
            SIMD4<Float>(values[0], values[1], values[2], values[3]),
            SIMD4<Float>(values[4], values[5], values[6], values[7]),
            SIMD4<Float>(values[8], values[9], values[10], values[11]),
            SIMD4<Float>(values[12], values[13], values[14], values[15])
        ))
        calibrationTransform = m
    }
}
