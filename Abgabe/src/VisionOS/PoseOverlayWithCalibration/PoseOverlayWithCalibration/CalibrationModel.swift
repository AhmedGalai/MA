import SwiftUI
import simd

@MainActor
class CalibrationModel: ObservableObject {
    @Published var xTranslation: Float = 0.0
    @Published var yTranslation: Float = 0.0
    @Published var zTranslation: Float = 0.0

    @Published var xRotation: Float = 0.0
    @Published var yRotation: Float = 0.0
    @Published var zRotation: Float = 0.0

    var transform: simd_float4x4 {
        let translation = simd_float4x4(translation: SIMD3<Float>(xTranslation, yTranslation, zTranslation))
        let rotationX = simd_float4x4(rotationX: xRotation * .pi / 180.0)
        let rotationY = simd_float4x4(rotationY: yRotation * .pi / 180.0)
        let rotationZ = simd_float4x4(rotationZ: zRotation * .pi / 180.0)
        return translation * rotationZ * rotationY * rotationX
    }
}
