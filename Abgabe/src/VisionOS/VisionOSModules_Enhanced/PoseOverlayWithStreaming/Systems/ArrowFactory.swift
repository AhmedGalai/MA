import RealityKit
import simd
import SwiftUI
import UIKit

enum ArrowFactory {
    /// Arrow along +Z. Slightly smaller than before.
    static func makeArrow(color: Color) -> Entity {
        let root = Entity()

        let totalLength: Float = 0.35
        let headLength:  Float = 0.10
        let shaftLength: Float = totalLength - headLength
        let shaftRadius: Float = 0.016
        let headRadius:  Float = 0.040

        let shaft = ModelEntity(
            mesh: .generateCylinder(height: shaftLength, radius: shaftRadius),
            materials: [UnlitMaterial(color: UIColor(color))]
        )
        shaft.orientation = simd_quatf(angle: .pi / 2, axis: [1,0,0])
        shaft.position    = [0, 0, shaftLength * 0.5]

        let head = ModelEntity(
            mesh: .generateCone(height: headLength, radius: headRadius),
            materials: [UnlitMaterial(color: UIColor(color))]
        )
        head.orientation = simd_quatf(angle: .pi / 2, axis: [1,0,0])
        head.position    = [0, 0, shaftLength + headLength * 0.5]

        root.addChild(shaft)
        root.addChild(head)
        return root
    }
}
