import Foundation
import simd

struct MeshData {
    var vertices: [SIMD3<Double>] = []
    var faces: [[Int]] = []
    var edges: [(Int,Int)] = []
    var com: SIMD3<Double> = .zero
    var axisLen: Double = 1.0
}

/// Minimal ASCII PLY parser (verts + faces only). Produces unique edges, COM, axisLen.
func parsePLYAscii(_ data: Data) throws -> MeshData {
    guard let text = String(data: data, encoding: .ascii) else { throw NSError(domain: "ply", code: -10) }
    var lines = text.split(whereSeparator: \.isNewline).map(String.init)
    guard lines.first == "ply" else { throw NSError(domain: "ply", code: -11) }
    guard lines.dropFirst().first?.hasPrefix("format ascii") == true else {
        throw NSError(domain: "ply", code: -12, userInfo: [NSLocalizedDescriptionKey: "Only ASCII PLY supported"])
    }
    var idx = 1
    var vCount = 0, fCount = 0
    while idx < lines.count {
        let l = lines[idx]
        if l.hasPrefix("element vertex") {
            vCount = Int(l.split(separator: " ").last!) ?? 0
        } else if l.hasPrefix("element face") {
            fCount = Int(l.split(separator: " ").last!) ?? 0
        } else if l == "end_header" { idx += 1; break }
        idx += 1
    }
    if vCount <= 0 { throw NSError(domain: "ply", code: -13) }

    // vertices
    var vertices: [SIMD3<Double>] = []
    vertices.reserveCapacity(vCount)
    for _ in 0..<vCount {
        let comps = lines[idx].split(separator: " ").compactMap { Double($0) }
        guard comps.count >= 3 else { throw NSError(domain: "ply", code: -14) }
        vertices.append(SIMD3(comps[0], comps[1], comps[2]))
        idx += 1
    }

    // faces
    var faces: [[Int]] = []
    faces.reserveCapacity(fCount)
    for _ in 0..<fCount {
        let parts = lines[idx].split(separator: " ").map { String($0) }
        let n = Int(parts[0]) ?? 0
        let idxs = parts.dropFirst().prefix(n).compactMap { Int($0) }
        if n == 3 {
            faces.append(idxs)
        } else if n > 3 {
            for k in 1..<(n-1) { faces.append([idxs[0], idxs[k], idxs[k+1]]) }
        }
        idx += 1
    }

    // edges (unique)
    var set = Set<String>()
    var edges: [(Int,Int)] = []
    func key(_ a:Int,_ b:Int)->String { a < b ? "\(a)-\(b)" : "\(b)-\(a)" }
    for f in faces {
        let e01 = key(f[0], f[1])
        let e12 = key(f[1], f[2])
        let e20 = key(f[2], f[0])
        for e in [e01,e12,e20] where !set.contains(e) {
            set.insert(e)
            let parts = e.split(separator: "-").compactMap { Int($0) }
            edges.append((parts[0], parts[1]))
        }
    }

    // COM & axis length
    let com = vertices.reduce(SIMD3<Double>(0,0,0), +) / Double(vertices.count)
    let xs = vertices.map{$0.x}, ys = vertices.map{$0.y}, zs = vertices.map{$0.z}
    let extents = SIMD3(xs.max()!-xs.min()!, ys.max()!-ys.min()!, zs.max()!-zs.min()!)
    let axisLen = max(1e-6, 0.25 * max(extents.x, max(extents.y, extents.z)))

    return MeshData(vertices: vertices, faces: faces, edges: edges, com: com, axisLen: axisLen)
}

