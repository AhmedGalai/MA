//
//  PoseResponse.swift
//  PoseOverlayApp
//
//  Created by match on 30.10.25.
//

import Foundation
import simd

struct PoseResponse: Decodable {
    let status: String?
    let transformation_matrix: [Matrix4x4DTO]

    private enum CodingKeys: String, CodingKey {
        case status
        case transformation_matrix
        case legacyMatrix = "T"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        status = try container.decodeIfPresent(String.self, forKey: .status)

        if let matrices = try container.decodeIfPresent([Matrix4x4DTO].self, forKey: .transformation_matrix) {
            transformation_matrix = matrices
        } else if let matrices = try container.decodeIfPresent([Matrix4x4DTO].self, forKey: .legacyMatrix) {
            transformation_matrix = matrices
        } else if let single = try container.decodeIfPresent(Matrix4x4DTO.self, forKey: .transformation_matrix) {
            transformation_matrix = [single]
        } else if let single = try container.decodeIfPresent(Matrix4x4DTO.self, forKey: .legacyMatrix) {
            transformation_matrix = [single]
        } else {
            transformation_matrix = []
        }
    }
}

// Accept numbers or numeric strings
struct FlexibleNumber: Decodable {
    let value: Double
    init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if let d = try? c.decode(Double.self) { value = d; return }
        if let s = try? c.decode(String.self),
           let d = Double(s.trimmingCharacters(in: .whitespacesAndNewlines)) { value = d; return }
        throw DecodingError.dataCorruptedError(in: c, debugDescription: "Expected number or numeric string")
    }
}

// 4×4 represented as 4 row arrays
struct Matrix4x4DTO: Decodable {
    let rows: [[FlexibleNumber]]
    init(from decoder: Decoder) throws {
        var container = try decoder.unkeyedContainer()
        var rows: [[FlexibleNumber]] = []
        while !container.isAtEnd {
            rows.append(try container.decode([FlexibleNumber].self))
        }
        guard rows.count == 4, rows.allSatisfy({ $0.count == 4 }) else {
            throw DecodingError.dataCorrupted(.init(codingPath: decoder.codingPath,
                                                    debugDescription: "Matrix must be 4x4"))
        }
        self.rows = rows
    }
}
