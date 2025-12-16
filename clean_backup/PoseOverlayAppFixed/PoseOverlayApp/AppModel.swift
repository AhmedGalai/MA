//
//  AppModel.swift
//  PoseOverlay
//
//  Created by match on 30.10.25.
//

import SwiftUI

/// Maintains app-wide state that needs to be shared between windows/immersive spaces.
@MainActor
final class AppModel: ObservableObject {
    /// Raw string typed by the user for the Python API host.
    @Published var baseURLString: String {
        didSet { baseURL = AppModel.normalizeURL(from: baseURLString) }
    }

    /// Sanitized URL derived from `baseURLString`.
    @Published private(set) var baseURL: URL?

    /// Persisted model list and selection (mirrors `/models` + `/select_model`).
    @Published var availableModels: [String] = []
    @Published var selectedModel: String?
    @Published var isLoadingModels = false
    @Published var lastModelError: String?

    /// Useful for coordinating immersive-space buttons across windows.
    @Published var immersiveSpacePresented = false

    init(defaultBaseURL: String = "http://127.0.0.1:8000") {
        self.baseURLString = defaultBaseURL
        self.baseURL = AppModel.normalizeURL(from: defaultBaseURL)
    }

    func updateBaseURL(_ raw: String) {
        baseURLString = raw
        baseURL = AppModel.normalizeURL(from: raw)
    }

    func setSelectedModel(_ name: String?) {
        selectedModel = name
    }

    func setAvailableModels(_ names: [String]) {
        availableModels = names.sorted()
    }

    func setImmersiveSpacePresented(_ flag: Bool) {
        immersiveSpacePresented = flag
    }

    private static func normalizeURL(from raw: String) -> URL? {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        return URL(string: trimmed)
    }
}
