//
//  LogsView.swift
//  PoseOverlayApp
//
//  Created by match on 30.10.25.
//

import SwiftUI
import UIKit

struct LogsView: View {
    @EnvironmentObject private var logs: LogStore

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Logs").font(.title3).bold()
                Spacer()
                Button("Copy") {
                    UIPasteboard.general.string = logs.lines.joined(separator: "\n")
                }
                Button("Clear") { logs.clear() }
            }

            Divider()

            ScrollView {
                LazyVStack(alignment: .leading, spacing: 6) {
                    ForEach(Array(logs.lines.enumerated()), id: \.offset) { _, line in
                        Text(line).font(.system(.body, design: .monospaced)).textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                }
                .padding(.vertical, 6)
            }
        }
        .padding()
    }
}
