import SwiftUI

struct HelpView: View {
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                Text("Pose Overlay Help")
                    .font(.title2)
                    .bold()

                helpSection(title: "Connect") {
                    Text("Set the API host and port in the main window, then tap Connect. The app will start streaming and polling.")
                }

                helpSection(title: "Windows") {
                    Text("Open Anchor Setup to position the world anchor with sliders.")
                    Text("Open ROI Window to place the ROI disk over the object.")
                    Text("Open Debug Viewer for feeds, matrices, and controls.")
                }

                helpSection(title: "ROI") {
                    Text("Use the ROI Color picker and Radius slider in the main window.")
                    Text("The HSV filter uses the same color and variance to create the binary mask.")
                }

                helpSection(title: "Processing Load") {
                    Text("Use the Processing slider in Debug Viewer to reduce load. A stride of 2 means every 2nd frame is processed.")
                }

                helpSection(title: "FoundationPose") {
                    Text("When a pose is available it is shown in 3D with the label \"foundationpose\" and in the AVP overlays.")
                    Text("If no pose is available, the pose defaults to the anchor position and a log message is shown.")
                    Text("Use \"Save next request\" in Debug Viewer to store the next FoundationPose inputs/outputs.")
                }

                helpSection(title: "Debug Streams") {
                    Text("The Debug Viewer mirrors the /debug endpoint. Use Firefox for MJPEG-heavy pages if Safari drops streams.")
                }
            }
            .padding()
        }
    }

    private func helpSection(title: String, @ViewBuilder content: () -> some View) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title)
                .font(.headline)
            content()
                .font(.body)
        }
        .padding()
        .background(.thinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }
}
