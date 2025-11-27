# Missing Figures and Diagrams

This document lists all missing figures, diagrams, and visual content identified in the thesis.

## Critical Priority (Must Have Before Submission)

### 1. CV Pipeline Diagram
**Location:** design.tex:247
**Placeholder Text:** `Placeholder: CV pipeline diagram (AirPlay RGB → intrinsics → ROI mask → depth → FoundationPose)`
**Description:** Complete computer vision pipeline flow diagram showing:
- AirPlay RGB frame acquisition
- Intrinsics estimation (ArUco-based)
- ROI mask reconstruction (HSV filtering)
- Depth estimation (RealSense/Transformer fallback)
- FoundationPose integration
- Data flow between components

**Suggested Visual Elements:**
- Block diagram with arrows showing data flow
- Input/Output data types for each stage
- Optional/alternative paths (e.g., depth fallback)
- Processing times for each stage

**File to Create:** `figures/cv_pipeline_flow.pdf` or `.png`

---

## High Priority (Needed for Complete Evaluation)

### 2. Latency/Performance Chart
**Location:** evaluation.tex:107
**Placeholder Text:** "Placeholder for latency/performance chart"
**Description:** Quantitative performance metrics visualization showing:
- End-to-end latency breakdown (simulator vs real device)
- Component-wise timing (frame capture, processing, pose estimation, rendering)
- Comparison across test setups (A, B, C)
- Frame rate measurements

**Data Sources:**
- Setup A (simulator): Mock latencies from prototype tests
- Setup B (real device): Actual measurements from final_pipeline with RealSense
- Setup C (motion/stress): Performance under different conditions

**Suggested Chart Types:**
- Stacked bar chart for latency breakdown
- Line chart for frame rate over time
- Box plot for latency distribution

**File to Create:** `figures/performance_latency_chart.pdf`

**Data to Measure/Provide:**
```
Component               Setup A    Setup B    Setup C
---------------------------------------------------------
Frame Acquisition       __ms       __ms       __ms
ArUco Detection         __ms       __ms       __ms
Depth Estimation        __ms       __ms       __ms
Pose API Round-trip     __ms       __ms       __ms
Rendering               __ms       __ms       __ms
---------------------------------------------------------
Total Latency           __ms       __ms       __ms
Frame Rate              __fps      __fps      __fps
```

---

### 3. User Experience Ratings
**Location:** evaluation.tex:163
**Placeholder Text:** "Placeholder for user experience ratings"
**Description:** Visualization of usability assessment results:
- ROI definition ease of use
- Gaze-based interaction naturalness
- UI window layout effectiveness
- Overall system usability

**Data Sources:**
- User testing sessions (if conducted)
- Self-assessment ratings
- Comparison with baseline methods (if applicable)

**Suggested Chart Types:**
- Horizontal bar chart with rating scales (1-5 or 1-10)
- Radar/spider chart for multi-dimensional usability
- Likert scale visualization

**File to Create:** `figures/user_experience_ratings.pdf`

**Data to Provide:**
```
Metric                          Rating (1-5)    Notes
----------------------------------------------------------
Ease of ROI definition          __/5
Gaze interaction naturalness    __/5
Window layout effectiveness     __/5
Pose visualization clarity      __/5
Overall system usability        __/5
```

---

### 4. Pose Overlay Comparison
**Location:** evaluation.tex:233
**Placeholder Text:** "Placeholder for overlay comparison"
**Description:** Visual comparison of AR pose overlays across test scenarios:
- Static conditions (Setup B): Expected good alignment
- Motion conditions (Setup C): Showing jitter/latency effects
- Different objects (cube, ball, cylinder, etc.)

**Suggested Layout:**
- Side-by-side comparison images
- Before/after filtering (if EKF/PF results available)
- Annotated screenshots from Vision Pro

**File to Create:** `figures/pose_overlay_comparison.pdf` or multiple `.png` files

**Images Needed:**
- `figures/overlay_static_good.png` - Successful alignment
- `figures/overlay_static_failed.png` - Failure case
- `figures/overlay_motion_jitter.png` - Motion artifacts
- `figures/overlay_comparison_grid.pdf` - Combined comparison

---

## Medium Priority (Nice to Have)

### 5. Epipolar Geometry Diagram
**Location:** appendix.tex:92
**Placeholder Text:** (Implicit - section describes stereo vision but no figure included)
**Description:** Illustration of classical stereo vision geometry:
- Two camera setup with baseline
- Epipolar lines and epipolar plane
- Disparity calculation
- Triangulation for depth

**Suggested Visual Elements:**
- Two camera frustums with optical centers
- Image planes with corresponding points
- Epipolar lines connecting corresponding points
- Baseline and depth triangle

**File to Create:** `figures/epipolar_geometry.pdf`

**Alternative:** Use existing diagram from cited source (Hartley & Zisserman) with proper attribution

---

### 6. ZoeDepth/Depth-Anything Architecture
**Location:** appendix.tex:141
**Placeholder Text:** (Implicit - section describes monocular depth but no figure)
**Description:** High-level architecture of the depth estimation model:
- Encoder-decoder structure
- Transformer blocks
- Multi-scale feature extraction
- Depth prediction head

**Note:** Implementation uses **Depth-Anything V2**, not ZoeDepth. Update appendix text accordingly.

**Suggested Visual Elements:**
- Block diagram showing model architecture
- Input RGB → Feature extraction → Depth map output
- Key components (ViT encoder, DPT decoder)

**File to Create:** `figures/depth_anything_architecture.pdf`

**Alternative:**
- Use official diagram from Depth-Anything paper with citation
- Omit figure and reference the original paper for details

---

## Low Priority (Optional)

### 7. Corrected System Architecture Diagram
**Location:** design.tex (Figure 4.1 exists but may need updates)
**Current Figure:** `figures/AVP.png` / `AVP.drawio.svg`
**Issue:** Based on verification report, some components may need clarification:
- FoundationPose API integration (external vs integrated)
- Actual vs. documented data flow in /avp_pose
- RealSense multi-camera setup (if used)

**Recommendation:** Review existing `AVP.drawio.svg` and update if needed

---

### 8. Occlusion Handling Evaluation
**Location:** evaluation.tex:289-293 (mentioned but no figure)
**Description:** "Qualitative inspection showed that EKF smoothed jitter, PF handled abrupt motions, and GP excelled with structured occlusion patterns."

**Status:** ⚠️ **CAUTION** - Based on codebase audit, PF and GP are NOT implemented. Only Kalman filter exists (not integrated).

**Options:**
1. **If simulation exists:** Add simulation results figure
2. **If no simulation:** Remove claims or mark as "theoretical analysis"
3. **If only Kalman was tested:** Show only EKF results

**Do NOT create fabricated figures for unimplemented features.**

---

### 9. ArUco Board Calibration Setup
**Location:** design.tex:243 (described but no photo/diagram)
**Description:** Physical setup showing:
- 3×4 ArUco marker board
- Marker size (30mm) and separation (10mm)
- Typical positioning relative to Vision Pro

**Suggested Visual:**
- Photo of actual calibration board
- Diagram showing board layout with dimensions
- Example detection result with reprojected corners

**File to Create:** `figures/aruco_calibration_setup.jpg` or `.pdf`

---

### 10. RealSense Multi-Camera Setup
**Location:** design.tex:290-295 (described but not visualized)
**Description:** If multi-camera RealSense setup was used:
- Camera positioning
- Overlapping fields of view
- Coordinate frame relationships

**Status:** Uncertain if this was implemented in final system

**Recommendation:** Only create if actually used in evaluation

---

## Existing Figures Status

### Complete and Available (26 figures):
✅ System architecture (AVP.png)
✅ PbD illustrations (6 figures)
✅ FoundationPose overview (6 figures)
✅ CNNPose/PoseCNN (3 figures)
✅ Other pose estimation methods (4 figures)
✅ Python prototype screenshots (2 figures)
✅ 3D models (plymodels.png)
✅ Institutional branding (2 files)

---

## Summary Statistics

| Priority Level | Count | Status |
|----------------|-------|--------|
| **Critical**   | 1     | Missing |
| **High**       | 3     | Missing |
| **Medium**     | 2     | Missing |
| **Low**        | 3     | Optional/Uncertain |
| **Complete**   | 26    | Available |

**Total Missing:** 6 required figures + 3 optional

---

## Action Items

### Before Submission (Critical Path):
1. **CV Pipeline Diagram** - Create from scratch or use design document
2. **Performance Chart** - Requires actual measurement data
3. **User Experience Ratings** - Requires user study data or self-assessment
4. **Pose Overlay Comparison** - Requires screenshots from Vision Pro testing

### If Time Permits:
5. Epipolar geometry diagram (or cite external source)
6. Depth model architecture (or cite paper)

### DO NOT CREATE:
- Fabricated occlusion filter comparison results (PF/GP not implemented)
- RealSense multi-camera setup (if not used)
- Any figures based on non-existent data

---

## Figure Creation Guidelines

1. **Use consistent visual style** across all figures
2. **High resolution** (300 DPI minimum for raster images)
3. **Vector formats preferred** (PDF, SVG) for diagrams
4. **Clear labels and legends** for all charts
5. **Color-blind friendly** palettes
6. **Match thesis font** (likely Computer Modern or similar)
7. **Include figure captions** in LaTeX, not embedded in images

---

## Next Steps

1. Prioritize Critical and High priority figures
2. Collect actual measurement data for performance charts
3. Conduct user testing or document solo evaluation methodology
4. Capture AR overlay screenshots from Vision Pro
5. Create diagrams using tools like:
   - draw.io (for pipeline diagrams)
   - matplotlib/seaborn (for charts)
   - TikZ/LaTeX (for geometric diagrams)
   - Inkscape (for vector graphics)

---

**Document Status:** Generated during thesis completion audit
**Last Updated:** 2025-11-27
**Related Documents:** MISSING_CONTENT.md, TODO.md
