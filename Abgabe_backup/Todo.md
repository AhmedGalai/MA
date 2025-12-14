# Master's Thesis - Abgabe Directory TODO

**Project**: Augmented Reality-Enhanced Programming by Demonstration: 6D Pose Estimation Using Apple Vision Pro for Intuitive Robot Teaching
**Author**: Ahmed Galai
**Date**: November 25, 2025
**Status**: Final Submission Preparation

---

## Overview

This document tracks the current status of the thesis submission directory, identifies mismatches between documentation and implementation, and outlines next steps for completion.

---

## Directory Structure Status

### ✅ Completed

```
Abgabe/
├── latex/                          # LaTeX thesis source
│   └── Masterarbeit/
│       ├── main.tex               # Main thesis document
│       ├── main.pdf               # Compiled PDF (if present)
│       ├── content/               # Chapter files
│       ├── figures/               # Images and diagrams
│       └── sources.bib            # Bibliography
│
├── src/                           # Source code (organized)
│   ├── Kubuntu/                   # Linux backend
│   │   ├── backend/              # Python API server
│   │   ├── models/               # 3D mesh models (.ply)
│   │   ├── docs/                 # Documentation
│   │   └── README.md             # Kubuntu-specific guide
│   │
│   ├── MacOS/                     # macOS backend
│   │   ├── backend/              # Python API server
│   │   ├── models/               # 3D mesh models (.ply)
│   │   ├── docs/                 # Documentation
│   │   ├── start.sh              # Startup script
│   │   └── README.md             # MacOS-specific guide
│   │
│   └── VisionOS/                  # Vision Pro app
│       ├── visionos/             # Swift/visionOS project
│       │   └── PoseOverlayApp/
│       └── README.md             # VisionOS development guide
│
└── Todo.md                        # This file
```

### 📊 Current Statistics

- **Python files**: 94 (backend pipeline, APIs, utilities)
- **Swift files**: 18 (VisionOS application)
- **3D models**: 18 (.ply files, duplicated in Kubuntu/MacOS)
- **Documentation**: 6 markdown files (READMEs)
- **LaTeX chapters**: 10 (introduction, research, design, etc.)

---

## Thesis vs Implementation Mapping

### Chapter 4: System Design

**Documented Components**:
1. ✅ Apple Vision Pro client (visionOS)
2. ✅ AirPlay receiver (external, uxplay)
3. ✅ Main API backend (Python/Flask)
4. ✅ FoundationPose 6D pose estimation API (external collaborator)

**Implementation Status**:
- ✅ Main API: `src/MacOS/backend/final_pipeline/main_api.py`
- ✅ CV Pipeline: `src/MacOS/backend/final_pipeline/pipeline_core.py`
- ✅ RealSense depth: `src/MacOS/backend/final_pipeline/realsense_depth.py`
- ✅ VisionOS app: `src/VisionOS/visionos/PoseOverlayApp/`

**Mismatches**:
- ⚠️ **External dependencies not included**:
  - FoundationPose API (collaborator's Docker container)
  - UxPlay (AirPlay receiver, system package)
  - Depth estimation models (HuggingFace endpoints)
- ⚠️ **Calibration data not included**: ArUco calibration matrices
- ⚠️ **Legacy pipelines included**: `full_python_pipeline_backup_v*` directories

---

### Chapter 5: Development

**Documented Features**:
1. ✅ Mock API integration → Real API integration
2. ✅ Main window (model selection, depth mode)
3. ✅ ROI window (radius, color filter)
4. ✅ Debug window (logs, diagnostics)
5. ✅ Immersive view (3D pose overlay)

**Implementation Status**:
- ✅ All UI components present in Swift code
- ✅ Network services implemented (`PoseService`, `ModelService`, `HeadPoseService`)
- ✅ Matrix conversion utilities (`MatrixUtils.swift`)
- ✅ Concurrency via Swift async/await

**Mismatches**:
- ⚠️ **Mock API code still present**: Legacy endpoints in some backend files
- ⚠️ **Python prototype included**: Desktop TkInter UI (not documented as deliverable)
- ⚠️ **Debug utilities scattered**: Multiple `tk_debugging*.py` files

---

### Chapter 6: Evaluation

**Documented Evaluation**:
- Technical performance metrics (latency, accuracy)
- Usability assessment (user tasks, feedback)
- Lighting/occlusion robustness tests

**Implementation Status**:
- ❌ **Evaluation data NOT included**:
  - Raw experiment data
  - Latency measurements
  - User study results
  - Test scenarios/scripts

**Action Required**:
- Decide if evaluation data should be added to `Abgabe/`
- If yes: Create `evaluation/` directory with datasets

---

### Appendix

**Documented**:
- Occlusion handling strategies (EKF, PF, GP-based)
- Multi-camera stereo depth projection
- Willems' Fundamental Lemma (data-driven correction)

**Implementation Status**:
- ⚠️ **Partially implemented**:
  - Pose filtering hooks exist in API (`filter` parameter)
  - Actual filter modules NOT in current codebase
  - Simulation code NOT included

**Note**: Appendix describes design/theory, not necessarily deployed code

---

## Known Issues & Mismatches

### 1. Multiple Pipeline Versions

**Issue**: Backend contains 4+ pipeline implementations:
- `backend/final_pipeline/` ← **Current production**
- `backend/full_python_pipeline/` ← Legacy
- `backend/full_python_pipeline_backup/` ← Legacy
- `backend/full_python_pipeline_backup_v2/` ← Legacy
- `backend/full_python_pipeline_backup_v3/` ← Legacy

**Impact**: Confusing for code review, increases directory size

**Recommendation**:
- ✅ Keep `final_pipeline/` (documented in thesis)
- ⚠️ Consider removing backup versions OR
- 📁 Move to `archive/` directory with explanation

---

### 2. Missing External Dependencies

**Issue**: External components mentioned in thesis but not included:

1. **FoundationPose API** (collaborator's backend)
   - Thesis: Section 4.2
   - Source: [matchcow_pose_api]
   - Status: Dockerized, hosted separately

2. **AnyDepth / ZoeDepth** (monocular depth models)
   - Thesis: Section 4.3.3
   - Source: HuggingFace endpoints
   - Status: Cloud-based inference

3. **UxPlay** (AirPlay receiver)
   - Thesis: Section 5.4
   - Source: GitHub (FDH2/UxPlay)
   - Status: System-level installation

**Recommendation**:
- Document in `External_Dependencies.md`
- Provide links/references
- Clarify what is/isn't included in submission

---

### 3. Platform-Specific Code Duplication

**Issue**: Kubuntu and MacOS directories contain identical backend code

**Current State**:
- `src/Kubuntu/backend/` == `src/MacOS/backend/` (except `start.sh`)
- Models duplicated: `src/Kubuntu/models/` == `src/MacOS/models/`

**Impact**:
- Redundancy (increases size)
- Maintenance burden (need to sync changes)

**Alternatives**:
1. **Keep as-is**: Clear separation for platform-specific deployment
2. **Shared backend**: `src/backend/` + platform-specific startup scripts
3. **Symlinks**: Link common files (doesn't work for all filesystems)

**Current Choice**: Keep separated (chosen for thesis submission clarity)

---

### 4. Missing Startup/Deployment Documentation

**Issue**: No unified getting-started guide at top level

**What's Missing**:
- Quick start for reviewers
- Dependencies overview
- System architecture diagram (as image)
- Build/run instructions summary

**Recommendation**:
- Create `README.md` in `Abgabe/` root
- Reference platform-specific READMEs
- Add architecture diagram from thesis (Figure 4.1)

---

### 5. VisionOS Build Artifacts

**Issue**: Xcode project may contain build artifacts

**Check**:
```bash
cd src/VisionOS/visionos/PoseOverlayApp
ls -la | grep -E "DerivedData|xcuserdata|build"
```

**Recommendation**:
- Remove before submission:
  - `*.xcuserdata/` (user-specific)
  - `DerivedData/` (build cache)
  - `build/` (compiled binaries)
- Keep: `.xcodeproj`, source files, `Packages/`

---

### 6. Calibration and Configuration Files

**Issue**: Some config files may contain environment-specific values

**Check**:
- `backend/app_config.py` - May have hardcoded IPs
- `backend/final_pipeline/config.py` - Paths may be absolute
- `.env` files (if any) - Should not include secrets

**Recommendation**:
- Use placeholder values (e.g., `API_HOST = "192.168.1.XXX"`)
- Include `.env.example` templates
- Document in READMEs

---

## Next Steps

### Priority 1: Critical for Submission

- [ ] **Compile final PDF**:
  ```bash
  cd latex/Masterarbeit
  pdflatex main.tex
  bibtex main
  pdflatex main.tex
  pdflatex main.tex
  ```

- [ ] **Create root README.md**:
  - Project overview
  - Directory structure
  - Quick start guide
  - References to detailed READMEs

- [ ] **Clean VisionOS build artifacts**:
  ```bash
  cd src/VisionOS/visionos/PoseOverlayApp
  rm -rf DerivedData build
  find . -name "*.xcuserdata" -exec rm -rf {} +
  ```

- [ ] **Document external dependencies**:
  - Create `External_Dependencies.md`
  - List FoundationPose API, UxPlay, depth models
  - Provide links and versions

---

### Priority 2: Recommended Improvements

- [ ] **Archive legacy pipelines**:
  ```bash
  mkdir -p archive
  mv src/Kubuntu/backend/full_python_pipeline_backup* archive/
  mv src/MacOS/backend/full_python_pipeline_backup* archive/
  ```

- [ ] **Add system architecture diagram**:
  - Export Figure 4.1 from thesis as PNG
  - Include in root README.md

- [ ] **Verify all file paths**:
  - Search for hardcoded absolute paths
  - Replace with relative or configurable paths

- [ ] **Add `.gitignore` (if using git)**:
  ```
  __pycache__/
  *.pyc
  .DS_Store
  venv/
  *.log
  DerivedData/
  xcuserdata/
  ```

---

### Priority 3: Optional Enhancements

- [ ] **Include evaluation data** (if approved):
  - Create `evaluation/` directory
  - Latency measurements CSV
  - User study responses
  - Test scenario descriptions

- [ ] **Add video demonstrations**:
  - System setup walkthrough
  - Usage demonstration
  - Pose estimation in action
  - Store in `media/` or link to external hosting

- [ ] **Create unified requirements.txt**:
  - Consolidate Python dependencies
  - Pin versions for reproducibility
  - Document CUDA/PyTorch versions

- [ ] **Write troubleshooting FAQ**:
  - Common issues during setup
  - Network connectivity problems
  - Platform-specific gotchas

---

## Testing Checklist

Before final submission, verify:

### Backend (Kubuntu/MacOS)

- [ ] `start.sh` executes without errors
- [ ] Virtual environment creates successfully
- [ ] Dependencies install from requirements.txt
- [ ] Main API starts on port 5000
- [ ] `/health` endpoint responds
- [ ] `/models` endpoint lists .ply files
- [ ] Sample model selection works

### VisionOS App

- [ ] Xcode project opens without errors
- [ ] Code compiles for visionOS target
- [ ] Signing configured (or team placeholder)
- [ ] Info.plist has required permissions
- [ ] No hardcoded IP addresses in code
- [ ] README accurately describes build steps

### Documentation

- [ ] All READMEs render correctly (Markdown syntax)
- [ ] Code blocks have proper syntax highlighting
- [ ] File paths are accurate
- [ ] Cross-references work (e.g., "see Section 4.2")
- [ ] No broken links to external resources

---

## File Size Considerations

### Current Estimates

- LaTeX + PDF: ~10-20 MB
- Python backend: ~5 MB (excluding venv)
- Swift/VisionOS: ~2-3 MB (excluding build artifacts)
- 3D models (.ply): ~5-10 MB (18 files × ~500KB each)
- **Total**: ~25-40 MB

### If Size is Issue

Reduce by:
1. Removing backup pipeline versions (saves ~3-5 MB)
2. Keeping only essential .ply models (saves ~5-8 MB)
3. Compressing PDF (if very large)
4. Excluding high-res figures from LaTeX (use lower DPI)

---

## Submission Format Notes

Depending on university requirements:

### Digital Submission (ZIP/USB)

```
Abgabe.zip
├── latex/                  # Thesis source
├── main.pdf               # Compiled thesis (copy to root)
├── src/                   # Source code
├── README.md              # Top-level guide
├── Todo.md                # This file
└── External_Dependencies.md
```

### Printed Submission

- Print `main.pdf`
- Include USB stick with full `Abgabe/` directory
- Or: Upload to university portal

### Git Repository (if allowed)

- Initialize git repo
- Add `.gitignore`
- Commit organized structure
- Tag as `submission-v1.0`
- Push to GitLab/GitHub (if approved)

---

## Contact & References

**Author**: Ahmed Galai
**Matriculation Number**: 10007404
**Institution**: Karlsruhe Institute of Technology (KIT)
**Institute**: Institute for Anthropomatics and Robotics (IAR)
**Thesis Type**: Master's Thesis

**Supervisors**:
- (List supervisors as per thesis cover page)

**External Collaborators**:
- FoundationPose API: [matchcow_pose_api] (referenced in thesis)
- Streamlit frontend: [matchcow_serp_frontend] (referenced in thesis)

---

## Version History

| Date       | Version | Changes                          |
|------------|---------|----------------------------------|
| 2025-11-25 | 1.0     | Initial organization complete    |
| 2025-11-25 | 1.1     | Added platform-specific READMEs  |
| 2025-11-25 | 1.2     | Created comprehensive Todo.md    |

---

## Appendix: Automated Checks

### Quick Validation Script

```bash
#!/bin/bash
# validate_submission.sh

echo "=== Abgabe Directory Validation ==="

# Check main components exist
echo -n "LaTeX source... "
[ -f "latex/Masterarbeit/main.tex" ] && echo "✓" || echo "✗ MISSING"

echo -n "Compiled PDF... "
[ -f "latex/Masterarbeit/main.pdf" ] && echo "✓" || echo "⚠ Not compiled"

echo -n "Kubuntu backend... "
[ -d "src/Kubuntu/backend" ] && echo "✓" || echo "✗ MISSING"

echo -n "MacOS backend... "
[ -d "src/MacOS/backend" ] && echo "✓" || echo "✗ MISSING"

echo -n "VisionOS app... "
[ -d "src/VisionOS/visionos" ] && echo "✓" || echo "✗ MISSING"

# Check READMEs
echo -n "Platform READMEs... "
COUNT=$(find src -name "README.md" | wc -l)
echo "$COUNT found"

# Check for build artifacts
echo -n "Build artifacts... "
ARTIFACTS=$(find src -name "DerivedData" -o -name "*.xcuserdata" | wc -l)
[ "$ARTIFACTS" -eq 0 ] && echo "✓ Clean" || echo "⚠ $ARTIFACTS found (should remove)"

# Count source files
echo "=== Source File Counts ==="
echo "Python files: $(find src -name "*.py" | wc -l)"
echo "Swift files: $(find src -name "*.swift" | wc -l)"
echo "3D models: $(find src -name "*.ply" | wc -l)"

echo ""
echo "Validation complete. Review any ✗ or ⚠ items above."
```

Run with:
```bash
chmod +x validate_submission.sh
./validate_submission.sh
```

---

## Final Notes

This Todo.md serves as:
1. **Submission checklist** for author
2. **Navigation guide** for reviewers
3. **Known issues** documentation
4. **Future work** tracker

**Last Updated**: November 25, 2025
**Status**: Ready for final review and compilation
