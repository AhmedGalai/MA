# File Organization and Cleanup Plan

This document provides a detailed plan for organizing platform-specific files and removing unnecessary files to streamline the thesis submission.

---

## Executive Summary

**Current Issues:**
- 204MB of duplicate code and models
- 100% identical codebases in MacOS/ and Kubuntu/ directories
- 4 duplicate copies of 3D models (136MB total)
- 2 backup files in production codebase
- Misleading directory structure (no truly platform-specific code)

**After Cleanup:**
- Single unified Python backend
- One copy of models (34MB)
- Clear separation of concerns
- 51% reduction in total size (204MB → 100MB)

---

## Phase 1: Immediate Cleanup (Safe Operations)

### Step 1.1: Remove Backup Files

**Files to DELETE:**
```bash
rm ./src/MacOS/full_python_pipeline/tk_debugging_unified_backup.py
rm ./src/Kubuntu/full_python_pipeline/tk_debugging_unified_backup.py
```

**Justification:** These are backup files with `_backup` suffix. Should not be in production codebase.

**Impact:** Removes 952 lines of dead code, saves ~50KB

---

### Step 1.2: Create Unified Models Directory

**Current Structure:**
```
src/
├── MacOS/
│   ├── models/ (34MB) ← DELETE
│   └── full_python_pipeline/
│       └── models/ (34MB) ← DELETE
└── Kubuntu/
    ├── models/ (34MB) ← DELETE
    └── full_python_pipeline/
        └── models/ (34MB) ← DELETE
```

**Target Structure:**
```
src/
└── models/ (34MB) ← KEEP ONE COPY
```

**Commands:**
```bash
# Create unified models directory at src level
mkdir -p ./src/models

# Copy models from MacOS (arbitrarily choosing MacOS as source)
cp ./src/MacOS/models/*.ply ./src/models/

# Verify all 9 models present
ls -lh ./src/models/
# Expected: Ball.ply, Banana.ply, cube.ply, cylinder.ply, Football.ply,
#           Power Drill-ply.ply, rectangle.ply, Screw.ply, Spanner-ply.ply
```

**DO NOT DELETE old model dirs yet** - wait until Phase 2 to ensure references updated.

---

### Step 1.3: Identify Platform-Specific Files

**Analysis Complete - Result: NO truly platform-specific Python files found**

**Only candidate:** `screen_capture.py` uses `mss` library
- **Platform:** Initially appears MacOS-only
- **Reality:** `mss` library is cross-platform (Windows, macOS, Linux)
- **Conclusion:** Not actually platform-specific

**Truly Platform-Specific:**
- ✅ `VisionOS/` directory (Swift code for Apple Vision Pro)
  - **Action:** Leave unchanged

**Platform-Agnostic:**
- ❌ All Python code in `MacOS/` and `Kubuntu/`
  - **Action:** Consolidate in Phase 2

---

## Phase 2: Directory Consolidation

### Step 2.1: Create Unified Python Backend

**Target Structure:**
```
src/
├── python_backend/              # NEW unified directory
│   ├── config.py                # Unified config (NEW)
│   ├── main_api.py              # From MacOS/Kubuntu
│   ├── test_system.py
│   ├── select_default_model.py
│   ├── screen_capture.py        # From MacOS
│   ├── start.sh                 # From MacOS
│   ├── requirements.txt         # NEW - consolidated
│   ├── docs/
│   │   ├── ARCHITECTURE.md
│   │   ├── LATEX_GUIDE.md
│   │   ├── PROJECT_STRUCTURE.md
│   │   ├── QUICK_START.md
│   │   └── fullsystem.tex
│   ├── final_pipeline/
│   │   ├── *.py (11 files)
│   │   ├── *.md (6 files)
│   │   └── requirements.txt
│   └── full_python_pipeline/
│       ├── *.py (9 files - minus backup)
│       ├── *.md (3 files)
│       └── requirements.txt
├── models/                      # Unified models (from Step 1.2)
└── VisionOS/                    # Unchanged
```

### Step 2.2: Consolidation Commands

```bash
cd /home/ag/Desktop/MA/Abgabe/src

# Create new unified backend directory
mkdir -p python_backend

# Copy MacOS content to unified backend (using MacOS arbitrarily)
cp -r MacOS/* python_backend/

# Copy MacOS-specific screen_capture.py to root of python_backend
# (Already there from previous copy)

# Remove the now-duplicate model directories from python_backend
rm -rf python_backend/models
rm -rf python_backend/full_python_pipeline/models

# DO NOT DELETE MacOS/Kubuntu directories yet
# First verify everything works with new structure
```

### Step 2.3: Update Path References

**Files to modify** (in `python_backend/`):

1. **main_api.py** - Update model path references
2. **select_default_model.py** - Update model path
3. **full_python_pipeline/main_api.py** - Update model path
4. **final_pipeline/pipeline_core.py** - Update model path

**Required Changes:**

**Before:**
```python
model_path = os.path.join("models", model_name)
# or
model_path = os.path.join("full_python_pipeline/models", model_name)
```

**After:**
```python
# Calculate path relative to src/
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(src_dir, "models", model_name)
```

**Script to update all references:**
```bash
# This will be done in Phase 3 with specific file edits
```

---

## Phase 3: Configuration Consolidation

### Step 3.1: Create Unified Configuration File

**Create:** `src/python_backend/config.py`

See CODEBASE AUDIT REPORT Section 7.2 for complete unified config template.

**Key Features:**
- Single source of truth for all configuration
- Environment variable support
- No hardcoded IPs or ports
- Logical grouping (main_api, pose_api, realsense, aruco, etc.)

### Step 3.2: Remove Old Config Files

**After creating unified config:**

```bash
cd python_backend

# Keep config.py as new unified config
# Remove old configs
rm app_config.py
rm full_python_pipeline/app_config.py
# Note: final_pipeline/config.py will be merged into unified config
```

### Step 3.3: Update Imports

**Files to update** (search for `from app_config import`):

1. `main_api.py`
2. `full_python_pipeline/main_api.py`
3. `test_system.py`
4. Any other files importing config

**Before:**
```python
from app_config import APP_CONFIG
```

**After:**
```python
from config import CONFIG, MAIN_API_URL, POSE_API_URL
```

---

## Phase 4: Code Quality Fixes

### Step 4.1: Fix Bare Except Clauses

**File:** `python_backend/final_pipeline/pose_manager.py`

**Lines 58 and 64:**

**Before:**
```python
try:
    aruco_dict = aruco.getPredefinedDictionary(getattr(aruco, dict_name))
except:
    aruco_dict = aruco.Dictionary_get(getattr(aruco, dict_name))
```

**After:**
```python
try:
    aruco_dict = aruco.getPredefinedDictionary(getattr(aruco, dict_name))
except (AttributeError, TypeError):
    # Fallback for older OpenCV versions (<4.7)
    aruco_dict = aruco.Dictionary_get(getattr(aruco, dict_name))
```

### Step 4.2: Remove Hardcoded IPs and Commented Code

**Already removed in unified config** - No additional action needed if unified config used.

**If updating old files individually:**

**File:** `full_python_pipeline/app_config.py`
- Delete lines with hardcoded `10.145.8.86`
- Remove all commented-out code

### Step 4.3: Address TODOs

**File:** `python_backend/final_pipeline/pose_estimator.py`

**Line 157:**
```python
# TODO: Implement proper point correspondence
```

**Options:**
1. **Option A:** Add issue reference
   ```python
   # NOTE: Point correspondence not yet implemented
   # Currently using centroid-based estimation
   # See: Future work section in thesis (Section 7.1)
   ```

2. **Option B:** Actually implement (if time permits)

3. **Option C:** Remove if not needed

**Recommendation:** Option A - document as future work

**Line 210:**
```python
# TODO: Implement ICP refinement
```

**Same options as above.**

---

## Phase 5: Final Verification and Cleanup

### Step 5.1: Verification Checklist

Before deleting old directories, verify:

- [ ] All model files accessible from new `src/models/` location
- [ ] All API endpoints still work
- [ ] Configuration loads correctly
- [ ] Import statements resolve
- [ ] Tests pass (`test_system.py`)

**Verification Commands:**
```bash
cd python_backend

# Check imports
python3 -c "from config import CONFIG; print(CONFIG)"

# Check model loading
python3 select_default_model.py

# Run system test (if safe in environment)
# python3 test_system.py

# Check Flask app starts
python3 main_api.py --help  # or whatever startup command
```

### Step 5.2: Delete Old Directories

**⚠️ ONLY AFTER VERIFICATION PASSES:**

```bash
cd /home/ag/Desktop/MA/Abgabe/src

# Backup just in case
tar -czf ../src_backup_$(date +%Y%m%d).tar.gz MacOS/ Kubuntu/

# Delete old directories
rm -rf MacOS/
rm -rf Kubuntu/

# Verify VisionOS still intact
ls -la VisionOS/
```

### Step 5.3: Update Documentation

**Files to update:**

1. **Create:** `src/README.md`
   - Explain new structure
   - Point to python_backend/docs/
   - Quick start guide

2. **Update:** `python_backend/docs/PROJECT_STRUCTURE.md`
   - Remove references to MacOS/Kubuntu split
   - Document new unified structure

3. **Update:** `python_backend/docs/QUICK_START.md`
   - Update paths (no more MacOS/ prefix)
   - Update model paths

4. **Update:** Thesis `development.tex` if it mentions directory structure

---

## Phase 6: Thesis Submission Organization

### Step 6.1: Final Directory Structure for Submission

**Target structure:**
```
/home/ag/Desktop/MA/Abgabe/
├── latex/                        # Thesis documentation
│   └── Masterarbeit/
│       ├── Masterarbeit.tex      # Main LaTeX file
│       ├── content/*.tex         # Chapter files
│       ├── figures/*.{png,pdf}   # All figures
│       └── sources.bib           # Bibliography
├── src/                          # Source code
│   ├── README.md                 # Top-level documentation
│   ├── models/                   # 3D models (34MB)
│   ├── python_backend/           # Python backend services
│   │   ├── config.py
│   │   ├── main_api.py
│   │   ├── requirements.txt
│   │   ├── docs/
│   │   ├── final_pipeline/
│   │   └── full_python_pipeline/
│   └── VisionOS/                 # Vision Pro app
│       └── PoseOverlayApp/
├── MISSING_FIGURES.md            # Lists missing figures
├── MISSING_CONTENT.md            # Lists missing content
├── FILE_ORGANIZATION.md          # This file
├── TODO.md                       # Final submission checklist
└── README.md                     # Top-level submission README
```

### Step 6.2: Platform-Specific Runfiles Organization

**Question:** Where should platform-specific runfiles go?

**Answer:** Create platform-specific launch scripts at top level of python_backend:

```
python_backend/
├── run_macos.sh          # MacOS-specific startup
├── run_linux.sh          # Linux-specific startup
├── run_windows.bat       # Windows-specific startup (if needed)
└── config.py             # Shared config
```

**Contents of `run_macos.sh`:**
```bash
#!/bin/bash
# MacOS-specific launcher for AR Pose Estimation Backend

# Check if screen_capture is needed
if command -v AirPlay &> /dev/null; then
    echo "Starting screen capture for AirPlay mirroring..."
    python3 screen_capture.py &
    CAPTURE_PID=$!
fi

# Start main API
python3 main_api.py

# Cleanup
if [ ! -z "$CAPTURE_PID" ]; then
    kill $CAPTURE_PID
fi
```

**Contents of `run_linux.sh`:**
```bash
#!/bin/bash
# Linux-specific launcher for AR Pose Estimation Backend

# Screen capture generally not needed on Linux (no AirPlay)
# Could use alternative screen sharing methods if needed

# Start main API
python3 main_api.py
```

**Existing `start.sh`:** Keep as general-purpose script, or rename to `run_macos.sh`

---

## Phase 7: Create Submission Package

### Step 7.1: Files to Include in Submission

**Essential:**
- ✅ All LaTeX source files
- ✅ All figures (existing + newly created)
- ✅ Complete source code (organized)
- ✅ README files
- ✅ Requirements.txt files

**Optional but Recommended:**
- ✅ Build instructions
- ✅ Configuration examples
- ✅ Architecture documentation

**Exclude:**
- ❌ `__pycache__/` directories
- ❌ `.DS_Store` files
- ❌ `.pyc` files
- ❌ Virtual environments (`venv/`, `env/`)
- ❌ Build artifacts (`Masterarbeit.aux`, `.log`, etc.)
- ❌ Backup files (`*_backup.py`, `*.bak`)
- ❌ Git repository (`.git/`)

### Step 7.2: Create .gitignore / Submission Exclusions

**Create:** `/home/ag/Desktop/MA/Abgabe/.excludefile`

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/

# MacOS
.DS_Store
.AppleDouble
.LSOverride

# LaTeX
*.aux
*.lof
*.log
*.lot
*.fls
*.out
*.toc
*.fmt
*.fot
*.cb
*.cb2
.*.lb
*.bbl
*.bcf
*.blg
*.run.xml
*.synctex.gz

# Backups
*~
*.bak
*_backup.*
*.swp

# IDE
.vscode/
.idea/
*.sublime-*

# Misc
*.log
.env
```

---

## Summary of Changes

### Files to DELETE:
| Path | Reason | Size |
|------|--------|------|
| `src/MacOS/` | Duplicate of python_backend | 68MB |
| `src/Kubuntu/` | Duplicate of python_backend | 68MB |
| `src/MacOS/models/` | Moved to src/models/ | 34MB |
| `src/MacOS/full_python_pipeline/models/` | Moved to src/models/ | 34MB |
| `src/Kubuntu/models/` | Moved to src/models/ | 34MB |
| `src/Kubuntu/full_python_pipeline/models/` | Moved to src/models/ | 34MB |
| `*_backup.py` files | Backup files | 50KB |

**Total Removed:** ~204MB

### Files to CREATE:
| Path | Purpose | Priority |
|------|---------|----------|
| `src/python_backend/` | Unified backend directory | Critical |
| `src/python_backend/config.py` | Unified configuration | Critical |
| `src/README.md` | Top-level documentation | High |
| `src/python_backend/run_macos.sh` | MacOS launcher | Medium |
| `src/python_backend/run_linux.sh` | Linux launcher | Medium |
| `src/python_backend/requirements.txt` | Consolidated deps | High |

### Files to MODIFY:
| Path | Changes | Priority |
|------|---------|----------|
| All Python files | Update model paths to `../models/` | Critical |
| All config imports | Update to use unified config | Critical |
| `pose_manager.py` | Fix bare except clauses | High |
| `pose_estimator.py` | Address TODOs | Medium |
| Thesis `development.tex` | Update directory structure references | Medium |

---

## Implementation Timeline

### Day 1: Safe Cleanup (1-2 hours)
- ✅ Remove backup files
- ✅ Create unified models directory
- ✅ Verify model accessibility

### Day 2: Consolidation (3-4 hours)
- ✅ Create python_backend directory
- ✅ Copy files from MacOS
- ✅ Create unified config
- ✅ Update path references
- ✅ Update imports

### Day 3: Testing (2-3 hours)
- ✅ Run tests
- ✅ Verify API endpoints
- ✅ Check model loading
- ✅ Validate configuration

### Day 4: Final Cleanup (1-2 hours)
- ✅ Delete old directories (after backup)
- ✅ Update documentation
- ✅ Create submission package

**Total Estimated Time:** 7-11 hours

---

## Risk Mitigation

### Backup Strategy:
```bash
# Before any destructive operations:
cd /home/ag/Desktop/MA/Abgabe
tar -czf ../Abgabe_backup_$(date +%Y%m%d_%H%M%S).tar.gz src/ latex/
```

### Rollback Plan:
If consolidation fails:
```bash
# Restore from backup
cd /home/ag/Desktop/MA/
tar -xzf Abgabe_backup_YYYYMMDD_HHMMSS.tar.gz
```

### Testing Checklist:
- [ ] Python imports resolve
- [ ] Configuration loads
- [ ] Models accessible
- [ ] API starts without errors
- [ ] (Optional) Full integration test

---

## Questions for User

1. **Should we keep separate platform directories?**
   - Recommendation: NO - consolidate into python_backend/

2. **Should we implement TODOs or document as future work?**
   - Recommendation: Document as future work

3. **Should we fix code issues now or in thesis documentation?**
   - Recommendation: Fix critical issues (bare excepts, hardcoded IPs), document minor ones

4. **Should we create improved LaTeX copy or edit in place?**
   - Will be addressed in next phase

---

**Document Status:** File organization and cleanup plan complete
**Last Updated:** 2025-11-27
**Related Documents:** MISSING_FIGURES.md, MISSING_CONTENT.md, TODO.md
