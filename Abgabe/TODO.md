# Master Thesis Completion TODO

This document provides a comprehensive checklist for completing and submitting the master thesis. Items are organized by priority and estimated time requirements.

**Last Updated:** 2025-11-27
**Thesis Title:** Augmented Reality-Enhanced Programming by Demonstration: 6D Pose Estimation Using Apple Vision Pro
**Current Completion:** ~92-95%
**Estimated Time to 100%:** 25-40 hours

---

## Overview of Deliverables

- [ ] **Thesis Documentation** (LaTeX) - 95% complete
- [ ] **Source Code** (Python + Swift) - 100% complete, needs organization
- [ ] **Evaluation Data** - Partially complete, needs measurements
- [ ] **Figures and Diagrams** - 26/32 complete (6 missing)
- [ ] **Bibliography** - Missing 10 entries
- [ ] **Code Organization** - Needs consolidation

---

## Critical Priority (Must Complete Before Submission)

### 1. Fix Bibliography Entries (2-4 hours)

**Status:** CRITICAL - LaTeX won't compile
**Blockers:** None
**Dependencies:** None

**Tasks:**
- [ ] Verify all 10 missing citations exist in literature
  - [ ] `apple_entitlements` - Apple Developer Documentation
  - [ ] `transformers-model` - Hugging Face paper
  - [ ] `ref:hartleyzisserman` - Hartley & Zisserman book
  - [ ] `ref:stereo` - Scharstein & Szeliski stereo vision
  - [ ] `chang2015shapenet` - ShapeNet dataset paper
  - [ ] `wang2019normalized` - NOCS paper
  - [ ] `lin2017focal` - Focal Loss (RetinaNet) paper
  - [ ] `groueix2018atlasnet` - AtlasNet paper
  - [ ] `he2022fs6d` - FS6D paper
  - [ ] `labbe2022megapose` - MegaPose paper
- [ ] Find complete BibTeX entries for each
- [ ] Add to `latex_improved/Masterarbeit/sources.bib`
- [ ] Verify LaTeX compilation succeeds
- [ ] Check all citations appear correctly in PDF

**Resources:**
- Google Scholar for BibTeX export
- ACM Digital Library, IEEE Xplore for conference papers
- ArXiv for preprints
- See `LATEX_IMPROVEMENTS.md` for template entries

---

### 2. Remove or Replace Placeholder Content (1-2 hours)

**Status:** CRITICAL - Unprofessional if left in
**Blockers:** None for removal, data collection for replacement
**Dependencies:** Evaluation data (for replacement)

**Tasks:**
- [ ] **evaluation.tex line 107** - Latency chart
  - **Option A:** Remove figure and update text to "Performance data to be collected"
  - **Option B:** Replace with table placeholder (see LATEX_IMPROVEMENTS.md)
  - **Option C:** Collect actual data and create chart (see Priority 2)
- [ ] **evaluation.tex line 163** - User experience ratings
  - **Option A:** Remove figure
  - **Option B:** Replace with table placeholder
  - **Option C:** Collect data and fill in
- [ ] **evaluation.tex line 233** - Pose overlay comparison
  - **Option A:** Remove figure
  - **Option B:** Note "Screenshots to be captured"
  - **Option C:** Take actual screenshots from Vision Pro
- [ ] **design.tex line 247** - CV pipeline diagram
  - **Option A:** Remove reference
  - **Option B:** Use TikZ placeholder (see LATEX_IMPROVEMENTS.md)
  - **Option C:** Create actual diagram (see Priority 2)
- [ ] **appendix.tex line 92** - Epipolar geometry
  - **Option A:** Reference external source (Hartley & Zisserman)
  - **Option B:** Create diagram (see Priority 3)
- [ ] **appendix.tex line 141** - ZoeDepth/Depth-Anything architecture
  - **Option A:** Reference paper/model card
  - **Option B:** Create diagram (see Priority 3)

**Recommendation:** For critical deadline, use Option A (remove) or Option B (placeholder table) for evaluation figures, Option B (TikZ) for CV pipeline.

---

### 3. Update Occlusion Handling Sections (1-2 hours)

**Status:** HIGH - Major discrepancy between docs and code
**Blockers:** None
**Dependencies:** Code audit (already completed)

**Tasks:**
- [ ] Add disclaimer to `design.tex` Section 4.6
  - [ ] Note that EKF implemented but not integrated
  - [ ] Clarify PF/GP as theoretical exploration
- [ ] Update `development.tex` Section 5.7
  - [ ] Describe current implementation status
  - [ ] Mention `coordinate_transformer.py` Kalman filter
  - [ ] Note lack of integration into main API
- [ ] Update `evaluation.tex` Section 6.5
  - [ ] Change "simulation results" to "theoretical analysis"
  - [ ] Add disclaimer about planned vs completed work
  - [ ] Update language to reflect expected behavior, not measured

**Resources:**
- See `LATEX_IMPROVEMENTS.md` Section 4 for exact text to add
- See `MISSING_CONTENT.md` Issue #3 for details

---

### 4. Fix Depth Model Inconsistency (30 minutes)

**Status:** HIGH - Technical inaccuracy
**Blockers:** None
**Dependencies:** None

**Tasks:**
- [ ] Update `appendix.tex` Section A.3 title
  - Change "ZoeDepth" → "Depth-Anything V2"
- [ ] Rewrite Section A.3 content
  - Describe Depth-Anything V2 architecture
  - Update implementation details
  - Fix citation if available
- [ ] Update any references to ZoeDepth in other chapters
- [ ] Search for "AnyDepth" and update if found

**Resources:**
- See `LATEX_IMPROVEMENTS.md` Section 3
- Depth-Anything V2 model card: https://huggingface.co/depth-anything/Depth-Anything-V2-base-hf

---

### 5. Fix API Endpoint Documentation (30 minutes)

**Status:** MEDIUM-HIGH - Documentation accuracy
**Blockers:** None
**Dependencies:** None

**Tasks:**
- [ ] Update `design.tex` Table 4.4
  - [ ] Change `/external_depth` → `/external_disparity`
  - [ ] Add `/detected_frame` endpoint
  - [ ] Remove `/mask_debug` if present
- [ ] Verify all endpoint names match implementation
- [ ] Update any text referencing changed endpoints

**Resources:**
- See `LATEX_IMPROVEMENTS.md` Section 5
- See System Verification Report for endpoint list

---

### 6. Resolve TODO in Conclusion (15 minutes)

**Status:** MEDIUM - Minor incompleteness
**Blockers:** None
**Dependencies:** None

**Tasks:**
- [ ] Review `conclusion.tex` line 46 TODO comment
- [ ] Choose implementation option:
  - **Option A:** Remove comment
  - **Option B:** Add paragraph about IMU/RL future work
- [ ] Implement chosen option

**Resources:**
- See `LATEX_IMPROVEMENTS.md` Section 6 for text template

---

### 7. Compile LaTeX and Fix Errors (30 minutes)

**Status:** CRITICAL - Must verify before submission
**Blockers:** Tasks 1-6 above
**Dependencies:** All LaTeX fixes above

**Tasks:**
- [ ] Navigate to `latex_improved/Masterarbeit/`
- [ ] Run `pdflatex Masterarbeit.tex`
- [ ] Run `bibtex Masterarbeit`
- [ ] Run `pdflatex Masterarbeit.tex` (twice)
- [ ] Check for compilation errors
- [ ] Verify all citations resolve (no "??" in PDF)
- [ ] Check all cross-references work
- [ ] Review PDF for formatting issues
- [ ] Check that no "Placeholder:" text appears

---

## High Priority (Should Complete If Time Permits)

### 8. Collect Evaluation Data and Create Figures (4-8 hours)

**Status:** HIGH - Strengthens thesis significantly
**Blockers:** Requires running system, Vision Pro access
**Dependencies:** Working system, test environment

**Tasks:**
- [ ] **Performance Measurements:**
  - [ ] Run Setup A (simulator) tests
    - Measure API round-trip times
    - Record frame rates
    - Log component latencies
  - [ ] Run Setup B (static real device) tests
    - Measure end-to-end latency
    - Time AirPlay capture
    - Time depth estimation
    - Time pose estimation
    - Record network round-trip
  - [ ] Run Setup C (motion) tests
    - Same measurements under motion
    - Note any outliers or degradation
  - [ ] Create performance chart (bar chart or table)
    - Use measured values
    - Show breakdown by component
    - Compare across setups

- [ ] **Usability Assessment:**
  - [ ] Conduct informal evaluation sessions
    - Test ROI definition ease
    - Assess gaze interaction
    - Evaluate window layout
    - Rate pose visualization
  - [ ] Collect ratings (1-5 scale) for each aspect
  - [ ] Calculate averages
  - [ ] Create usability ratings table/chart

- [ ] **Pose Overlay Screenshots:**
  - [ ] Capture screenshots from Vision Pro
    - Mac Mini box (static, good alignment)
    - Banana model (motion artifacts)
    - Spanner model (occlusion test)
  - [ ] Annotate screenshots if needed
  - [ ] Create comparison figure

**Tools Needed:**
- Running Python backend
- Vision Pro with app installed
- Screen recording / screenshot capability
- Timing instrumentation in code

**Output:**
- 3 figures for `evaluation.tex`
- Measured data for tables
- Screenshots for pose comparison

---

### 9. Create CV Pipeline Diagram (2-3 hours)

**Status:** MEDIUM-HIGH - Improves clarity
**Blockers:** None
**Dependencies:** None

**Tasks:**
- [ ] Choose diagram tool:
  - TikZ (LaTeX native)
  - draw.io / diagrams.net
  - Inkscape
  - Adobe Illustrator / Affinity Designer
- [ ] Design diagram showing:
  - AirPlay frame acquisition
  - Intrinsics estimation (ArUco)
  - ROI mask reconstruction (HSV)
  - Depth estimation (RealSense / Transformers)
  - 6D pose estimation (FoundationPose)
  - Data formats at each stage
  - Optional: timing annotations
- [ ] Export as PDF or PNG (300+ DPI)
- [ ] Add to `latex_improved/Masterarbeit/figures/`
- [ ] Update `design.tex` line 247 to reference actual figure

**Alternative:** Use TikZ template from `LATEX_IMPROVEMENTS.md` Section 7

---

### 10. Standardize Dataset Names (30 minutes)

**Status:** MEDIUM - Consistency
**Blockers:** None
**Dependencies:** None

**Tasks:**
- [ ] Search `research.tex` for dataset name variations
- [ ] Decide on standard formatting:
  - "YCB-Video" (with hyphen)
  - "Occluded-LINEMOD" (with hyphen)
  - "T-LESS" (check official name)
- [ ] Apply find & replace consistently
- [ ] Verify capitalization matches official dataset names

---

### 11. Fix Future Access Dates in Bibliography (15 minutes)

**Status:** LOW-MEDIUM - Minor error
**Blockers:** None
**Dependencies:** None

**Tasks:**
- [ ] Open `sources.bib`
- [ ] Find all entries with "Accessed: 2025-09-XX"
- [ ] Replace with actual access date or 2024-12-15
- [ ] Verify formatting consistency

---

## Medium Priority (Nice to Have)

### 12. Consolidate Source Code Directories (3-4 hours)

**Status:** MEDIUM - Improves submission quality
**Blockers:** None
**Dependencies:** None

**Tasks:**
- [ ] **Backup current state:**
  ```bash
  cd /home/ag/Desktop/MA/
  tar -czf Abgabe_backup_$(date +%Y%m%d).tar.gz Abgabe/
  ```

- [ ] **Create unified structure:**
  - [ ] Create `src/python_backend/` directory
  - [ ] Copy MacOS content to python_backend
  - [ ] Create `src/models/` directory
  - [ ] Copy models once to src/models/
  - [ ] Update path references in Python files
  - [ ] Test imports and model loading

- [ ] **Remove duplicates:**
  - [ ] Delete `src/MacOS/` (after verification)
  - [ ] Delete `src/Kubuntu/` (after verification)
  - [ ] Delete duplicate model directories

- [ ] **Remove backup files:**
  ```bash
  rm src/python_backend/full_python_pipeline/tk_debugging_unified_backup.py
  ```

- [ ] **Verify everything works:**
  - [ ] Test Python imports
  - [ ] Verify model loading
  - [ ] Check config resolution

**Resources:**
- See `FILE_ORGANIZATION.md` for detailed plan
- See `CODEBASE AUDIT REPORT` for rationale

**Time Savings:** Reduces codebase from 204MB to 100MB

---

### 13. Create Unified Configuration File (1-2 hours)

**Status:** MEDIUM - Code quality improvement
**Blockers:** None
**Dependencies:** Task 12 (code consolidation)

**Tasks:**
- [ ] Create `src/python_backend/config.py`
  - Use template from audit report Section 7.2
  - Add environment variable support
  - Remove hardcoded IPs and ports
- [ ] Remove old config files:
  - `app_config.py` (root)
  - `full_python_pipeline/app_config.py`
  - Merge `final_pipeline/config.py`
- [ ] Update imports in all Python files
- [ ] Test configuration loading
- [ ] Create `.env.example` file

**Resources:**
- See `CODEBASE AUDIT REPORT` Section 7.2 for complete template

---

### 14. Fix Code Quality Issues (1-2 hours)

**Status:** MEDIUM - Professional polish
**Blockers:** None
**Dependencies:** Task 12 (code consolidation)

**Tasks:**
- [ ] Fix bare except clauses
  - File: `final_pipeline/pose_manager.py` lines 58, 64
  - Add specific exception types
- [ ] Address TODOs in code
  - File: `pose_estimator.py` lines 157, 210
  - Either implement or document as future work
- [ ] Remove commented-out code
  - Clean up `app_config.py` files
  - Remove unused imports

**Resources:**
- See `CODEBASE AUDIT REPORT` Sections 2.2-2.4

---

### 15. Add Optional Diagrams to Appendix (2-3 hours)

**Status:** LOW-MEDIUM - Visual clarity
**Blockers:** None
**Dependencies:** None

**Tasks:**
- [ ] **Epipolar Geometry Diagram** (Appendix A.2):
  - Option A: Reference Hartley & Zisserman figure
  - Option B: Create simple TikZ diagram
  - Option C: Find CC-licensed image
- [ ] **Depth-Anything Architecture** (Appendix A.3):
  - Option A: Reference official paper figure
  - Option B: Create high-level block diagram
  - Option C: Use Hugging Face model card image
- [ ] Export and add to `figures/` directory
- [ ] Update appendix.tex references

---

### 16. Create Related Work Comparison Table (30 minutes)

**Status:** LOW - Helpful but not critical
**Blockers:** None
**Dependencies:** None

**Tasks:**
- [ ] Add comparison table to `research.tex`
  - Compare this work vs Soares et al. vs Lotsaris et al.
  - Columns: System, Platform, Pose Method, Interaction, Limitation
- [ ] Highlight unique contributions

**Resources:**
- See `LATEX_IMPROVEMENTS.md` Section 11 for template

---

### 17. Create Top-Level Documentation (1 hour)

**Status:** LOW - Submission completeness
**Blockers:** None
**Dependencies:** None

**Tasks:**
- [ ] Create `/home/ag/Desktop/MA/Abgabe/README.md`
  - Thesis title and author
  - Directory structure overview
  - How to compile LaTeX
  - How to run source code
  - System requirements
- [ ] Create `src/README.md`
  - Code organization
  - Quick start guide
  - Dependencies
- [ ] Update `python_backend/docs/` if needed

---

### 18. Final Proofreading Pass (2-3 hours)

**Status:** MEDIUM - Quality assurance
**Blockers:** All LaTeX changes complete
**Dependencies:** Tasks 1-11 complete

**Tasks:**
- [ ] Read entire thesis PDF start to finish
- [ ] Check for:
  - Typos and grammar errors
  - Inconsistent terminology
  - Broken cross-references
  - Missing figure/table captions
  - Formatting inconsistencies
  - Awkward phrasing
- [ ] Verify all technical claims are accurate
- [ ] Check that abstracts match content
- [ ] Ensure conclusions align with evaluation
- [ ] Verify all acronyms defined in list

**Tools:**
- Grammarly (if available)
- LaTeX spell checker
- PDF reader for review

---

## Low Priority (Future Improvements)

### 19. Enhance Limitations Discussion (30 minutes)

**Status:** LOW - Academic completeness
**Blockers:** None
**Dependencies:** None

**Tasks:**
- [ ] Expand `evaluation.tex` Section 6.6
- [ ] Add more detailed analysis of:
  - ADP restrictions impact
  - AirPlay latency quantification
  - Single-object limitation
  - Network dependency
  - Hardware requirements
  - Generalizability concerns

---

### 20. Add System Requirements Section (30 minutes)

**Status:** LOW - Documentation completeness
**Blockers:** None
**Dependencies:** None

**Tasks:**
- [ ] Add subsection to `development.tex`
- [ ] List hardware requirements
- [ ] List software dependencies
- [ ] Reference requirements.txt

**Resources:**
- See `LATEX_IMPROVEMENTS.md` Section 12

---

### 21. Add Ethical Considerations (30 minutes)

**Status:** LOW - Academic completeness
**Blockers:** None
**Dependencies:** None

**Tasks:**
- [ ] Add brief section to conclusion or evaluation
- [ ] Discuss:
  - Robot safety implications
  - Privacy concerns
  - Accessibility
  - Responsible AI use
- [ ] Keep concise (1-2 paragraphs)

---

## Pre-Submission Checklist

### Documentation Review

- [ ] All missing bibliography entries added and verified
- [ ] LaTeX compiles without errors
- [ ] All citations resolve correctly
- [ ] All cross-references work (no "??")
- [ ] No placeholder text visible in PDF
- [ ] All figures have captions
- [ ] All tables have captions
- [ ] Abstracts (German and English) present and accurate
- [ ] List of abbreviations complete
- [ ] List of symbols complete
- [ ] Acknowledgments section complete (if required)
- [ ] Declaration of authorship signed

### Code Review

- [ ] No backup files in submission
- [ ] No commented-out code in production files
- [ ] All TODOs addressed or documented
- [ ] README files present and accurate
- [ ] requirements.txt files complete
- [ ] No hardcoded passwords/secrets
- [ ] Code is organized logically
- [ ] Build/run instructions work

### Submission Package

- [ ] Create final submission directory
- [ ] Include PDF of thesis
- [ ] Include all LaTeX sources
- [ ] Include all figures
- [ ] Include complete source code
- [ ] Include README at top level
- [ ] Exclude unnecessary files (.DS_Store, __pycache__, etc.)
- [ ] Verify total package size reasonable
- [ ] Create archive (ZIP or TAR.GZ)
- [ ] Test extraction of archive

### Final Verification

- [ ] PDF opens correctly in multiple PDF readers
- [ ] All pages render correctly
- [ ] Figures are high quality (not pixelated)
- [ ] Page numbers correct
- [ ] Table of contents matches actual content
- [ ] Bibliography formatting consistent
- [ ] No broken URLs in bibliography
- [ ] Code compiles/runs on clean system
- [ ] All required forms signed

---

## Time Estimates Summary

| Priority | Tasks | Estimated Time |
|----------|-------|----------------|
| **Critical** | 1-7 | 6-11 hours |
| **High** | 8-11 | 7-12 hours |
| **Medium** | 12-18 | 9-14 hours |
| **Low** | 19-21 | 1.5-3 hours |
| **Pre-submission** | Review & Package | 2-3 hours |

**Total Estimated Time:** 25.5-43 hours

**Minimum for Submission:** Complete all Critical tasks (6-11 hours)

---

## Recommended Completion Order

### Week 1: Critical Fixes (Day 1-2, 6-11 hours)
1. Add missing bibliography entries
2. Remove/replace placeholder content
3. Update occlusion handling sections
4. Fix depth model inconsistency
5. Fix API endpoint documentation
6. Resolve conclusion TODO
7. Compile LaTeX and fix errors

**Milestone:** Thesis compiles and is technically submittable

### Week 2: Data Collection (Day 3-4, 7-12 hours)
8. Collect evaluation data and create figures
9. Create CV pipeline diagram
10. Standardize dataset names
11. Fix bibliography dates

**Milestone:** All figures present, evaluation complete

### Week 3: Code Cleanup (Day 5-6, 9-14 hours)
12. Consolidate source code directories
13. Create unified configuration
14. Fix code quality issues

**Milestone:** Code organized and professional

### Week 4: Polish (Day 7, 3-6 hours)
15-18. Optional enhancements
Final proofreading
Pre-submission checklist

**Milestone:** Thesis ready for submission

---

## Risk Management

### High Risk Items

| Risk | Impact | Mitigation |
|------|--------|------------|
| Missing citations don't exist | Can't compile | Verify all papers before deadline, have backup plan |
| Can't collect evaluation data | Weak evaluation | Use placeholder tables, acknowledge limitations |
| LaTeX won't compile | Can't submit | Start compilation testing early, have backup template |

### Backup Plans

**If running out of time:**
1. **Priority:** Complete only Critical tasks (Tasks 1-7)
2. **If no evaluation data:** Use qualitative descriptions only, remove figure placeholders
3. **If diagrams can't be created:** Use text descriptions or reference external sources
4. **If code can't be cleaned:** Submit as-is with note about duplication

---

## Support Resources

- **Related Documents:**
  - `MISSING_FIGURES.md` - List of all missing figures
  - `MISSING_CONTENT.md` - List of content gaps
  - `FILE_ORGANIZATION.md` - Code reorganization plan
  - `LATEX_IMPROVEMENTS.md` - LaTeX fix templates
  - Audit reports in this directory

- **External Resources:**
  - Google Scholar for citations
  - Hartley & Zisserman book for geometry
  - Hugging Face for model documentation
  - OpenCV documentation for CV pipeline

- **Tools:**
  - LaTeX: Overleaf, TeXStudio, or VS Code with LaTeX Workshop
  - Diagrams: draw.io, TikZ, Inkscape
  - Bibliography: JabRef, Zotero
  - Python: VS Code, PyCharm

---

## Contact and Questions

**Supervisor:** David Wendorff, M.Sc
**Institution:** Institute of Assembly Technology and Industrial Robotics, Leibniz University Hannover

---

**Status:** This TODO list is current as of 2025-11-27.
**Next Review:** After completing Critical tasks
**Final Deadline:** [INSERT YOUR DEADLINE HERE]

---

## Notes and Progress Tracking

Use this space to track your progress:

```
Date: ___________
Tasks Completed:
-
-
-

Blockers Encountered:
-
-

Time Spent: _____ hours
Remaining: _____ hours
```

---

**Good luck with your thesis completion! 🎓**
