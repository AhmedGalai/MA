# Quick Action Items: Thesis Revisions

**Based on Supervisor Feedback Analysis**
**Date:** November 28, 2025

---

## Critical Issue: Occlusion Filtering Over-Emphasis

### Supervisor's Explicit Guidance:
> "occlusion awareness: I think this topic would be a separate research project in itself, which is why I consider it very optional"
>
> "we would rather have **fewer functions that work very well** than many functions that don't work quite as well"

### Current Problem:
Your thesis dedicates **~250 lines** across multiple chapters to EKF/PF/GP filtering methods that are:
- ❌ Not implemented
- ❌ Only theoretical/simulation-based
- ❌ Contradicting supervisor's "very optional" guidance

---

## Required Changes

### 1. Development.tex (Section 3.6, lines 307-320)

**CURRENT:**
```latex
\subsection{Hooks for Occlusion-Aware Pose Filtering}
[14 lines describing filter integration]
```

**ACTION:** ❌ **DELETE entire subsection** or reduce to 2-3 sentences in conclusion

---

### 2. Evaluation.tex (Section 4.4, lines 238-282)

**CURRENT:**
```latex
\subsection{Simulation-Based Occlusion and Filtering Experiments}
[45 lines with filter models, findings, equations]
```

**ACTION:** ❌ **DELETE entire subsection**
- If simulations were run: Mention briefly in one paragraph as "exploratory work"
- If not run: Remove completely

---

### 3. Design.tex (Section 4.5, lines 471-500)

**CURRENT:**
```latex
\subsection{Occlusion Handling and State Estimation (Design)}
[30 lines discussing strategies]
```

**ACTION:** ✂️ **REDUCE to 3-5 sentences**
- Keep only: "Future extensions could explore occlusion handling via filtering methods (EKF, PF) or multi-camera setups"
- Remove detailed mathematical discussion
- Move detailed content to appendix ONLY if academically necessary

---

### 4. Appendix.tex (lines 149-324)

**CURRENT:**
```latex
\subsection{Stochastic State Estimation and Occlusion Handling}
[175 lines: EKF derivations, PF formulation, Willems' Lemma, GP filtering]
```

**ACTION:** ⚠️ **KEEP but add clear disclaimer at top:**
```latex
\textbf{Note:} The methods described in this appendix represent exploratory
theoretical work and have not been implemented or integrated into the system.
They are documented here as potential directions for future research, as
discussed with the thesis supervisor.
```

**OR** ✂️ **REDUCE to 20-30 lines** with high-level overview only

---

### 5. Conclusion.tex (lines 11, 24)

**CURRENT:**
```latex
Simulation experiments with Extended Kalman Filters, Particle Filters, and
Gaussian Process–based, data-driven models (cf. Appendix~\ref{appendix:willems})
indicated that temporal filtering can mitigate some of these issues, but these
methods have not yet been deployed in the live pipeline.
```

**ACTION:** ✏️ **REPHRASE to de-emphasize:**
```latex
Initial theoretical investigations into temporal filtering methods
(Extended Kalman Filters, Particle Filters) suggest potential
for future pose smoothing, though these remain outside the scope
of the current implementation.
```

---

### 6. README.md

**CURRENT:**
- ❌ No mention of model weights download

**ACTION:** ➕ **ADD to "External Dependencies" section:**

```markdown
### FoundationPose Model Weights (REQUIRED)

⚠️ **The FoundationPose backend requires pre-trained weights not included in this repository.**

**Download:** https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i

**Installation:**
1. Download weights from the link above
2. Extract to `src/MacOS/models/` (or `src/Kubuntu/models/`)
3. Verify files are present before starting the backend

Without these weights, the pose estimation backend will not function.
```

---

## Priority Summary

| Action | File | Priority | Effort |
|--------|------|----------|--------|
| Delete/reduce Section 3.6 | development.tex | 🔴 CRITICAL | 5 min |
| Delete Section 4.4 | evaluation.tex | 🔴 CRITICAL | 5 min |
| Reduce Section 4.5 | design.tex | 🔴 CRITICAL | 10 min |
| Add disclaimer or reduce | appendix.tex | 🟡 MEDIUM | 5 min |
| Rephrase mentions | conclusion.tex | 🟡 MEDIUM | 5 min |
| Add weights instructions | README.md | 🔴 CRITICAL | 5 min |

**Total estimated time: ~35 minutes**

---

## Before/After Word Count

### Estimated Reduction:
- Development.tex: -14 lines (-~200 words)
- Evaluation.tex: -45 lines (-~600 words)
- Design.tex: -25 lines (-~350 words)
- Appendix.tex: -150 lines (-~2000 words) [if heavily reduced]
- **Total: ~3000+ words reduced**

### New Focus:
Instead of ~250 lines on unimplemented filters:
- ✅ ~10 lines brief mention in conclusion/outlook
- ✅ ~20 lines in appendix (optional, for completeness)
- ✅ Emphasis on what actually works

---

## Validation Checklist

After making changes, verify:

- [ ] Search for "EKF" in thesis - should only appear in appendix (if kept) and briefly in outlook
- [ ] Search for "Particle Filter" - same as above
- [ ] Search for "Gaussian Process" in context of filtering - same as above
- [ ] Search for "Willems" - should be minimal or removed
- [ ] Check README has weights download instructions
- [ ] Confirm no claims of "validated in simulation" for unimplemented features
- [ ] Ensure conclusion emphasizes core achievements, not future work

---

## Key Principle (from Supervisor)

> "fewer functions that work very well than many functions that don't work quite as well"

**Your thesis DOES have functions that work very well:**
- ✅ VisionOS app with gaze-based ROI
- ✅ Backend API integration
- ✅ Depth estimation (RealSense + MDE)
- ✅ ArUco calibration
- ✅ Head pose tracking
- ✅ AirPlay workaround for ADP limitations

**Focus on THESE achievements, not on theoretical future work!**

---

## Questions to Ask Yourself

Before submitting revisions:

1. **If a reader asks "what did you implement?"**
   - Can they get a clear answer without wading through unimplemented features?

2. **If your supervisor reads the revised version:**
   - Will they see their guidance reflected?
   - Is occlusion awareness now "very optional" in the presentation?

3. **If a future student wants to reproduce your work:**
   - Can they find the weights download link?
   - Will they waste time looking for filter code that doesn't exist?

---

**Next Steps:**
1. Make the 6 changes above
2. Run validation checklist
3. Rebuild PDF
4. Review for overall balance
5. Submit when satisfied

**Good luck!**
