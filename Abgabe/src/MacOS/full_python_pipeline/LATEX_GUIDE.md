# LaTeX Documentation Guide

## Overview

The file `fullsystem.tex` contains comprehensive LaTeX documentation for the complete AR pose estimation system. This document is structured to be included as a chapter in a master's thesis.

## Document Structure

### Main Sections

1. **System Architecture Overview** (§4.1)
   - Three-subsystem architecture
   - Dataflow diagram (placeholder for figure)
   - End-to-end workflow description

2. **Python Backend Pipeline** (§4.2)
   - Unified CV module features
   - Dual depth acquisition modes
   - ArUco marker detection
   - API endpoints

3. **Depth Acquisition Modes** (§4.2.2)
   - **Transformers Mode:** Depth-Anything-V2 monocular estimation
   - **RealSense Mode:** Hardware depth with alignment

4. **RealSense Coordinate Transformation** (§4.2.3)
   - Full mathematical derivation
   - 4-step process: back-projection → transform → project → Z-buffer
   - Extrinsics calibration methods
   - Z-buffering algorithm (pseudocode)

5. **ArUco Board Configuration** (§4.2.4)
   - 3×4 marker grid specification
   - 3D marker layout equations
   - PnP pose estimation workflow

6. **Main API Server** (§4.2.5)
   - All endpoint descriptions
   - Head pose tracking format
   - Model management
   - AVP pose endpoint workflow

7. **Head Pose Correction** (§4.2.6)
   - Quaternion to rotation matrix conversion
   - SE(3) transformation composition
   - Staleness checks

8. **VisionOS Client Application** (§4.3)
   - Application structure
   - Pose service and snapshot retrieval
   - Coordinate system transformations (OpenCV ↔ RealityKit)
   - Immersive space rendering
   - Sensor integration

9. **Integration and Workflow** (§4.4)
   - End-to-end dataflow (3 phases)
   - Concurrency design (backend threads, Swift concurrency)

10. **Performance Analysis** (§4.5)
    - Latency breakdown table
    - Optimization strategies

11. **Configuration and Deployment** (§4.6)
    - Backend configuration
    - Launch sequence (3-4 terminals)

## Key Mathematical Content

### Coordinate Transformations

**Back-projection (Pixel to 3D):**
```latex
\begin{bmatrix} X \\ Y \\ Z \end{bmatrix}_{\text{RS}}
= Z \cdot \mathbf{K}_{\text{RS}}^{-1} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}
```

**Rigid Transformation:**
```latex
\begin{bmatrix} X \\ Y \\ Z \end{bmatrix}_{\text{AVP}}
= \mathbf{R}_{\text{AVP←RS}} \begin{bmatrix} X \\ Y \\ Z \end{bmatrix}_{\text{RS}}
+ \mathbf{t}_{\text{AVP←RS}}
```

**Projection (3D to Pixel):**
```latex
u' = \frac{f'_x \cdot X_{\text{AVP}}}{Z_{\text{AVP}}} + c'_x
```

**Quaternion to Rotation Matrix:**
```latex
\mathbf{R}_{\text{head}} = \begin{bmatrix}
1 - 2(q_y^2 + q_z^2) & 2(q_x q_y - q_w q_z) & 2(q_x q_z + q_w q_y) \\
2(q_x q_y + q_w q_z) & 1 - 2(q_x^2 + q_z^2) & 2(q_y q_z - q_w q_x) \\
2(q_x q_z - q_w q_y) & 2(q_y q_z + q_w q_x) & 1 - 2(q_x^2 + q_y^2)
\end{bmatrix}
```

**Head Pose Correction:**
```latex
\mathbf{T}_{\text{corrected}} = \mathbf{T}_{\text{head}} \cdot \mathbf{T}_{\text{object}}
```

**OpenCV to RealityKit Conversion:**
```latex
\mathbf{C} = \begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & -1 & 0 & 0 \\
0 & 0 & -1 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}
```

### ArUco Board Layout

**Marker Position Mapping:**
```latex
\text{row} = \lfloor i / 4 \rfloor, \quad \text{col} = i \mod 4
```

**Corner Positions:**
```latex
x_0 = \text{col} \cdot (s + d), \quad
y_0 = \text{row} \cdot (s + d)
```
where `s = 0.030m` (marker size), `d = 0.010m` (separation)

## Figure Placeholders

The document includes placeholders for the following figures:

1. **Figure 4.1:** System architecture diagram
   - Should show: Vision Pro ↔ Python Backend ↔ Pose API
   - Include dataflow arrows and component labels

2. **Figure 4.2:** ROI selector
   - Screenshot of Vision Pro circular ROI overlay
   - Gaze point indicator

3. **Figure 4.3:** CAD model overlay
   - AR scene with pose-aligned 3D model
   - Real-world object + virtual overlay

4. **Figure 4.4:** Settings menu
   - Floating panel with model picker, API config

5. **Figure 4.5:** Queue workflow diagram
   - Threaded architecture for Python prototype
   - Show: Snapshot Queue → Mask/ROI Queue → Pose Queue

## Tables

**Table 4.1: Latency Breakdown**
- Columns: Stage | Transformers Mode | RealSense Mode
- Rows: Frame capture, ArUco, Depth, Alignment, Encoding, Network, API, etc.
- Total latency ranges provided

**Table 4.2: Python vs VisionOS Concurrency** (inline)
- Mapping between Python threading and Swift concurrency

## Usage Instructions

### Including in Main LaTeX Document

```latex
\documentclass{report}
% ... preamble ...

\begin{document}
% ... front matter ...

\include{chapters/systemdesign}      % Your Chapter 3
\include{full_python_pipeline/fullsystem}  % This file (Chapter 4)
\include{chapters/evaluation}        % Chapter 5

\end{document}
```

### Required LaTeX Packages

```latex
\usepackage{amsmath}        % For math environments
\usepackage{amssymb}        % For math symbols
\usepackage{graphicx}       % For figures
\usepackage{float}          % For [H] placement
\usepackage{algorithm}      % For algorithms
\usepackage{algorithmic}    % For algorithmic environment
\usepackage{xcolor}         % For colored text
\usepackage{verbatim}       % For code blocks
```

### Compiling

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or use your preferred LaTeX editor (Overleaf, TeXShop, TeXstudio, etc.)

## Customization

### Adding Actual Figures

Replace placeholder boxes like:
```latex
\fbox{\parbox{0.9\linewidth}{System Architecture Diagram Placeholder}}
```

with:
```latex
\includegraphics[width=0.9\linewidth]{figures/system_arch.pdf}
```

### Adjusting Section Numbering

If this is not Chapter 4, modify the `\section{}` commands:
- For Chapter 5: Keep as-is (sections auto-number)
- For Appendix: Use `\section*{}` for unnumbered sections

### Citation Style

The document uses `\cite{}` commands. Ensure your bibliography includes:
- `depth_anything_v2` - Depth-Anything-V2 paper
- `apple_macmini_specs` - Mac Mini specifications
- `apple_entitlements` - Apple developer documentation

Example BibTeX entries:
```bibtex
@misc{depth_anything_v2,
  title={Depth Anything V2},
  author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and others},
  year={2024},
  howpublished={\url{https://github.com/DepthAnything/Depth-Anything-V2}}
}

@online{apple_macmini_specs,
  title={Mac mini (M2, 2023) - Technical Specifications},
  author={Apple Inc.},
  year={2023},
  url={https://support.apple.com/kb/SP884}
}
```

## Consistency with Reference Chapter

The document style mirrors the provided reference chapter (VisionOS development), including:
- Section hierarchy and numbering
- Code block formatting (`\begin{verbatim}`)
- Equation styling (centered, numbered where appropriate)
- Figure/table captions
- Technical terminology (e.g., "visionOS" with monospace font)

## Extending the Document

### Adding New Sections

To add content about additional components:

```latex
\subsubsection{New Component Name}

Description of the component...

\textbf{Key Features:}
\begin{itemize}
    \item Feature 1
    \item Feature 2
\end{itemize}

\paragraph{Technical Details}
Detailed explanation with equations if needed:
\[
\mathbf{x} = \mathbf{A}^{-1}\mathbf{b}
\]
```

### Adding Performance Metrics

Extend Table 4.1 with additional rows for new processing stages.

### Adding API Endpoints

Follow the pattern in §4.2.5:
```latex
\begin{itemize}
    \item \texttt{GET /new\_endpoint} -- Description
\end{itemize}
```

## Quality Checklist

Before submission, verify:

- [ ] All placeholders replaced with actual figures
- [ ] All equations compile without errors
- [ ] Bibliography entries exist for all `\cite{}` commands
- [ ] Table and figure numbering is consistent
- [ ] Code blocks use correct syntax highlighting (if using `listings` package)
- [ ] Cross-references work (`\ref{sec:...}`, `\ref{fig:...}`)
- [ ] Acronyms defined on first use
- [ ] Units consistent (meters, milliseconds, etc.)
- [ ] No orphan/widow lines
- [ ] Page breaks make sense (no single lines on new pages)

## Contact

For questions about this documentation structure:
- Check PROJECT_STRUCTURE.md for code organization
- Check full_python_pipeline/README.md for pipeline details
- Refer to the reference chapter for VisionOS-specific details

---

**Document Version:** 1.0
**Last Updated:** 2025-11-09
**Compatibility:** LaTeX2e, pdfTeX, XeTeX, LuaTeX
