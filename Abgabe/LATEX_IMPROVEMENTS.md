# LaTeX Improvements for Thesis Submission

This document outlines the improvements made to the `latex_improved/` copy and recommendations for easily implementable enhancements.

---

## Overview

A copy of the thesis has been created at `/home/ag/Desktop/MA/Abgabe/latex_improved/` for implementing improvements. The original `latex/` directory remains unchanged as a backup.

---

## Critical Fixes Required (Must Do Before Compilation)

### 1. Add Missing Bibliography Entries

**File to Edit:** `latex_improved/Masterarbeit/sources.bib`

**Missing entries to add:**

```bibtex
@online{apple_entitlements,
    title = {Entitlements | Apple Developer Documentation},
    author = {{Apple Inc.}},
    year = {2024},
    url = {https://developer.apple.com/documentation/bundleresources/entitlements},
    note = {Accessed: 2024-12-15}
}

@misc{transformers-model,
    title = {Transformers: State-of-the-Art Natural Language Processing},
    author = {Wolf, Thomas and Debut, Lysandre and Sanh, Victor and others},
    year = {2020},
    publisher = {Association for Computational Linguistics},
    doi = {10.18653/v1/2020.emnlp-demos.6}
}

@book{ref:hartleyzisserman,
    title = {Multiple View Geometry in Computer Vision},
    author = {Hartley, Richard and Zisserman, Andrew},
    edition = {2nd},
    year = {2004},
    publisher = {Cambridge University Press},
    isbn = {0521540518}
}

@inproceedings{ref:stereo,
    title = {A taxonomy and evaluation of dense two-frame stereo correspondence algorithms},
    author = {Scharstein, Daniel and Szeliski, Richard},
    booktitle = {International Journal of Computer Vision},
    volume = {47},
    number = {1},
    pages = {7--42},
    year = {2002},
    publisher = {Springer}
}

@inproceedings{chang2015shapenet,
    title = {ShapeNet: An Information-Rich 3D Model Repository},
    author = {Chang, Angel X and Funkhouser, Thomas and Guibas, Leonidas and others},
    booktitle = {arXiv preprint arXiv:1512.03012},
    year = {2015}
}

@inproceedings{wang2019normalized,
    title = {Normalized Object Coordinate Space for Category-Level 6D Object Pose and Size Estimation},
    author = {Wang, He and Sridhar, Srinath and Huang, Jingwei and others},
    booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    pages = {2642--2651},
    year = {2019}
}

@inproceedings{lin2017focal,
    title = {Focal Loss for Dense Object Detection},
    author = {Lin, Tsung-Yi and Goyal, Priya and Girshick, Ross and others},
    booktitle = {Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
    pages = {2980--2988},
    year = {2017}
}

@inproceedings{groueix2018atlasnet,
    title = {A Papier-M\^ach\'e Approach to Learning 3D Surface Generation},
    author = {Groueix, Thibault and Fisher, Matthew and Kim, Vladimir G and others},
    booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    pages = {216--224},
    year = {2018}
}

@inproceedings{he2022fs6d,
    title = {FS6D: Few-Shot 6D Pose Estimation of Novel Objects},
    author = {He, Yisheng and Sun, Wei and Huang, Haibin and others},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    pages = {6638--6648},
    year = {2022}
}

@inproceedings{labbe2022megapose,
    title = {MegaPose: 6D Pose Estimation of Novel Objects via Render \& Compare},
    author = {Labb\'e, Yann and Manuelli, Lucas and Mousavian, Arsalan and others},
    booktitle = {Conference on Robot Learning (CoRL)},
    year = {2022}
}
```

**Verification Note:** These citations are plausible but should be verified against actual papers. Use Google Scholar to confirm DOIs, venues, and page numbers.

---

## Recommended Improvements with Placeholders

### 2. Update Evaluation Section with Data Placeholders

**File:** `latex_improved/Masterarbeit/content/evaluation.tex`

**Replace placeholder text at line 107 with:**

```latex
\begin{figure}[H]
    \centering
    \begin{tikzpicture}
        \begin{axis}[
            width=0.9\textwidth,
            height=6cm,
            ybar stacked,
            bar width=15pt,
            xlabel={Test Setup},
            ylabel={Latency (ms)},
            legend style={at={(0.5,-0.2)}, anchor=north, legend columns=3},
            symbolic x coords={Simulator (A), Static (B), Motion (C)},
            xtick=data,
            ymin=0,
            ymajorgrids=true,
            grid style=dashed,
        ]
        \addplot coordinates {(Simulator (A),\_\_) (Static (B),\_\_) (Motion (C),\_\_)};
        \addplot coordinates {(Simulator (A),\_\_) (Static (B),\_\_) (Motion (C),\_\_)};
        \addplot coordinates {(Simulator (A),\_\_) (Static (B),\_\_) (Motion (C),\_\_)};
        \addplot coordinates {(Simulator (A),\_\_) (Static (B),\_\_) (Motion (C),\_\_)};
        \addplot coordinates {(Simulator (A),\_\_) (Static (B),\_\_) (Motion (C),\_\_)};
        \legend{Frame Acquisition, ArUco/Mask, Depth, Pose Estimation, Network RT}
        \end{axis}
    \end{tikzpicture}
    \caption[Prototype performance characteristics]{Latency breakdown across test setups. Values to be measured: Simulator (A): Frame \_\_ ms, ArUco \_\_ ms, Depth \_\_ ms, Pose \_\_ ms, Network \_\_ ms. Static (B): \_\_ / \_\_ / \_\_ / \_\_ / \_\_ ms. Motion (C): \_\_ / \_\_ / \_\_ / \_\_ / \_\_ ms.}
    \label{fig:performanceplot}
\end{figure}
```

**Or simpler table-based placeholder:**

```latex
\begin{table}[H]
    \centering
    \caption[Performance measurements across test setups]{Latency breakdown across test setups (values to be measured).}
    \label{fig:performanceplot}
    \begin{tabular}{lrrr}
        \toprule
        \textbf{Component} & \textbf{Setup A (ms)} & \textbf{Setup B (ms)} & \textbf{Setup C (ms)} \\
        \midrule
        Frame Acquisition       & \_\_     & \_\_     & \_\_ \\
        ArUco Detection         & \_\_     & \_\_     & \_\_ \\
        Mask Reconstruction     & \_\_     & \_\_     & \_\_ \\
        Depth Estimation        & \_\_     & \_\_     & \_\_ \\
        Pose Estimation         & \_\_     & \_\_     & \_\_ \\
        Network Round-Trip      & \_\_     & \_\_     & \_\_ \\
        \midrule
        \textbf{Total Latency}  & \_\_     & \_\_     & \_\_ \\
        \textbf{Frame Rate}     & \_\_ fps & \_\_ fps & \_\_ fps \\
        \bottomrule
    \end{tabular}
\end{table}
```

**Replace placeholder at line 163:**

```latex
\begin{figure}[H]
    \centering
    \begin{tabular}{lc}
        \toprule
        \textbf{Usability Aspect} & \textbf{Rating (1--5)} \\
        \midrule
        Ease of ROI definition & \_\_ / 5 \\
        Gaze interaction naturalness & \_\_ / 5 \\
        Window layout effectiveness & \_\_ / 5 \\
        Pose visualization clarity & \_\_ / 5 \\
        Overall system usability & \_\_ / 5 \\
        \midrule
        \textbf{Average} & \_\_ / 5 \\
        \bottomrule
    \end{tabular}
    \caption[User ratings on usability and intuitiveness]{Internal user ratings on usability and intuitiveness. Values represent averages across \_\_ test sessions with \_\_ participants.}
    \label{fig:uxratings}
\end{figure}
```

**Replace placeholder at line 233:**

```latex
\begin{figure}[H]
    \centering
    \fbox{\parbox[c][5cm][c]{10cm}{\centering
        \textbf{To be replaced with screenshots:}\\[1em]
        (a) Mac Mini box - static overlay\\
        (b) Banana model - motion artifacts\\
        (c) Spanner model - occlusion handling\\[1em]
        Screenshots should show AR overlay from Vision Pro with visible object and 3D arrow.
    }}
    \caption[Pose overlay comparison in different motion conditions]{Pose overlay comparison for different objects and motion patterns. Images to be captured from Vision Pro during evaluation sessions.}
    \label{fig:poseoverlay}
\end{figure}
```

---

### 3. Fix Depth Model Inconsistency

**File:** `latex_improved/Masterarbeit/content/appendix.tex`

**Find Section A.3 (around line 112):**

**Replace title:**
```latex
\subsection{ZoeDepth: Monocular Depth Estimation}
\label{appendix:zoedepth}
```

**With:**
```latex
\subsection{Depth-Anything V2: Monocular Depth Estimation}
\label{appendix:depthanything}
```

**Update content (around lines 112-150) to describe Depth-Anything V2 instead of ZoeDepth:**

**Before:**
```latex
ZoeDepth~\cite{bhat2023zoedepth} is a recent approach to metric monocular depth estimation...
```

**After:**
```latex
Depth-Anything V2 is a foundation model for monocular depth estimation based on a vision transformer (ViT) architecture. The model was trained on large-scale diverse datasets to provide robust depth predictions across different domains and lighting conditions.

\textbf{Key Architecture Components:}
\begin{itemize}
    \item \textbf{Vision Transformer (ViT) Encoder:} Processes input RGB images as patch embeddings, capturing global context through self-attention mechanisms.
    \item \textbf{Dense Prediction Transformer (DPT) Decoder:} Reconstructs fine-grained depth maps from multi-scale transformer features.
    \item \textbf{Metric Depth Head:} Produces calibrated metric depth values rather than relative depth rankings.
\end{itemize}

In this work, the Depth-Anything-V2-base-hf model from Hugging Face Transformers~\cite{transformers-model} was used, providing a balance between accuracy and inference speed on CPU/GPU backends.

The model outputs depth maps $D(u,v)$ where each pixel $(u,v)$ corresponds to a metric distance estimate in meters. These depth maps are then integrated with the CV pipeline for 6D pose estimation when external depth sensors (RealSense) are unavailable.

\textbf{Integration in CV Pipeline:}
The Depth-Anything V2 model serves as a fallback depth estimation method in the full Python pipeline. When configured with \texttt{use\_realsense = False}, the pipeline automatically loads the transformer model and applies it to captured RGB frames, producing aligned depth maps for subsequent pose estimation stages.
```

**Note:** Add proper citation if Depth-Anything V2 paper exists, or reference the Hugging Face model card.

---

### 4. Clarify Occlusion Handling Implementation Status

**File:** `latex_improved/Masterarbeit/content/design.tex`

**Find Section 4.6 (Occlusion Handling, around line 480):**

**Add disclaimer at the beginning of section:**

```latex
\subsection{Occlusion Handling and State Estimation}
\label{subsec:occlusion}

\textbf{Note:} This section describes the \emph{theoretical framework} and \emph{planned implementation} for occlusion-aware pose filtering. The Kalman filter module was implemented in the \texttt{coordinate\_transformer.py} module of the final pipeline, but full integration into the production API and evaluation of Particle Filter (PF) and Gaussian Process (GP) approaches remain as future work.

% Rest of section continues as before...
```

**File:** `latex_improved/Masterarbeit/content/development.tex`

**Find Section 5.7 (around line 309):**

**Update to:**

```latex
\subsection{Hooks for Occlusion-Aware Pose Filtering}
\label{subsec:dev-filtering-hooks}

The \texttt{/avp\_pose} endpoint in the main API was designed with extensibility in mind to support future integration of occlusion-aware filters (EKF, PF, GP). While the current production system uses basic head pose correction via matrix multiplication, the architecture includes hooks for more sophisticated filtering:

\begin{itemize}
    \item A \texttt{KalmanPoseFilter} class was implemented in \texttt{final\_pipeline/coordinate\_transformer.py} with standard predict-update cycles.
    \item The filter accepts object pose and headset pose as inputs and outputs smoothed transformations.
    \item Integration into the \texttt{/avp\_pose} endpoint is planned for future releases, allowing runtime selection of filter types.
\end{itemize}

\textbf{Current Status:} The Kalman filter implementation exists but is not yet integrated into the main API request-response flow. Particle Filter and Gaussian Process approaches described in Appendix~\ref{appendix:willems} were explored theoretically but not implemented in code. Future work should focus on integrating the existing Kalman filter and implementing PF/GP alternatives for comparative evaluation.
```

**File:** `latex_improved/Masterarbeit/content/evaluation.tex`

**Find Section 6.5 (Simulation-Based Occlusion, around line 238):**

**Update subsection title and add disclaimer:**

```latex
\subsection{Theoretical Analysis of Occlusion Handling Approaches}
\label{subsec:occlusion-analysis}

\textbf{Note:} This section presents a \emph{theoretical analysis} of occlusion-aware filtering methods based on the mathematical frameworks described in Appendix~\ref{appendix:willems}. Actual simulation experiments were planned but not completed within the thesis timeline. The analysis below describes the expected behavior of each approach based on their documented properties.

% Continue with existing content, but change "simulation runs" to "theoretical analysis suggests"...
```

**Update line 273 from:**
```latex
Simulation runs with synthetic occlusions and measurement dropouts showed:
```

**To:**
```latex
Based on the theoretical properties of each filter and their documented behavior in literature:
```

---

### 5. Update API Endpoint Documentation

**File:** `latex_improved/Masterarbeit/content/design.tex`

**Find Table 4.4 (around line 170):**

**Change `/external_depth` to `/external_disparity`:**

**Before:**
```latex
/external_depth & POST & Submit externally computed depth or disparity map. \\
```

**After:**
```latex
/external_disparity & POST & Submit externally computed depth or disparity map from RealSense or other sensors. \\
```

**Add missing endpoint:**

```latex
/detected_frame & GET & Retrieve RGB frame with ArUco markers drawn for debugging visualization. \\
```

**Remove non-existent endpoint if present:**
```latex
/mask_debug & GET & ... ← DELETE THIS ROW
```

---

### 6. Fix TODO in Conclusion

**File:** `latex_improved/Masterarbeit/content/conclusion.tex`

**Find line 46:**

**Remove or implement TODO:**

**Option A (Remove):**
```latex
% TODO removed - IMU and RL approaches beyond current scope
```

**Option B (Implement):**
```latex
Additional sensor modalities such as IMU data from the Vision Pro could be integrated into the state estimation pipeline to improve pose predictions during rapid motion or visual occlusions. Furthermore, reinforcement learning approaches could be explored where the reward signal is derived from pose estimation error relative to known ground truth or multi-camera consensus, enabling adaptive tuning of filter parameters in real-time robot teaching scenarios.
```

---

### 7. Add CV Pipeline Diagram Description

**File:** `latex_improved/Masterarbeit/content/design.tex`

**Find line 247:**

**Replace:**
```latex
Placeholder: CV pipeline diagram (AirPlay RGB → intrinsics → ROI mask → depth → FoundationPose)
```

**With description (until figure can be created):**

```latex
\begin{figure}[H]
    \centering
    \begin{tikzpicture}[node distance=1.5cm, auto,
        box/.style={rectangle, draw, minimum height=1cm, minimum width=3cm, align=center}]

        \node[box] (airplay) {AirPlay\\RGB Frame};
        \node[box, below of=airplay] (intrinsics) {Intrinsics\\Estimation};
        \node[box, below of=intrinsics] (roi) {ROI Mask\\Reconstruction};
        \node[box, below of=roi] (depth) {Depth\\Estimation};
        \node[box, below of=depth] (pose) {6D Pose\\Estimation};

        \draw[->] (airplay) -- (intrinsics) node[midway, right] {1080p frame};
        \draw[->] (intrinsics) -- (roi) node[midway, right] {Camera matrix $K$};
        \draw[->] (roi) -- (depth) node[midway, right] {Binary mask};
        \draw[->] (depth) -- (pose) node[midway, right] {Depth map $D(u,v)$};
        \draw[->] (pose) -- ++(0,-1.5) node[below] {4×4 Transform};

        \node[right=2cm of intrinsics, text width=3cm, align=left] {ArUco-based\\calibration};
        \node[right=2cm of roi, text width=3cm, align=left] {HSV filtering\\Color thresholds};
        \node[right=2cm of depth, text width=3cm, align=left] {RealSense or\\Depth-Anything V2};

    \end{tikzpicture}
    \caption[Computer vision pipeline overview]{Simplified overview of the CV pipeline stages from RGB acquisition to 6D pose output. Diagram to be refined with detailed timing and data flow annotations.}
    \label{fig:cvpipeline}
\end{figure}
```

**Note:** Requires `\usepackage{tikz}` and `\usetikzlibrary{positioning, arrows.meta}` in preamble.

---

### 8. Add Epipolar Geometry Diagram Reference

**File:** `latex_improved/Masterarbeit/content/appendix.tex`

**Find line 92 (Stereo Vision section):**

**Replace missing diagram with citation:**

```latex
\begin{figure}[H]
    \centering
    \fbox{\parbox[c][5cm][c]{10cm}{\centering
        \textbf{Epipolar geometry illustration}\\[0.5em]
        Refer to Hartley \& Zisserman~\cite{ref:hartleyzisserman}, Figure 9.1\\[0.5em]
        Shows: Two cameras $C$ and $C'$ with baseline $b$, epipolar lines, and triangulation for depth $Z = f \cdot b / d$.
    }}
    \caption[Epipolar geometry in stereo vision]{Epipolar geometry for stereo depth estimation. Illustration reproduced conceptually from Hartley \& Zisserman (2004).}
    \label{fig:epipolar}
\end{figure}
```

---

### 9. Standardize Dataset Names

**Files:** `latex_improved/Masterarbeit/content/research.tex`

**Find all instances of:**
- "Occluded-LINEMOD"
- "OccludedLINEMOD"
- "YCB-Video"

**Standardize to:**
- "Occluded-LINEMOD" (with hyphen)
- "YCB-Video" (with hyphen)

**Use consistent capitalization throughout.**

---

### 10. Fix Future Access Dates in Bibliography

**File:** `latex_improved/Masterarbeit/sources.bib`

**Find entries with "Accessed: 2025-09-XX":**

**Replace with actual access dates or current date:**

```bibtex
note = {Accessed: 2024-12-15}  % or actual date when accessed
```

---

## Optional Enhancements

### 11. Add Related Work Comparison Table

**File:** `latex_improved/Masterarbeit/content/research.tex`

**Add at end of Section 2.1 (PbD) or 2.2 (6D Pose):**

```latex
\begin{table}[H]
    \centering
    \caption{Comparison of AR-based robot programming approaches}
    \label{tab:relatedwork-comparison}
    \begin{tabular}{lllll}
        \toprule
        \textbf{System} & \textbf{Platform} & \textbf{Pose Method} & \textbf{Interaction} & \textbf{Limitation} \\
        \midrule
        Soares et al.~\cite{soares2021holographic} & HoloLens 2 & Manual annotation & Gesture-based & Manual pose \\
        Lotsaris et al.~\cite{lotsaris2021augmented} & HoloLens 1 & Marker-based & Touch + ROS & Fixed markers \\
        This work & Vision Pro & AI-based (FoundationPose) & Gaze + gesture & ADP constraints \\
        \bottomrule
    \end{tabular}
\end{table}
```

### 12. Add System Requirements Section

**File:** `latex_improved/Masterarbeit/content/development.tex`

**Add new subsection after 5.1:**

```latex
\subsection{System Requirements}
\label{subsec:requirements}

\textbf{Hardware Requirements:}
\begin{itemize}
    \item Apple Vision Pro with visionOS 1.0 or later
    \item Mac with Apple Silicon (M1/M2) or Intel processor
    \item (Optional) Intel RealSense D435 or D455 depth camera
    \item WiFi network for AVP-backend communication
\end{itemize}

\textbf{Software Requirements:}
\begin{itemize}
    \item Xcode 15+ with visionOS SDK
    \item Python 3.8 or later
    \item Key dependencies: Flask, OpenCV, NumPy, PyTorch, Transformers
    \item (Optional) PyRealSense2 for hardware depth
\end{itemize}

See \texttt{src/python\_backend/requirements.txt} for complete Python dependencies.
```

---

## Summary of Changes

| File | Change Type | Priority | Status |
|------|-------------|----------|--------|
| sources.bib | Add 10 missing citations | Critical | Needs verification |
| evaluation.tex | Replace figure placeholders with tables | High | Template provided |
| appendix.tex | Update ZoeDepth → Depth-Anything V2 | High | Template provided |
| design.tex | Add occlusion disclaimer | High | Template provided |
| design.tex | Fix API endpoint names | Medium | Template provided |
| development.tex | Clarify filter implementation status | Medium | Template provided |
| evaluation.tex | Update simulation section | Medium | Template provided |
| conclusion.tex | Remove/implement TODO | Low | Options provided |
| research.tex | Standardize dataset names | Low | Find & replace |
| sources.bib | Fix future access dates | Low | Find & replace |

---

## LaTeX Compilation Checklist

Before compiling:

- [ ] All missing bibliography entries added
- [ ] All `\cite{}` references resolve
- [ ] No "Placeholder:" text visible in PDF
- [ ] All figures referenced in text exist (or removed)
- [ ] All tables have proper captions
- [ ] No TODO comments visible
- [ ] Bibliography compiles without warnings
- [ ] All cross-references resolve (no "??")

**Compilation commands:**
```bash
cd latex_improved/Masterarbeit/
pdflatex Masterarbeit.tex
bibtex Masterarbeit
pdflatex Masterarbeit.tex
pdflatex Masterarbeit.tex
```

---

## Next Steps

1. **Verify all bibliography entries** - Check that papers actually exist
2. **Collect actual measurement data** - Fill in \_\_ placeholders with real values
3. **Create missing figures** - CV pipeline, performance charts
4. **Take AR overlay screenshots** - From Vision Pro evaluation
5. **Proofread all changes** - Ensure consistency and accuracy
6. **Compile and check PDF** - Verify no LaTeX errors
7. **Final review** - Read through improved version completely

---

**Document Status:** LaTeX improvement guide complete
**Last Updated:** 2025-11-27
**Target Directory:** `/home/ag/Desktop/MA/Abgabe/latex_improved/`
