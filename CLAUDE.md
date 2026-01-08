# NMRaster - Project Instructions for Claude

## Overview

NMRaster is a next-generation NMR (Nuclear Magnetic Resonance) analysis platform built with Rust/Tauri backend and React/TypeScript frontend. The core innovation is **simultaneous multi-experiment analysis** using factor graphs and belief propagation to process HSQC, NOESY, and TOCSY spectra together.

**Goal:** Achieve 90%+ automated backbone assignment accuracy by treating NMR analysis as a global optimization problem.

---

## Tech Stack

### Backend (src-tauri/)
- **Rust 1.75+** with Tauri 2.x
- **ndarray/nalgebra** - Matrix operations
- **rustfft** - FFT processing
- **petgraph** - Factor graph for inference
- **rusqlite** - SQLite database with WAL mode
- **ort** - ONNX Runtime for ML inference

### Frontend (src/)
- **React 19** with TypeScript
- **Zustand** - State management
- **Tailwind CSS v4** - Styling
- **Vite 7** - Build tool

---

## Project Structure

```
nmraster/
├── src-tauri/                 # Rust backend
│   ├── src/
│   │   ├── commands/          # Tauri command handlers
│   │   │   ├── spectrum.rs    # load_spectrum_1d, get_spectrum_1d, etc.
│   │   │   ├── assignment.rs  # load_molecule_from_sequence, etc.
│   │   │   ├── analysis.rs    # pick_peaks_1d, integrate_peak
│   │   │   └── database.rs    # CRUD operations
│   │   ├── data/              # Domain types
│   │   │   ├── spectrum.rs    # Spectrum1D, Spectrum2D, Peak
│   │   │   ├── molecule.rs    # Molecule graph (petgraph)
│   │   │   ├── experiment.rs  # ChemicalShift, ExperimentType
│   │   │   └── constraint.rs  # DistanceConstraint, NOE
│   │   ├── processing/        # Signal processing
│   │   │   ├── fft.rs         # FFT, zero-fill, fftshift
│   │   │   ├── phasing.rs     # Phase correction
│   │   │   ├── baseline.rs    # Baseline correction
│   │   │   └── apodization.rs # Window functions
│   │   ├── inference/         # Factor graph inference
│   │   │   ├── factor_graph.rs
│   │   │   ├── belief_propagation.rs
│   │   │   ├── scoring.rs
│   │   │   └── assignment.rs
│   │   ├── db/                # Database layer
│   │   ├── ml/                # ONNX model integration
│   │   ├── state/             # AppState (thread-safe)
│   │   └── error.rs           # Error types
│   └── Cargo.toml
├── src/                       # React frontend
│   ├── components/
│   │   ├── spectrum/          # SpectrumCanvas, SpectrumViewer, PeakList
│   │   └── molecule/          # SequenceViewer
│   ├── stores/                # Zustand stores
│   │   ├── spectrumStore.ts
│   │   ├── assignmentStore.ts
│   │   └── uiStore.ts
│   ├── types/tauri.ts         # TypeScript types for Tauri commands
│   ├── lib/tauri.ts           # Type-safe invoke wrappers
│   └── App.tsx
├── masterplan.md              # Technical specification
└── PLAN.md                    # Implementation progress tracker
```

---

## Development Commands

```bash
# Frontend only (fast iteration)
npm run dev

# Full Tauri app (compiles Rust + runs frontend)
npm run tauri dev

# Production build
npm run tauri build

# Type check frontend
npm run build

# Check Rust backend
cargo check --manifest-path src-tauri/Cargo.toml

# Run Rust tests
cargo test --manifest-path src-tauri/Cargo.toml
```

---

## Code Conventions

### Rust
- Use `thiserror` for error types, implement `serde::Serialize` for Tauri returns
- Tauri commands are `async` and take `State<'_, AppState>`
- Use `parking_lot::RwLock` for concurrent state access
- Prefer `ndarray::Array1<f64>` for spectrum data
- All Tauri commands in `src-tauri/src/commands/` module

### TypeScript/React
- Types for Tauri commands in `src/types/tauri.ts` - keep in sync with Rust structs
- Tauri invoke wrappers in `src/lib/tauri.ts` - never call `invoke()` directly elsewhere
- State management via Zustand stores in `src/stores/`
- Components use Tailwind classes, custom utilities in `src/index.css`
- Canvas rendering for spectrum visualization (not WebGL)

### Adding a New Tauri Command
1. Define Rust handler in appropriate `src-tauri/src/commands/*.rs`
2. Register in `src-tauri/src/lib.rs` invoke_handler
3. Add TypeScript types to `src/types/tauri.ts`
4. Add wrapper function to `src/lib/tauri.ts`

---

## Domain Context (NMR Terminology)

**Spectrum Types:**
- **1D** - Single frequency dimension (e.g., 1H proton)
- **2D HSQC** - H-N correlation, backbone fingerprint
- **2D NOESY** - Through-space correlations (<5Å), distance constraints
- **2D TOCSY** - Through-bond correlations, spin systems

**Key Concepts:**
- **Chemical Shift (ppm)** - Resonance frequency, characteristic of atom environment
- **Peak** - Local maximum in spectrum, corresponds to atom(s)
- **Assignment** - Mapping peaks to atoms in the molecule
- **Spin System** - Group of J-coupled atoms (e.g., one residue's backbone)
- **BMRB** - BioMagResBank, database of typical chemical shifts

**Processing Pipeline:**
1. FID (Free Induction Decay) → FFT → Frequency spectrum
2. Phase correction (ph0, ph1) → Real spectrum
3. Baseline correction → Clean baseline
4. Peak picking → Peak list
5. Assignment → Map peaks to atoms

---

## Key Architecture Decisions

1. **Factor Graph for Assignment:** Peaks are variable nodes, constraints are factor nodes. Belief propagation finds globally consistent assignments.

2. **Hybrid Data Layer:** SQLite for structured data (molecules, shifts), in-memory HashMap for active spectra, Zarr planned for large spectral arrays.

3. **Canvas 2D Rendering:** Simpler than WebGL, sufficient for 1D/2D spectra. May add WebGL for 3D/4D later.

4. **Type-Safe Tauri Bridge:** All command types defined in both Rust and TypeScript. Wrappers ensure type safety at the boundary.

---

## Unified Observation Model

The core architectural principle for NMRaster's inference engine.

### Core Principle

**An observation is a set of correlated chemical shifts with uncertainty.**

Every peak from any experiment (HSQC, TOCSY, NOESY, HNCA, HNCACB, etc.) becomes a unified `Observation`:

```rust
struct Observation {
    id: Uuid,
    dimensions: Vec<ObservedDimension>,  // What nuclei we observed
    intensity: f64,                       // Signal strength
    source_experiment: ExperimentType,    // Metadata only - NOT used for factor logic
}

struct ObservedDimension {
    nucleus: NucleusType,  // H1, C13, N15
    shift: f64,            // ppm value
}
```

### Key Principles

1. **Nucleus-Typed, Not Experiment-Typed**
   - Factors apply based on what nucleus dimensions are present
   - NOT based on which experiment the peak came from
   - Example: "If observation has C13 → apply typing" (not "If HSQC-13C → apply typing")

2. **All Carbons Contribute to Typing**
   - CA from HNCA → typing
   - CB from HNCACB → typing
   - C from HSQC-13C → typing
   - All combined in ONE joint typing factor

3. **All Protons Contribute to Typing**
   - H from any experiment influences residue type scoring
   - C-H pairs scored jointly using KDE densities

4. **Per-Nucleus Tolerances**
   - H1: ~0.03 ppm (not "HSQC tolerance" vs "NOESY tolerance")
   - C13: ~0.4 ppm
   - N15: ~0.4 ppm
   - Tolerances anneal during BP (loose → tight)

5. **Factors Apply by Dimension Presence**

| Factor | Trigger | Effect |
|--------|---------|--------|
| Typing | Has C13 or H1 | KDE scoring against residue types |
| Grouping | Shared nucleus shift | Same shift → same residue |
| Sequential | Has intra/inter flag | CA(i) ↔ CA(i-1) linking |
| Backbone | Has N15 + H1 | One NH per residue |

### Why This Matters

Traditional NMR software processes experiments sequentially:
1. TOCSY → spin systems
2. HSQC-13C → carbon typing
3. NOESY → sequential links

The unified model processes ALL observations SIMULTANEOUSLY in one factor graph. Weak evidence from multiple sources converges to strong assignments.

---

## Testing

### Rust
- Unit tests in each module (`#[cfg(test)]`)
- Property-based tests with `proptest` for numerical code
- Integration tests for database operations

### Frontend
- TypeScript type checking via `tsc`
- Manual testing with "Load Demo Data" button

---

## Reference Files

- `masterplan.md` - Full technical specification with algorithms
- `PLAN.md` - Implementation progress and status
- `src-tauri/src/commands/*.rs` - Tauri command signatures
- `src/types/tauri.ts` - TypeScript type definitions

---

## Common Tasks

### Add a new experiment type
1. Add variant to `ExperimentType` enum in `src-tauri/src/data/spectrum.rs`
2. Update `parse_experiment_type()` in `src-tauri/src/commands/spectrum.rs`

### Add a new processing step
1. Create function in appropriate `src-tauri/src/processing/*.rs`
2. Add Tauri command wrapper if needed
3. Expose in frontend via stores/components

### Modify peak picking
- Algorithm in `src-tauri/src/commands/analysis.rs`
- Parameters type `PeakPickingParams`
- UI in `src/components/spectrum/SpectrumViewer.tsx`

---

## Performance Targets

- Process 2D HSQC (2048x512) in <100ms
- Belief propagation converges within 50 iterations
- Handle 4D datasets up to 1GB
- 90%+ backbone assignment accuracy
