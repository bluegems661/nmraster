# NMRaster Implementation Plan

A structured roadmap for building the next-generation NMR platform with simultaneous multi-experiment analysis.

**Last Updated:** 2026-01-06
**Status:** Phase 1-3 Backend Complete | Frontend Complete (including Assignment UI)

---

## Progress Overview

| Goal | Status | Progress |
|------|--------|----------|
| Goal 1: Foundation | ✅ Complete | 100% |
| Goal 2: Data Architecture | ✅ Complete | 100% |
| Goal 3: Signal Processing | ✅ Complete | 100% |
| Goal 4: Inference Engine | ✅ Complete | 100% |
| Goal 5: Tauri Commands | ✅ Complete | 100% |
| Goal 6: React Frontend | ✅ Complete | 100% |
| Goal 7: ML Integration | 🔶 Stubs Only | 30% |
| Goal 8: Testing | 🔶 Unit Tests | 40% |
| Goal 9: CRYSTALLINE Density | ⏳ Not Started | 0% |

---

## Goal 1: Foundation Layer ✅

Establish the core Rust/Tauri project structure with essential dependencies and error handling.

### Subgoal 1.1: Project Initialization ✅
- [x] Initialize Tauri 2.0 project with React frontend
- [x] Configure Cargo.toml with all dependencies from masterplan
- [x] Set up TypeScript configuration for frontend
- [x] Configure build scripts and development workflow

**Implementation Notes:**
- Created with `npx create-tauri-app` using react-ts template
- Rust 1.92.0 installed via rustup
- Cargo.toml includes: tauri 2.9, rusqlite 0.32, ndarray 0.16, rustfft 6.2, petgraph 0.6, ort 2.0.0-rc.10

### Subgoal 1.2: Error Handling System ✅
- [x] Create `error.rs` with thiserror error types
- [x] Define domain-specific error variants (spectrum, assignment, database, ML)
- [x] Implement error conversion traits for external crate errors

**Implementation Notes:**
- `NmrError` - top-level enum with From traits
- `SpectrumError` - InvalidDimensions, Fft, PhaseCorrection, etc.
- `DatabaseError` - wraps rusqlite::Error
- `InferenceError` - ConvergenceFailure, InvalidGraph, etc.
- `ModelError` - OnnxRuntime, ModelNotFound, HashMismatch
- Implements `serde::Serialize` for Tauri command returns

### Subgoal 1.3: Application State Management ✅
- [x] Create `state/app_state.rs` with Mutex-wrapped state
- [x] Define state structures for active spectrum, assignments, experiments
- [x] Implement state initialization and cleanup

**Implementation Notes:**
- `AppState` with `Arc<Mutex<Option<Database>>>` for thread-safe DB access
- `RwLock<HashMap<Uuid, T>>` for molecules, spectra, peaks, shifts, constraints
- `with_db()` and `with_db_mut()` helper methods for safe DB access
- Auto-initializes database in Tauri setup

---

## Goal 2: Data Architecture ✅

Build the hybrid data layer combining SQLite for structured data, graph relationships, and Zarr for spectral arrays.

### Subgoal 2.1: SQLite Database Setup ✅
- [x] Implement `db/connection.rs` with WAL mode configuration
- [x] Create `db/migrations.rs` with schema versioning
- [x] Build migration files for all tables from masterplan schema

**Implementation Notes:**
- WAL mode enabled for better concurrency
- Foreign keys enforced
- 5 migrations implemented:
  1. Core tables: molecules, chains, residues, atoms
  2. Experiments and spectra tables
  3. Peaks and assignments tables
  4. Constraint tables (distance, dihedral)
  5. Relaxation data tables (T1, T2, CPMG)
- `schema_migrations` table tracks applied versions

**Tables Created (12 total):**
| Table | Columns | Indexes |
|-------|---------|---------|
| molecules | id, name, sequence_length, sequence | - |
| chains | id, molecule_id, chain_code, polymer_type | molecule_id |
| residues | id, molecule_id, chain_id, sequence_code | molecule_id, chain_id, unique(seq) |
| atoms | id, residue_id, atom_name, element | residue_id |
| experiments | id, name, type, date, frequency | - |
| spectra | id, experiment_id, dimensions, nucleus_types | experiment_id |
| peaks | id, spectrum_id, position, intensity | spectrum_id |
| chemical_shift_lists | id, molecule_id, name | molecule_id |
| chemical_shifts | id, list_id, atom_id, value | list_id, atom_id, value |
| peak_assignments | id, peak_id, dimension, atom_id | peak_id, atom_id |
| distance_constraints | id, set_id, atom1_id, atom2_id | set_id, atoms |
| dihedral_constraints | id, set_id, atoms[4] | set_id |

### Subgoal 2.2: Spectrum Data Structures ✅
- [x] Create `data/spectrum.rs` with ndarray-backed spectrum types
- [x] Implement 1D, 2D, 3D, 4D spectrum representations
- [x] Add metadata structures (nucleus types, sweep widths, offsets)

**Implementation Notes:**
- `Spectrum1D` - real/imag arrays, ppm_axis, from_fid() constructor
- `Spectrum2D` - 2D array with F1/F2 axes, slice methods
- `Spectrum3D`, `Spectrum4D` - defined for future use
- `SpectrumMetadata` - id, name, experiment_type, nucleus_types, sw, offsets
- `Peak` - position, intensity, volume, line_width, SNR, assignments
- `NucleusType` enum with gyromagnetic ratios
- `ExperimentType` enum (HSQC, NOESY, TOCSY, HNCO, etc.)

### Subgoal 2.3: Molecular Graph ✅
- [x] Implement `data/molecule.rs` using petgraph
- [x] Create node types for residues and atoms
- [x] Add edge types for bonds and sequential connectivity
- [x] Implement sequence parsing (one-letter codes)

**Implementation Notes:**
- `Molecule` with `DiGraph<MoleculeNode, MoleculeEdge>`
- `MoleculeNode::Residue` and `MoleculeNode::Atom` variants
- `MoleculeEdge::Bond`, `::Sequence`, `::Contains`
- `from_sequence()` - parses one-letter codes, creates residues + atoms
- `Residue::backbone_atoms()` and `sidechain_atoms()` - returns standard atom names
- `get_atoms_for_residue()`, `next_residue()`, `prev_residue()` navigation
- One-letter to three-letter code conversion for all 20 amino acids

### Subgoal 2.4: Experiment and Constraint Types ✅
- [x] Create `data/experiment.rs` with experiment metadata
- [x] Implement `data/constraint.rs` for distance and dihedral constraints
- [x] Add relaxation data structures (T1, T2, NOE, CPMG)

**Implementation Notes:**
- `Experiment` - name, type, date, frequency, temperature, pH
- `ChemicalShiftList` with `Vec<ChemicalShift>`
- `ChemicalShift` - atom_id, value, error, ambiguity_code, confidence
- `BMRBStatistics` - default_protein_stats() with mean/std for backbone atoms
- `DistanceConstraint` - atom pair, bounds, from_noe(), violation()
- `DihedralConstraint` - 4 atoms, target angle, tolerance
- `NOEIntensityClass` - Strong/Medium/Weak/VeryWeak with distance bounds
- `RelaxationData` and `CPMGDispersion` for dynamics

---

## Goal 3: Signal Processing Pipeline ✅

Implement the FFT and signal processing modules for spectrum transformation.

### Subgoal 3.1: FFT Processing ✅
- [x] Create `processing/fft.rs` with rustfft integration
- [x] Implement forward and inverse FFT for 1D-4D data
- [x] Add zero-filling and interpolation

**Implementation Notes:**
- `fft_1d()` - in-place FFT with fftshift
- `ifft_1d()` - inverse with normalization
- `fft_2d()` - row then column FFT
- `fftshift_1d/2d()` and `ifftshift_1d()` - center zero frequency
- `zero_fill_1d()` and `zero_fill_to_power_of_2()`
- `process_fid_1d()` - complete FID to spectrum pipeline
- Unit tests for delta function, roundtrip, sine wave

### Subgoal 3.2: Phase Correction ✅
- [x] Implement `processing/phasing.rs`
- [x] Add zero-order and first-order phase correction
- [x] Implement automatic phase correction algorithms

**Implementation Notes:**
- `apply_phase_correction()` - ph0/ph1 with pivot point
- `phase_correct_real()` - returns only real part
- `auto_phase_entropy()` - grid search minimizing derivative entropy
- `auto_phase_acme()` - simplex refinement (Nelder-Mead)
- Works in log-space for numerical stability

### Subgoal 3.3: Baseline Correction ✅
- [x] Create `processing/baseline.rs`
- [x] Implement polynomial baseline fitting
- [x] Add spline-based baseline correction

**Implementation Notes:**
- `polynomial_baseline()` - least squares fit to specified regions
- `auto_baseline()` - iterative fitting, removes points above baseline
- `spline_baseline()` - knot-based with local minima detection
- `fit_polynomial()` - Vandermonde matrix with Gaussian elimination
- Configurable degree and iterations

### Subgoal 3.4: Apodization Functions ✅
- [x] Implement `processing/apodization.rs`
- [x] Add window functions: exponential, Gaussian, sine-bell, cosine-bell
- [x] Support shifted windows for resolution enhancement

**Implementation Notes:**
- `exponential_multiply()` - line broadening (lb_hz)
- `gaussian_multiply()` - Gaussian decay
- `lorentz_to_gauss()` - resolution enhancement
- `sine_bell()`, `sine_bell_squared()`, `cosine_bell()`
- `hanning()`, `hamming()`, `blackman()`, `kaiser()`
- `create_window()` - factory function with `WindowType` enum
- `bessel_i0()` - modified Bessel function for Kaiser window

---

## Goal 4: Inference Engine ✅

Build the factor graph and belief propagation system for simultaneous multi-experiment analysis.

### Subgoal 4.1: Factor Graph Construction ✅
- [x] Create `inference/factor_graph.rs` with petgraph-based structure
- [x] Define variable nodes (chemical shifts, assignments)
- [x] Define factor nodes (constraint potentials)

**Implementation Notes:**
- `FactorGraph` with `DiGraph<FactorNode, f64>`
- `FactorNode::Variable { id, domain }` - possible chemical shifts
- `FactorNode::Factor { potential, connected_vars }`
- `FactorPotential` enum:
  - `ChemicalShiftPrior` - mean, std from BMRB
  - `PeakConsistency` - observed peak position, tolerance
  - `SequentialConnectivity` - expected shift difference
  - `NOEDistance` - distance bounds
- `add_chemical_shift_variable()`, `add_bmrb_prior_factor()`, `add_peak_factor()`

### Subgoal 4.2: Belief Propagation ✅
- [x] Implement `inference/belief_propagation.rs`
- [x] Add message initialization (uniform distributions)
- [x] Implement variable-to-factor messages
- [x] Implement factor-to-variable messages
- [x] Add convergence detection

**Implementation Notes:**
- `run_belief_propagation()` - main entry point with max_iterations, tolerance
- `initialize_messages()` - uniform log-space distributions
- `propagate_one_step()` - single iteration, returns max delta
- `compute_message()` - variable→factor and factor→variable
- `compute_factor_message()` - marginalizes potential over other variables
- `normalize_log_message()` - numerical stability
- `get_marginals()` - extracts final probabilities with softmax
- Convergence when max message delta < tolerance

### Subgoal 4.3: Scoring Functions ✅
- [x] Create `inference/scoring.rs`
- [x] Implement BMRB statistics integration
- [x] Add chemical shift consistency scoring
- [x] Implement NOE distance compatibility scoring
- [x] Build sequential connectivity scoring

**Implementation Notes:**
- `score_bmrb_consistency()` - Gaussian probability from BMRB stats
- `score_peak_consistency()` - linear penalty within tolerance
- `score_sequential_connectivity()` - expected shift difference scoring
- `AssignmentScore` - combines bmrb, peak, connectivity scores
- `ScoreWeights` - configurable weights (default: bmrb=1, peak=2, conn=1.5)

### Subgoal 4.4: Global Assignment Algorithm ✅
- [x] Implement `inference/assignment.rs`
- [x] Add FLYA-style cost function
- [x] Integrate evolutionary optimization (optional)
- [x] Build marginal extraction for final assignments

**Implementation Notes:**
- `AtomAssignment` - atom_id, assigned_shift, confidence, alternatives
- `run_global_assignment()` - runs BP then extracts assignments
- Extracts best assignment from marginals
- Reports alternatives above 5% probability threshold
- Unit tests verify assignment to correct domain value

---

## Goal 5: Tauri Commands and API ✅

Create the Tauri command layer bridging Rust backend to React frontend.

### Subgoal 5.1: Spectrum Commands ✅
- [x] Implement `commands/spectrum.rs`
- [x] Add commands: load_spectrum, process_spectrum, get_spectrum_data
- [ ] Support multiple file formats (Bruker, Varian, NMRPipe) - *Future*

**Implemented Commands (5):**
| Command | Parameters | Returns |
|---------|------------|---------|
| `load_spectrum_1d` | name, real[], imag[], sw, offset, freq | spectrum_id |
| `get_spectrum_1d` | id | SpectrumDataResponse |
| `process_spectrum_1d` | id, zero_fill_factor | new_spectrum_id |
| `list_spectra` | - | Vec<SpectrumInfo> |
| `get_spectrum_peaks` | spectrum_id | Vec<PeakInfo> |

### Subgoal 5.2: Assignment Commands ✅
- [x] Create `commands/assignment.rs`
- [ ] Implement `run_global_assignment` command from masterplan - *API ready, not exposed*
- [x] Add commands for manual assignment, validation

**Implemented Commands (8):**
| Command | Parameters | Returns |
|---------|------------|---------|
| `load_molecule_from_sequence` | name, sequence, chain_code | molecule_id |
| `get_active_molecule` | - | MoleculeInfo |
| `get_molecule_residues` | molecule_id | Vec<ResidueInfo> |
| `get_residue_atoms` | molecule_id, seq_code | Vec<AtomInfo> |
| `create_shift_list` | name, molecule_id | list_id |
| `add_chemical_shift` | list_id, atom_id, value, ... | shift_id |
| `get_shifts_for_residue` | list_id, seq_code | Vec<ShiftInfo> |
| `list_shift_lists` | - | Vec<ShiftListInfo> |

### Subgoal 5.3: Database Commands ✅
- [x] Build `commands/database.rs`
- [x] Add CRUD operations for molecules, peaks, assignments
- [x] Implement cross-experiment query commands

**Implemented Commands (6):**
| Command | Parameters | Returns |
|---------|------------|---------|
| `init_database` | path | () |
| `save_molecule_to_db` | molecule_id | () |
| `load_molecule_from_db` | molecule_id | Option<String> |
| `list_molecules_in_db` | - | Vec<DbMoleculeInfo> |
| `delete_molecule_from_db` | molecule_id | () |
| `get_db_stats` | - | DbStats |
| `get_app_stats` | - | StateStats |

### Subgoal 5.4: Analysis Commands ✅
- [x] Create `commands/analysis.rs`
- [x] Add peak picking command with ML integration
- [x] Implement integration and volume calculation

**Implemented Commands (3):**
| Command | Parameters | Returns |
|---------|------------|---------|
| `pick_peaks_1d` | spectrum_id, PeakPickingParams | Vec<PickedPeak> |
| `integrate_peak` | spectrum_id, center_ppm, width_ppm | IntegrationResult |
| `clear_spectrum_peaks` | spectrum_id | () |

---

## Goal 6: React Frontend ✅

Build the user interface with spectrum visualization and assignment management.

### Subgoal 6.1: State Management ✅
- [x] Set up Zustand stores (`stores/spectrumStore.ts`, `stores/assignmentStore.ts`, `stores/uiStore.ts`)
- [x] Define state interfaces matching Rust data types (`types/tauri.ts`)
- [x] Implement Tauri invoke wrappers (`lib/tauri.ts`)

**Implementation Notes:**
- `spectrumStore.ts` - spectrum data, peaks, view state, tool mode
- `assignmentStore.ts` - molecule, residues, shifts, assignment status
- `uiStore.ts` - panel visibility, preferences (persisted)
- Full TypeScript types for all 23+ Tauri commands

### Subgoal 6.2: Spectrum Visualization ✅
- [x] Create `components/spectrum/SpectrumCanvas.tsx` with Canvas 2D rendering
- [x] Build `components/spectrum/SpectrumViewer.tsx` wrapper component
- [x] Add zoom, pan, and selection interactions
- [ ] Implement contour rendering for 2D spectra - *Future*

**Implementation Notes:**
- Canvas 2D renderer with DPI scaling
- Mouse wheel zoom, drag to pan
- Click to select peaks
- Real-time mouse position display (ppm, intensity)
- Peak pick dialog with SNR threshold

### Subgoal 6.3: Peak Management ✅
- [x] Create `components/spectrum/PeakList.tsx` interactive table
- [x] Add peak selection and editing
- [x] Implement peak-to-spectrum linking

**Implementation Notes:**
- Sortable by ppm, intensity, SNR
- Filter by ppm
- Click to select (syncs with canvas)
- Selected peak highlighted in both views

### Subgoal 6.4: Assignment Interface ✅
- [x] Build `components/assignment/AssignmentTable.tsx`
- [x] Create `components/assignment/SpinSystemView.tsx`
- [x] Add probability visualization for ambiguous assignments
- [x] Add `run_assignment` Tauri command
- [x] Extend assignmentStore with spin systems and results

**Implementation Notes:**
- AssignmentTable: Sortable/filterable table with confidence badges
- SpinSystemView: Canvas-based visualization with atom nodes and correlation edges
- Confidence colors: green (high), amber (medium), red (low)
- Cross-selection sync between AssignmentTable, SpinSystemView, SequenceViewer

### Subgoal 6.5: Molecule Display ✅
- [x] Create `components/molecule/SequenceViewer.tsx`
- [x] Add residue coloring by assignment status
- [x] Implement click-to-select residue interaction

**Implementation Notes:**
- Horizontal scrolling sequence display
- Color-coded: unassigned (gray), partial (amber), complete (green)
- Click to select residue
- Legend included

### Subgoal 6.6: Main Layout ✅
- [x] Create main App layout with resizable panels
- [x] Add spectrum list sidebar
- [x] Add toolbar with panel toggles
- [x] Include demo data loader

**Files Created (11 TypeScript files):**
```
src/
├── types/tauri.ts           (150 lines)
├── lib/tauri.ts             (195 lines)
├── stores/
│   ├── spectrumStore.ts     (175 lines)
│   ├── assignmentStore.ts   (130 lines)
│   └── uiStore.ts           (55 lines)
├── components/
│   ├── spectrum/
│   │   ├── SpectrumCanvas.tsx    (320 lines)
│   │   ├── SpectrumViewer.tsx    (160 lines)
│   │   ├── PeakList.tsx          (140 lines)
│   │   └── SpectrumList.tsx      (75 lines)
│   └── molecule/
│       └── SequenceViewer.tsx    (85 lines)
└── App.tsx                  (185 lines)
```

**Total:** ~1,670 lines of TypeScript/React code

---

## Goal 7: ML Integration 🔶

Integrate ONNX models for peak picking and assignment prediction.

### Subgoal 7.1: ONNX Runtime Setup ✅
- [x] Configure `ort` crate with download-binaries feature
- [x] Create `ml/inference.rs` with session management
- [ ] Add thread pool configuration for inference

**Implementation Notes:**
- Using ort 2.0.0-rc.10 (latest RC)
- Placeholder inference functions ready for real models
- `preprocess_spectrum_for_ml()` - resampling and normalization
- `postprocess_peak_predictions()` - threshold and PPM conversion

### Subgoal 7.2: Model Registry ✅
- [x] Implement `ml/model_registry.rs` with version management
- [x] Add model loading with hash verification
- [ ] Support model hot-reloading

**Implementation Notes:**
- `ModelRegistry` with `RwLock<HashMap<String, Arc<LoadedModel>>>`
- `get_or_load()` - lazy loading with caching
- `list_models()`, `unload()`, `clear()` methods
- Model path: `{model_dir}/{model_id}/{version}/model.onnx`

### Subgoal 7.3: Inference Pipeline 🔶
- [x] Create inference functions for peak picking (stub)
- [ ] Build assignment prediction pipeline
- [x] Add result caching (`ml/cache.rs`)

**Implementation Notes:**
- `InferenceCache<T>` - LRU-style with max age and max entries
- Default: 5 minute TTL, 100 entries max
- `get()`, `set()`, `remove()`, `clear()` methods

---

## Goal 8: Testing and Validation 🔶

Ensure correctness and performance through comprehensive testing.

### Subgoal 8.1: Unit Tests ✅
- [x] Test FFT processing against known results
- [x] Test belief propagation convergence
- [x] Test database operations

**Implemented Tests:**
- `fft.rs`: delta function, roundtrip, sine wave, zero-fill
- `phasing.rs`: zero phase, 90° rotation, auto-phase
- `baseline.rs`: polynomial fit, auto-baseline
- `apodization.rs`: exponential decay, window endpoints
- `factor_graph.rs`: creation, peak factors
- `belief_propagation.rs`: simple BP, softmax
- `assignment.rs`: global assignment
- `scoring.rs`: BMRB, peak, connectivity scoring
- `molecule.rs`: from_sequence, atom lookup, navigation
- `constraint.rs`: distance satisfaction, violation
- `db/`: migrations, queries, save/load

### Subgoal 8.2: Property-Based Testing
- [ ] Set up proptest for numerical operations
- [ ] Test spectrum processing invariants
- [ ] Test assignment algorithm properties

### Subgoal 8.3: Benchmarking
- [ ] Create criterion benchmarks for FFT
- [ ] Benchmark belief propagation iterations
- [ ] Profile memory usage for large spectra

### Subgoal 8.4: Integration Testing
- [ ] Test end-to-end assignment workflow
- [ ] Validate against known NMR datasets
- [ ] Cross-platform testing (macOS, Windows, Linux)

---

## Goal 9: CRYSTALLINE Density Crystallization Engine ⏳

Implement the CRYSTALLINE density crystallization paradigm as an alternative analysis mode alongside the traditional factor graph approach.

### Subgoal 9.1: Density Data Structures
- [ ] Create `data/density.rs` with PeakState<D>, CrystallinePeak<D>
- [ ] Add DensityField<D> enum (Particles, GMM, Hybrid)
- [ ] Define ParticleCloud<D>, Particle<D> types
- [ ] Define VariationalGMM<D>, GaussianComponent<D> types

### Subgoal 9.2: Density Representation Module
- [ ] Create `density/` module structure
- [ ] Implement `density/particle.rs` - Particle cloud operations
  - Particle initialization from BMRB priors
  - Weight update with likelihood
  - Systematic resampling
  - Effective sample size computation
- [ ] Implement `density/gmm.rs` - Variational Gaussian Mixture Model
  - DP-GMM (Dirichlet Process) fitting
  - Component creation/pruning
  - ELBO computation
- [ ] Implement `density/kde.rs` - Kernel density estimation
  - Multi-dimensional Gaussian kernels
  - Bandwidth selection (Scott's rule, Silverman)
  - Grid evaluation

### Subgoal 9.3: Topological Persistence Module
- [ ] Create `topology/` module structure
- [ ] Implement `topology/union_find.rs` - Union-Find with persistence tracking
  - Path compression
  - Union by rank
  - Persistence value storage
- [ ] Implement `topology/persistence.rs` - Persistent homology computation
  - Superlevel set filtration
  - Birth-death pair extraction
  - Persistence diagram
  - Noise threshold estimation

### Subgoal 9.4: Crystallization Engine Module
- [ ] Create `crystallize/` module structure
- [ ] Implement `crystallize/criteria.rs` - Crystallization criteria
  - Entropy threshold check
  - Persistence threshold check
  - MDL (Minimum Description Length) gain
  - Multi-experiment evidence count
- [ ] Implement `crystallize/entropy.rs` - Entropy computation
  - Shannon entropy for discrete distributions
  - Differential entropy for continuous
  - Normalized entropy for comparison
- [ ] Implement `crystallize/mod.rs` - Main crystallization state machine
  - CrystallizationState struct
  - Evidence record tracking
  - Progress computation
  - Region extraction and crystallization

### Subgoal 9.5: Density Tauri Commands
- [ ] `initialize_density` - Initialize from sequence + BMRB priors
- [ ] `add_experiment_evidence` - Update density with experiment
- [ ] `get_density_field` - Evaluate density for visualization
- [ ] `get_peak_states` - Get current peak state summaries
- [ ] Update AppState with crystallization_states storage

### Subgoal 9.6: Frontend Density UI
- [ ] Create `stores/densityStore.ts` - Zustand state management
- [ ] Implement `components/density/DensityView.tsx` - Density field heatmap
- [ ] Implement `components/density/CrystallizationProgress.tsx` - Progress indicator
- [ ] Implement `components/density/UncertaintyEllipse.tsx` - Confidence regions
- [ ] Add TypeScript types to `types/tauri.ts`
- [ ] Add invoke wrappers to `lib/tauri.ts`

**Implementation Notes:**
- CRYSTALLINE mode is optional; users can choose traditional or CRYSTALLINE
- Both modes share underlying data structures
- Crystallized peaks can feed into factor graph for refinement
- Const generics (`<const D: usize>`) handle 1D, 2D, 3D, 4D spectra

---

## Implementation Priority Order

**Phase 1: MVP Core** ✅ COMPLETE (Backend)
1. ✅ Goal 1 (Foundation)
2. ✅ Goal 2.1-2.2 (Database + Spectrum structures)
3. ✅ Goal 3.1 (FFT processing)
4. ✅ Goal 5.1 (Spectrum commands)
5. ⏳ Goal 6.1-6.2 (State management + Spectrum viewer) - **NEXT**

**Phase 2: Full Data Layer** ✅ COMPLETE
6. ✅ Goal 2.3-2.4 (Molecule graph + constraints)
7. ✅ Goal 3.2-3.4 (Phase, baseline, apodization)
8. ✅ Goal 5.3 (Database commands)

**Phase 3: Inference Engine** ✅ COMPLETE
9. ✅ Goal 4.1-4.4 (Full factor graph and BP)
10. ✅ Goal 5.2 (Assignment commands)
11. ⏳ Goal 6.3-6.5 (Peak and assignment UI)

**Phase 4: ML and Polish**
12. 🔶 Goal 7 (ML integration) - stubs complete
13. 🔶 Goal 8 (Testing) - unit tests complete
14. ✅ Goal 5.4 (Analysis commands)

**Phase 5: CRYSTALLINE Density Crystallization** ⏳ NEW
15. ⏳ Goal 9.1 (Density data structures)
16. ⏳ Goal 9.2 (Density representation: particle, GMM, KDE)
17. ⏳ Goal 9.3 (Topological persistence)
18. ⏳ Goal 9.4 (Crystallization engine)
19. ⏳ Goal 9.5 (Density Tauri commands)
20. ⏳ Goal 9.6 (Frontend density UI)

---

## Files Created (33 Rust source files)

```
src-tauri/src/
├── lib.rs                    ✅ Tauri entry point (126 lines)
├── main.rs                   ✅ Windows subsystem (6 lines)
├── error.rs                  ✅ Error types (115 lines)
├── commands/
│   ├── mod.rs                ✅
│   ├── spectrum.rs           ✅ 5 commands (165 lines)
│   ├── assignment.rs         ✅ 8 commands (185 lines)
│   ├── database.rs           ✅ 7 commands (131 lines)
│   └── analysis.rs           ✅ 3 commands (160 lines)
├── state/
│   ├── mod.rs                ✅
│   └── app_state.rs          ✅ (180 lines)
├── processing/
│   ├── mod.rs                ✅
│   ├── fft.rs                ✅ (185 lines)
│   ├── phasing.rs            ✅ (175 lines)
│   ├── baseline.rs           ✅ (195 lines)
│   └── apodization.rs        ✅ (225 lines)
├── data/
│   ├── mod.rs                ✅
│   ├── spectrum.rs           ✅ (320 lines)
│   ├── experiment.rs         ✅ (265 lines)
│   ├── molecule.rs           ✅ (340 lines)
│   └── constraint.rs         ✅ (230 lines)
├── inference/
│   ├── mod.rs                ✅
│   ├── factor_graph.rs       ✅ (165 lines)
│   ├── belief_propagation.rs ✅ (225 lines)
│   ├── assignment.rs         ✅ (85 lines)
│   └── scoring.rs            ✅ (110 lines)
├── db/
│   ├── mod.rs                ✅
│   ├── connection.rs         ✅ (75 lines)
│   ├── migrations.rs         ✅ (195 lines)
│   └── queries.rs            ✅ (175 lines)
└── ml/
    ├── mod.rs                ✅
    ├── model_registry.rs     ✅ (85 lines)
    ├── inference.rs          ✅ (95 lines)
    └── cache.rs              ✅ (115 lines)
```

**Total:** ~4,000 lines of Rust code

---

## Success Metrics

- **Assignment Accuracy**: Target 90%+ backbone assignment accuracy on benchmark datasets
- **Performance**: Process 2D HSQC (2048x512) in <100ms
- **BP Convergence**: Converge within 50 iterations for typical datasets
- **Memory**: Handle 4D datasets up to 1GB without OOM

---

## Next Steps

1. ✅ **Frontend Development** (Goal 6) - COMPLETE
   - Zustand stores, TypeScript types, Tauri wrappers
   - SpectrumViewer with Canvas 2D rendering
   - Peak picking UI, PeakList with sorting/filtering
   - SequenceViewer with assignment status colors
   - Demo data loader for testing

2. ✅ **Assignment Interface** (Goal 6.4) - COMPLETE
   - AssignmentTable for editing shifts with confidence display
   - SpinSystemView canvas-based visualization
   - Probability visualization (confidence colors, alternatives)
   - `run_assignment` Tauri command exposing global assignment
   - Cross-component selection synchronization

3. **ML Model Integration** (Goal 7)
   - Train/obtain peak picking model
   - Implement real ONNX inference

4. **File Format Support**
   - Bruker TopSpin format reader
   - NMRPipe format reader
   - Varian/Agilent format reader

5. **Integration Testing**
   - End-to-end workflow tests
   - Benchmark with real NMR datasets

6. **CRYSTALLINE Density Crystallization** (Goal 9) - NEW
   - Implement density data structures (PeakState, DensityField)
   - Create density module (particle cloud, GMM, KDE)
   - Build topology module (persistent homology)
   - Implement crystallization engine
   - Add density Tauri commands
   - Create frontend density visualization
