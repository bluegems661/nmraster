# NMR Platform: Complete Implementation Specification

## Project Codename: **CRYSTALLINE**
### Continuous-density Recognition Yielding Structures Through Adaptive Lattice-based Inference in NMR Experiments

---

## Executive Summary

CRYSTALLINE is a next-generation, open-source NMR spectroscopy platform that fundamentally reimagines how peak picking, assignment, and structure calculation are performed. Unlike traditional sequential workflows where humans process one spectrum at a time, CRYSTALLINE reasons about **all experimental data simultaneously** through a unified probabilistic framework.

The core innovation is the **density crystallization paradigm**: instead of forcing binary peak detection decisions upfront, spectral data begins as continuous probability densities over chemical shift space. These densities "crystallize" into discrete peaks only when sufficient multi-experiment evidence accumulates—naturally handling crowded regions where traditional methods fail.

**Target Users**: NMR spectroscopists in academia and industry  
**Initial Focus**: Peptides (extensible to natural products, small molecules, oligonucleotides)  
**License**: Open source (MIT or Apache 2.0)  
**Platform**: Desktop (Windows, macOS, Linux) via Tauri 2.0

---

## Part 1: Architecture Overview

### 1.1 Technology Stack

```
┌─────────────────────────────────────────────────────────────────┐
│                    React + TypeScript Frontend                   │
│         (WebGL spectrum visualization, assignment UI)            │
├─────────────────────────────────────────────────────────────────┤
│                      Tauri 2.0 IPC Bridge                        │
├─────────────────────────────────────────────────────────────────┤
│                    Rust Processing Core                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │   RustFFT    │  │   ndarray    │  │   petgraph           │  │
│  │   (FFT)      │  │   (arrays)   │  │   (molecular graph)  │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │   rusqlite   │  │   ort        │  │   rayon              │  │
│  │   (database) │  │   (ONNX ML)  │  │   (parallelism)      │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│            PyO3 Bridge (optional, for Python ML models)          │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Core Paradigm: Density Crystallization

Traditional workflow:
```
Raw Spectrum → Peak Pick → Assign → Calculate Structure
     ↓              ↓         ↓              ↓
   (data)      (binary)   (manual)      (restraints)
```

CRYSTALLINE workflow:
```
All Spectra → Unified Density Field → Evidence Accumulation → Crystallization → Structure
     ↓               ↓                        ↓                    ↓              ↓
 (parallel)    (probabilistic)         (factor graph)         (threshold)    (ensemble)
```

### 1.3 Key Innovations

1. **Continuous Density Representation**: Peaks exist as probability distributions, not binary entities
2. **Simultaneous Multi-Experiment Analysis**: Factor graphs connect all experiments for global inference
3. **Topological Persistence for Peak Significance**: Parameter-free peak detection using persistent homology
4. **Information-Theoretic Crystallization**: Peaks "emerge" when entropy drops below threshold
5. **Uncertainty Propagation**: Full posterior distributions flow through to structure calculation
6. **Provenance Tracking**: Git-like version control for all analysis decisions

---

## Part 2: Project Structure

### 2.1 Directory Layout

```
crystalline/
├── src-tauri/
│   ├── Cargo.toml
│   ├── build.rs
│   ├── tauri.conf.json
│   └── src/
│       ├── lib.rs                          # Tauri plugin registration
│       ├── main.rs                         # Entry point
│       ├── error.rs                        # Error types (thiserror)
│       │
│       ├── commands/                       # Tauri command handlers
│       │   ├── mod.rs
│       │   ├── spectrum.rs                 # Load, process spectra
│       │   ├── density.rs                  # Density field operations
│       │   ├── crystallize.rs              # Peak crystallization
│       │   ├── assignment.rs               # Global assignment
│       │   ├── structure.rs                # Structure calculation
│       │   └── project.rs                  # Project management
│       │
│       ├── data/                           # Core data structures
│       │   ├── mod.rs
│       │   ├── spectrum.rs                 # Spectrum types (1D, 2D, 3D, 4D)
│       │   ├── molecule.rs                 # Molecular graph (petgraph)
│       │   ├── peak.rs                     # Peak and PeakState types
│       │   ├── density.rs                  # DensityField representations
│       │   ├── experiment.rs               # Experiment metadata
│       │   ├── constraint.rs               # NOE, dihedral, etc.
│       │   └── assignment.rs               # Assignment with uncertainty
│       │
│       ├── io/                             # File I/O
│       │   ├── mod.rs
│       │   ├── bruker.rs                   # Bruker format reader
│       │   ├── nmrstar.rs                  # NMR-STAR format
│       │   ├── nef.rs                      # NEF format
│       │   ├── bmrb.rs                     # BMRB database queries
│       │   ├── zarr.rs                     # Zarr spectral storage
│       │   └── parquet.rs                  # Parquet peak lists
│       │
│       ├── processing/                     # Signal processing
│       │   ├── mod.rs
│       │   ├── fft.rs                      # FFT (rustfft)
│       │   ├── apodization.rs              # Window functions
│       │   ├── phasing.rs                  # Phase correction
│       │   ├── baseline.rs                 # Baseline correction
│       │   └── referencing.rs              # Chemical shift referencing
│       │
│       ├── density/                        # Density field operations
│       │   ├── mod.rs
│       │   ├── particle.rs                 # Particle cloud representation
│       │   ├── gmm.rs                      # Variational GMM (DP-GMM)
│       │   ├── kde.rs                      # Kernel density estimation
│       │   ├── sparse_grid.rs              # Sparse grid for high-D
│       │   └── evolution.rs                # Density dynamics
│       │
│       ├── topology/                       # Topological data analysis
│       │   ├── mod.rs
│       │   ├── persistence.rs              # Persistent homology
│       │   ├── filtration.rs               # Sublevel set filtration
│       │   └── union_find.rs               # Union-find with persistence
│       │
│       ├── inference/                      # Probabilistic inference
│       │   ├── mod.rs
│       │   ├── factor_graph.rs             # Factor graph construction
│       │   ├── belief_propagation.rs       # Message passing
│       │   ├── variational.rs              # Variational inference
│       │   └── scoring.rs                  # Multi-experiment scoring
│       │
│       ├── crystallize/                    # Peak crystallization
│       │   ├── mod.rs
│       │   ├── criteria.rs                 # Crystallization criteria
│       │   ├── nucleation.rs               # Nucleation theory model
│       │   ├── entropy.rs                  # Entropy computation
│       │   └── mdl.rs                      # MDL model selection
│       │
│       ├── structure/                      # Structure calculation
│       │   ├── mod.rs
│       │   ├── restraints.rs               # Restraint generation
│       │   ├── geometry.rs                 # Distance geometry
│       │   ├── ensemble.rs                 # Ensemble generation
│       │   └── validation.rs               # Structure validation
│       │
│       ├── ml/                             # Machine learning
│       │   ├── mod.rs
│       │   ├── registry.rs                 # Model registry
│       │   ├── inference.rs                # ONNX inference
│       │   └── shift_prediction.rs         # Chemical shift prediction
│       │
│       ├── db/                             # Database layer
│       │   ├── mod.rs
│       │   ├── connection.rs               # SQLite connection
│       │   ├── migrations.rs               # Schema migrations
│       │   ├── queries.rs                  # Prepared statements
│       │   └── provenance.rs               # Audit trail
│       │
│       └── state/                          # Application state
│           ├── mod.rs
│           └── app_state.rs                # Mutex-wrapped state
│
├── src/                                    # React frontend
│   ├── main.tsx
│   ├── App.tsx
│   ├── index.css
│   │
│   ├── components/
│   │   ├── spectrum/
│   │   │   ├── SpectrumViewer.tsx          # Main 1D/2D viewer
│   │   │   ├── SpectrumCanvas.tsx          # WebGL rendering
│   │   │   ├── ContourPlot.tsx             # 2D contour display
│   │   │   ├── DensityOverlay.tsx          # Density field visualization
│   │   │   └── PeakMarkers.tsx             # Peak annotation layer
│   │   │
│   │   ├── density/
│   │   │   ├── DensityView.tsx             # Density field display
│   │   │   ├── CrystallizationProgress.tsx # Progress indicator
│   │   │   └── UncertaintyEllipse.tsx      # Confidence regions
│   │   │
│   │   ├── assignment/
│   │   │   ├── AssignmentTable.tsx         # Chemical shift list
│   │   │   ├── SpinSystemView.tsx          # Spin system grouping
│   │   │   ├── SequenceMap.tsx             # Sequence with coverage
│   │   │   └── ConfidenceBar.tsx           # Assignment confidence
│   │   │
│   │   ├── molecule/
│   │   │   ├── SequenceViewer.tsx          # Sequence display
│   │   │   ├── Structure3D.tsx             # 3D structure (Three.js)
│   │   │   └── ContactMap.tsx              # NOE contact map
│   │   │
│   │   ├── experiment/
│   │   │   ├── ExperimentList.tsx          # Loaded experiments
│   │   │   ├── ExperimentConfig.tsx        # Parameters
│   │   │   └── EvidenceGraph.tsx           # Factor graph visualization
│   │   │
│   │   └── common/
│   │       ├── Toolbar.tsx
│   │       ├── StatusBar.tsx
│   │       └── Sidebar.tsx
│   │
│   ├── stores/                             # Zustand state management
│   │   ├── spectrumStore.ts
│   │   ├── densityStore.ts
│   │   ├── assignmentStore.ts
│   │   ├── moleculeStore.ts
│   │   └── projectStore.ts
│   │
│   ├── hooks/
│   │   ├── useSpectrum.ts
│   │   ├── useDensity.ts
│   │   ├── useTauriCommand.ts
│   │   └── useLinkedViews.ts               # Brushing/linking
│   │
│   └── lib/
│       ├── tauri.ts                        # Invoke wrappers
│       ├── webgl.ts                        # WebGL utilities
│       └── colormap.ts                     # Spectral colormaps
│
├── models/                                 # ML models
│   ├── peak_picker/
│   │   └── v1.0.0/
│   │       ├── model.onnx
│   │       └── metadata.json
│   └── shift_predictor/
│       └── v1.0.0/
│           ├── model.onnx
│           └── metadata.json
│
├── migrations/                             # Database migrations
│   ├── 001_initial.sql
│   ├── 002_density_fields.sql
│   └── 003_provenance.sql
│
├── tests/                                  # Test data and integration tests
│   ├── fixtures/
│   │   ├── bruker_1d/
│   │   ├── bruker_2d_hsqc/
│   │   └── bmrb_assignments/
│   └── integration/
│
├── package.json
├── tsconfig.json
├── vite.config.ts
└── README.md
```

---

## Part 3: Core Data Structures

### 3.1 Rust Type Definitions

```rust
// src-tauri/src/data/spectrum.rs

use ndarray::{Array1, Array2, Array3, Array4};
use serde::{Deserialize, Serialize};

/// Nucleus types in NMR
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Nucleus {
    H1,
    C13,
    N15,
    P31,
    F19,
}

/// Spectral dimension metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralDimension {
    pub nucleus: Nucleus,
    pub spectral_width_hz: f64,
    pub spectral_width_ppm: f64,
    pub carrier_frequency_hz: f64,
    pub num_points: usize,
    pub ppm_range: (f64, f64),  // (min, max) in ppm
}

/// Generic spectrum with dimensionality
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SpectrumData {
    D1(Array1<f64>),
    D2(Array2<f64>),
    D3(Array3<f64>),
    D4(Array4<f64>),
}

/// Complete spectrum with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Spectrum {
    pub id: uuid::Uuid,
    pub name: String,
    pub experiment_type: ExperimentType,
    pub dimensions: Vec<SpectralDimension>,
    pub data: SpectrumData,
    pub noise_level: Option<f64>,
    pub processing_history: Vec<ProcessingStep>,
}

/// Experiment types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExperimentType {
    // 1D experiments
    Proton1D,
    Carbon1D,
    
    // 2D experiments
    HSQC,
    HMBC,
    COSY,
    TOCSY,
    NOESY,
    ROESY,
    
    // 3D experiments
    HNCO,
    HNCA,
    HNCACB,
    CBCACONH,
    HCCH_TOCSY,
    NOESY_HSQC,
    
    // 4D experiments
    HCCH_NOESY,
    CC_NOESY,
    
    // Relaxation
    T1,
    T2,
    HetNOE,
    
    // Other
    Custom(String),
}
```

```rust
// src-tauri/src/data/density.rs

use ndarray::Array1;
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

/// Chemical shift position in N-dimensional space
pub type ChemShift<const D: usize> = [f64; D];

/// Particle in the density field
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Particle<const D: usize> {
    pub position: ChemShift<D>,
    pub weight: f64,
    pub component_id: Option<usize>,
}

/// Particle cloud representation of density
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParticleCloud<const D: usize> {
    pub particles: Vec<Particle<D>>,
    pub effective_sample_size: f64,
    pub resample_threshold: f64,
}

/// Gaussian mixture component
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaussianComponent<const D: usize> {
    pub mean: ChemShift<D>,
    pub covariance: [[f64; D]; D],
    pub weight: f64,
    pub precision: [[f64; D]; D],  // Inverse covariance
}

/// Variational Gaussian Mixture Model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariationalGMM<const D: usize> {
    pub components: Vec<GaussianComponent<D>>,
    pub concentration: f64,  // Dirichlet concentration parameter
    pub converged: bool,
    pub elbo: f64,  // Evidence lower bound
}

/// Density field representation (multiple options)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DensityField<const D: usize> {
    Particles(ParticleCloud<D>),
    GMM(VariationalGMM<D>),
    Hybrid {
        particles: ParticleCloud<D>,
        gmm: VariationalGMM<D>,
    },
}

/// Peak at various crystallization stages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PeakState<const D: usize> {
    /// Diffuse density - not yet a peak
    Diffuse {
        region_id: usize,
        center_estimate: ChemShift<D>,
        spread: f64,  // Approximate extent
    },
    
    /// Nucleating - gathering evidence
    Nucleating {
        mean: ChemShift<D>,
        covariance: [[f64; D]; D],
        persistence: f64,
        entropy: f64,
        evidence_sources: Vec<ExperimentType>,
    },
    
    /// Crystallized - definite peak
    Crystallized(Peak<D>),
}

/// Fully crystallized peak with uncertainty
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Peak<const D: usize> {
    pub id: uuid::Uuid,
    pub position: ChemShift<D>,
    pub covariance: [[f64; D]; D],
    pub intensity: f64,
    pub volume: Option<f64>,
    pub line_width: [f64; D],
    
    // Crystallization metadata
    pub persistence: f64,
    pub crystallization_entropy: f64,
    pub confidence: f64,
    pub evidence_sources: Vec<ExperimentType>,
    
    // Assignment (if any)
    pub assignments: Vec<PeakAssignment>,
}

/// Peak assignment with probability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeakAssignment {
    pub dimension: usize,
    pub atom_id: String,  // e.g., "A.5.HN" (chain.residue.atom)
    pub probability: f64,
}
```

```rust
// src-tauri/src/data/molecule.rs

use petgraph::graph::{DiGraph, NodeIndex};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Amino acid residue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Residue {
    pub chain_id: String,
    pub sequence_number: i32,
    pub residue_type: ResidueType,
    pub atoms: Vec<Atom>,
}

/// Atom in a residue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Atom {
    pub name: String,        // e.g., "HN", "CA", "CB"
    pub element: Element,
    pub position: Option<[f64; 3]>,  // 3D coordinates if known
}

/// Element type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Element {
    H, C, N, O, S, P, F,
}

/// Standard amino acid types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ResidueType {
    Ala, Arg, Asn, Asp, Cys, Gln, Glu, Gly, His, Ile,
    Leu, Lys, Met, Phe, Pro, Ser, Thr, Trp, Tyr, Val,
    // Non-standard
    Custom(char),
}

/// Molecular graph
#[derive(Debug, Clone)]
pub struct MolecularGraph {
    pub graph: DiGraph<AtomNode, BondEdge>,
    pub atom_index: HashMap<String, NodeIndex>,  // "A.5.HN" -> NodeIndex
    pub residues: Vec<Residue>,
    pub sequence: String,
}

/// Node in molecular graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtomNode {
    pub atom_id: String,
    pub element: Element,
    pub residue_number: i32,
    pub atom_name: String,
    
    // Chemical shift information
    pub assigned_shift: Option<f64>,
    pub shift_uncertainty: Option<f64>,
    pub bmrb_prior: Option<(f64, f64)>,  // (mean, std) from BMRB
}

/// Edge in molecular graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BondEdge {
    Covalent { order: u8 },
    Sequential,  // i -> i+1 backbone connectivity
    NOE { distance_estimate: Option<(f64, f64)> },  // (lower, upper) bounds
    HydrogenBond,
}
```

```rust
// src-tauri/src/inference/factor_graph.rs

use petgraph::graph::{DiGraph, NodeIndex};
use ndarray::Array1;
use std::collections::HashMap;

/// Variable node in factor graph
#[derive(Debug, Clone)]
pub struct VariableNode {
    pub id: String,
    pub node_type: VariableType,
    pub domain: VariableDomain,
}

#[derive(Debug, Clone)]
pub enum VariableType {
    ChemicalShift { atom_id: String },
    PeakExistence { peak_id: uuid::Uuid },
    Assignment { peak_id: uuid::Uuid, dimension: usize },
}

#[derive(Debug, Clone)]
pub enum VariableDomain {
    Continuous { min: f64, max: f64, discretization: usize },
    Binary,
    Categorical { num_states: usize },
}

/// Factor node in factor graph
#[derive(Debug, Clone)]
pub struct FactorNode {
    pub id: String,
    pub factor_type: FactorType,
    pub connected_variables: Vec<NodeIndex>,
}

#[derive(Debug, Clone)]
pub enum FactorType {
    /// Prior from BMRB statistics
    BMRBPrior {
        mean: f64,
        std: f64,
        atom_type: String,
    },
    
    /// Peak position must be consistent with chemical shift
    PeakConsistency {
        peak_position: f64,
        tolerance: f64,
        experiment_type: ExperimentType,
    },
    
    /// Sequential connectivity (i -> i+1)
    SequentialConnectivity {
        expected_shift_difference: f64,
        tolerance: f64,
    },
    
    /// NOE distance constraint
    NOEDistance {
        intensity: f64,
        distance_bounds: (f64, f64),
    },
    
    /// TOCSY spin system membership
    SpinSystem {
        residue_id: String,
    },
    
    /// J-coupling pattern
    JCoupling {
        coupling_constant: f64,
        tolerance: f64,
    },
    
    /// Custom factor with arbitrary potential function
    Custom {
        name: String,
        log_potential: Vec<f64>,  // Discretized potential
    },
}

/// Factor graph for global inference
pub struct FactorGraph {
    pub graph: DiGraph<FactorGraphNode, ()>,
    pub variable_indices: HashMap<String, NodeIndex>,
    pub factor_indices: HashMap<String, NodeIndex>,
    pub messages: HashMap<(NodeIndex, NodeIndex), Array1<f64>>,
}

#[derive(Debug, Clone)]
pub enum FactorGraphNode {
    Variable(VariableNode),
    Factor(FactorNode),
}

impl FactorGraph {
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            variable_indices: HashMap::new(),
            factor_indices: HashMap::new(),
            messages: HashMap::new(),
        }
    }
    
    /// Add experiment data to factor graph
    pub fn add_experiment(&mut self, experiment: &Experiment) {
        match experiment.experiment_type {
            ExperimentType::HSQC => self.add_hsqc_factors(experiment),
            ExperimentType::NOESY => self.add_noesy_factors(experiment),
            ExperimentType::TOCSY => self.add_tocsy_factors(experiment),
            _ => {}
        }
    }
    
    /// Run belief propagation to convergence
    pub fn infer(&mut self, max_iterations: usize, tolerance: f64) -> InferenceResult {
        self.initialize_messages();
        
        for iteration in 0..max_iterations {
            let max_delta = self.propagate_iteration();
            if max_delta < tolerance {
                return InferenceResult {
                    converged: true,
                    iterations: iteration,
                    marginals: self.compute_marginals(),
                };
            }
        }
        
        InferenceResult {
            converged: false,
            iterations: max_iterations,
            marginals: self.compute_marginals(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct InferenceResult {
    pub converged: bool,
    pub iterations: usize,
    pub marginals: HashMap<String, Array1<f64>>,
}
```

---

## Part 4: Density Crystallization Algorithm

### 4.1 Mathematical Foundation

The density crystallization process follows these principles:

**Initial State**: Probability density over chemical shift space initialized from BMRB priors

**Evidence Accumulation**: Each experiment updates the density via Bayesian inference:
```
p(ρ | D₁...Dₑ) ∝ p(Dₑ | ρ) · p(ρ | D₁...Dₑ₋₁)
```

**Crystallization Criterion**: A density region becomes a peak when ALL criteria met:
1. **Entropy threshold**: H(X | D) < H_crit (~0.02 ppm in ¹H)
2. **Persistence threshold**: Topological persistence > noise level
3. **MDL criterion**: Model with peak has lower description length

### 4.2 Algorithm Implementation

```rust
// src-tauri/src/crystallize/mod.rs

use crate::data::density::{DensityField, PeakState, Peak, ParticleCloud};
use crate::data::spectrum::ExperimentType;
use crate::topology::persistence::PersistenceDiagram;

/// Configuration for crystallization
#[derive(Debug, Clone)]
pub struct CrystallizationConfig {
    pub entropy_threshold: f64,           // Default: ln(0.02) for 0.02 ppm
    pub persistence_threshold: f64,       // Default: 3.0 (3x noise)
    pub mdl_threshold: f64,               // Default: 2.0 (bits)
    pub min_evidence_sources: usize,      // Default: 2 experiments
    pub particle_count: usize,            // Default: 10000
    pub resample_threshold: f64,          // Default: 0.5 (ESS ratio)
}

impl Default for CrystallizationConfig {
    fn default() -> Self {
        Self {
            entropy_threshold: -3.91,     // ln(0.02)
            persistence_threshold: 3.0,
            mdl_threshold: 2.0,
            min_evidence_sources: 2,
            particle_count: 10000,
            resample_threshold: 0.5,
        }
    }
}

/// State of the crystallization process
pub struct CrystallizationState<const D: usize> {
    pub density: DensityField<D>,
    pub peak_states: Vec<PeakState<D>>,
    pub persistence: PersistenceDiagram,
    pub evidence_history: Vec<EvidenceRecord>,
    pub crystallization_progress: f64,  // 0.0 to 1.0
}

/// Record of evidence accumulation
#[derive(Debug, Clone)]
pub struct EvidenceRecord {
    pub experiment_type: ExperimentType,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub num_observations: usize,
    pub log_likelihood: f64,
}

impl<const D: usize> CrystallizationState<D> {
    /// Initialize from sequence and BMRB priors
    pub fn initialize(
        sequence: &str,
        bmrb_priors: &BMRBPriors,
        config: &CrystallizationConfig,
    ) -> Self {
        // Sample initial particles from BMRB priors
        let particles = sample_from_bmrb_prior(sequence, bmrb_priors, config.particle_count);
        
        Self {
            density: DensityField::Particles(particles),
            peak_states: Vec::new(),
            persistence: PersistenceDiagram::empty(),
            evidence_history: Vec::new(),
            crystallization_progress: 0.0,
        }
    }
    
    /// Update density with new experiment
    pub fn update_with_experiment(
        &mut self,
        experiment: &Experiment<D>,
        config: &CrystallizationConfig,
    ) {
        // Extract observations from experiment
        let observations = extract_observations(experiment);
        
        // Update particles with likelihood
        if let DensityField::Particles(ref mut particles) = self.density {
            for obs in &observations {
                let likelihood = compute_likelihood(obs, experiment.experiment_type);
                particles.update_weights(&likelihood);
            }
            
            // Resample if ESS too low
            if particles.effective_sample_size() < config.resample_threshold * config.particle_count as f64 {
                particles.resample_systematic();
            }
        }
        
        // Record evidence
        self.evidence_history.push(EvidenceRecord {
            experiment_type: experiment.experiment_type,
            timestamp: chrono::Utc::now(),
            num_observations: observations.len(),
            log_likelihood: self.compute_log_likelihood(experiment),
        });
        
        // Update persistence diagram
        self.update_persistence();
        
        // Check for crystallization
        self.check_crystallization(config);
    }
    
    /// Update persistent homology from current density
    fn update_persistence(&mut self) {
        let kde = self.density.to_kde();
        self.persistence = PersistenceDiagram::from_density(&kde);
    }
    
    /// Check if any density regions should crystallize
    fn check_crystallization(&mut self, config: &CrystallizationConfig) {
        let candidates = self.persistence.significant_features(config.persistence_threshold);
        
        for candidate in candidates {
            let region = self.extract_region(&candidate);
            
            // Check all crystallization criteria
            let entropy = region.compute_entropy();
            let evidence_count = region.evidence_sources.len();
            let mdl_gain = region.compute_mdl_gain();
            
            if entropy < config.entropy_threshold
                && evidence_count >= config.min_evidence_sources
                && mdl_gain > config.mdl_threshold
            {
                // Crystallize this region into a peak
                let peak = self.crystallize_region(region);
                self.peak_states.push(PeakState::Crystallized(peak));
                
                // Remove particles from crystallized region
                self.density.remove_region(&candidate);
            } else if candidate.persistence > config.persistence_threshold * 0.5 {
                // Mark as nucleating
                self.peak_states.push(PeakState::Nucleating {
                    mean: region.center,
                    covariance: region.covariance,
                    persistence: candidate.persistence,
                    entropy,
                    evidence_sources: region.evidence_sources,
                });
            }
        }
        
        // Update progress
        self.crystallization_progress = self.compute_progress();
    }
    
    /// Crystallize a region into a definite peak
    fn crystallize_region(&self, region: DensityRegion<D>) -> Peak<D> {
        Peak {
            id: uuid::Uuid::new_v4(),
            position: region.center,
            covariance: region.covariance,
            intensity: region.total_weight,
            volume: Some(region.volume),
            line_width: region.estimate_linewidth(),
            persistence: region.persistence,
            crystallization_entropy: region.entropy,
            confidence: 1.0 - region.entropy.exp(),  // Convert entropy to confidence
            evidence_sources: region.evidence_sources,
            assignments: Vec::new(),
        }
    }
}
```

### 4.3 Persistence Computation

```rust
// src-tauri/src/topology/persistence.rs

use std::collections::BinaryHeap;

/// Persistence diagram storing birth-death pairs
#[derive(Debug, Clone)]
pub struct PersistenceDiagram {
    pub pairs: Vec<BirthDeathPair>,
}

#[derive(Debug, Clone)]
pub struct BirthDeathPair {
    pub birth: f64,
    pub death: f64,
    pub location: Vec<f64>,  // Location of the feature
    pub dimension: usize,    // Homological dimension (0 for peaks)
}

impl BirthDeathPair {
    pub fn persistence(&self) -> f64 {
        self.death - self.birth
    }
}

impl PersistenceDiagram {
    /// Compute persistence from density field using sublevel set filtration
    pub fn from_density<const D: usize>(density: &KernelDensityEstimate<D>) -> Self {
        // Get grid values
        let values = density.evaluate_on_grid();
        let indices: Vec<usize> = (0..values.len()).collect();
        
        // Sort by value (descending for superlevel sets = peaks)
        let mut sorted_indices = indices.clone();
        sorted_indices.sort_by(|&a, &b| {
            values[b].partial_cmp(&values[a]).unwrap()
        });
        
        // Union-Find for connected components
        let mut uf = UnionFind::new(values.len());
        let mut pairs = Vec::new();
        let mut component_birth: Vec<Option<f64>> = vec![None; values.len()];
        
        for &idx in &sorted_indices {
            let value = values[idx];
            let neighbors = density.get_neighbors(idx);
            
            // Find components of already-processed neighbors
            let mut neighbor_components: Vec<usize> = neighbors
                .iter()
                .filter(|&&n| component_birth[n].is_some())
                .map(|&n| uf.find(n))
                .collect();
            neighbor_components.sort();
            neighbor_components.dedup();
            
            if neighbor_components.is_empty() {
                // New component born
                component_birth[idx] = Some(value);
            } else {
                // Merge components - older component survives
                let oldest = neighbor_components[0];
                component_birth[idx] = component_birth[oldest];
                
                for &comp in &neighbor_components[1..] {
                    // Younger component dies
                    let birth = component_birth[uf.find(comp)].unwrap();
                    pairs.push(BirthDeathPair {
                        birth,
                        death: value,
                        location: density.index_to_position(comp),
                        dimension: 0,
                    });
                    uf.union(oldest, comp);
                }
                uf.union(oldest, idx);
            }
        }
        
        // Add infinite persistence for surviving components
        for i in 0..values.len() {
            if uf.find(i) == i {
                if let Some(birth) = component_birth[i] {
                    pairs.push(BirthDeathPair {
                        birth,
                        death: f64::NEG_INFINITY,
                        location: density.index_to_position(i),
                        dimension: 0,
                    });
                }
            }
        }
        
        PersistenceDiagram { pairs }
    }
    
    /// Get features above persistence threshold
    pub fn significant_features(&self, threshold: f64) -> Vec<&BirthDeathPair> {
        self.pairs
            .iter()
            .filter(|p| p.persistence() > threshold)
            .collect()
    }
    
    /// Estimate noise level from persistence distribution
    pub fn estimate_noise_threshold(&self, significance: f64) -> f64 {
        // Fit exponential to short-persistence features
        let persistences: Vec<f64> = self.pairs
            .iter()
            .map(|p| p.persistence())
            .filter(|&p| p.is_finite() && p > 0.0)
            .collect();
        
        if persistences.is_empty() {
            return 0.0;
        }
        
        // Use quantile as threshold
        let mut sorted = persistences.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let idx = ((1.0 - significance) * sorted.len() as f64) as usize;
        sorted[idx.min(sorted.len() - 1)]
    }
}

/// Union-Find with path compression
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }
    
    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }
    
    fn union(&mut self, x: usize, y: usize) {
        let px = self.find(x);
        let py = self.find(y);
        if px == py {
            return;
        }
        if self.rank[px] < self.rank[py] {
            self.parent[px] = py;
        } else if self.rank[px] > self.rank[py] {
            self.parent[py] = px;
        } else {
            self.parent[py] = px;
            self.rank[px] += 1;
        }
    }
}
```

---

## Part 5: Database Schema

### 5.1 SQLite Schema

```sql
-- migrations/001_initial.sql

-- Projects
CREATE TABLE projects (
    project_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    sequence TEXT,
    molecule_type TEXT CHECK (molecule_type IN ('peptide', 'protein', 'natural_product', 'small_molecule', 'oligonucleotide'))
);

-- Experiments
CREATE TABLE experiments (
    experiment_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(project_id),
    name TEXT NOT NULL,
    experiment_type TEXT NOT NULL,
    source_file TEXT,
    dimensions INTEGER NOT NULL,
    metadata JSON,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Spectra (references to Zarr storage)
CREATE TABLE spectra (
    spectrum_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id),
    zarr_path TEXT NOT NULL,
    noise_level REAL,
    processing_params JSON
);

-- Chemical shifts (assigned)
CREATE TABLE chemical_shifts (
    shift_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(project_id),
    atom_id TEXT NOT NULL,  -- Format: chain.residue.atom (e.g., "A.5.HN")
    value REAL NOT NULL,
    error REAL,
    confidence REAL DEFAULT 1.0,
    source TEXT,  -- 'crystallized', 'manual', 'imported'
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(project_id, atom_id)
);

CREATE INDEX idx_shifts_atom ON chemical_shifts(atom_id);
CREATE INDEX idx_shifts_value ON chemical_shifts(value);

-- Peak states (crystallization tracking)
CREATE TABLE peak_states (
    peak_state_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(project_id),
    state_type TEXT NOT NULL CHECK (state_type IN ('diffuse', 'nucleating', 'crystallized')),
    position JSON NOT NULL,  -- Array of chemical shifts
    covariance JSON,         -- Uncertainty matrix
    persistence REAL,
    entropy REAL,
    evidence_sources JSON,   -- Array of experiment types
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    crystallized_at TEXT
);

-- Peak assignments
CREATE TABLE peak_assignments (
    assignment_id TEXT PRIMARY KEY,
    peak_state_id TEXT NOT NULL REFERENCES peak_states(peak_state_id),
    dimension INTEGER NOT NULL,
    atom_id TEXT NOT NULL,
    probability REAL NOT NULL DEFAULT 1.0,
    UNIQUE(peak_state_id, dimension, atom_id)
);

CREATE INDEX idx_assignments_atom ON peak_assignments(atom_id);

-- Distance constraints
CREATE TABLE distance_constraints (
    constraint_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(project_id),
    atom1_id TEXT NOT NULL,
    atom2_id TEXT NOT NULL,
    lower_bound REAL,
    upper_bound REAL,
    target REAL,
    weight REAL DEFAULT 1.0,
    source_peak_id TEXT REFERENCES peak_states(peak_state_id),
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Dihedral constraints  
CREATE TABLE dihedral_constraints (
    constraint_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(project_id),
    atom1_id TEXT NOT NULL,
    atom2_id TEXT NOT NULL,
    atom3_id TEXT NOT NULL,
    atom4_id TEXT NOT NULL,
    angle REAL NOT NULL,
    error REAL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- BMRB statistics cache
CREATE TABLE bmrb_statistics (
    residue_type TEXT NOT NULL,
    atom_name TEXT NOT NULL,
    mean_shift REAL NOT NULL,
    std_shift REAL NOT NULL,
    sample_count INTEGER,
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (residue_type, atom_name)
);
```

```sql
-- migrations/002_density_fields.sql

-- Density field snapshots (for visualization/debugging)
CREATE TABLE density_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(project_id),
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    particle_count INTEGER,
    gmm_components INTEGER,
    crystallization_progress REAL,
    data_path TEXT  -- Path to serialized density field
);

-- Evidence accumulation log
CREATE TABLE evidence_log (
    log_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(project_id),
    experiment_id TEXT REFERENCES experiments(experiment_id),
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    observation_count INTEGER,
    log_likelihood REAL,
    density_snapshot_id TEXT REFERENCES density_snapshots(snapshot_id)
);
```

```sql
-- migrations/003_provenance.sql

-- Provenance tracking (git-like)
CREATE TABLE commits (
    commit_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(project_id),
    parent_commit_id TEXT REFERENCES commits(commit_id),
    message TEXT NOT NULL,
    author TEXT NOT NULL DEFAULT 'system',
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    state_hash TEXT NOT NULL  -- Hash of serialized state
);

CREATE INDEX idx_commits_project ON commits(project_id);

-- Branches
CREATE TABLE branches (
    branch_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(project_id),
    name TEXT NOT NULL,
    head_commit_id TEXT REFERENCES commits(commit_id),
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(project_id, name)
);

-- Changes tracked per commit
CREATE TABLE commit_changes (
    change_id TEXT PRIMARY KEY,
    commit_id TEXT NOT NULL REFERENCES commits(commit_id),
    entity_type TEXT NOT NULL,  -- 'peak', 'assignment', 'constraint', etc.
    entity_id TEXT NOT NULL,
    change_type TEXT NOT NULL CHECK (change_type IN ('create', 'update', 'delete')),
    old_value JSON,
    new_value JSON
);
```

---

## Part 6: Tauri Commands (API)

### 6.1 Command Definitions

```rust
// src-tauri/src/commands/mod.rs

pub mod spectrum;
pub mod density;
pub mod crystallize;
pub mod assignment;
pub mod structure;
pub mod project;

use tauri::Manager;

pub fn register_commands(app: &mut tauri::App) -> Result<(), Box<dyn std::error::Error>> {
    Ok(())
}
```

```rust
// src-tauri/src/commands/spectrum.rs

use tauri::State;
use crate::state::AppState;
use crate::error::Result;
use crate::data::spectrum::{Spectrum, ExperimentType};

/// Load spectrum from Bruker directory
#[tauri::command]
pub async fn load_bruker_spectrum(
    path: String,
    state: State<'_, AppState>,
) -> Result<Spectrum> {
    let spectrum = crate::io::bruker::read_bruker(&path)?;
    
    // Store in state
    let mut spectra = state.spectra.lock().await;
    spectra.insert(spectrum.id, spectrum.clone());
    
    Ok(spectrum)
}

/// Process spectrum (FFT, phasing, baseline)
#[tauri::command]
pub async fn process_spectrum(
    spectrum_id: uuid::Uuid,
    processing_params: ProcessingParams,
    state: State<'_, AppState>,
) -> Result<Spectrum> {
    let mut spectra = state.spectra.lock().await;
    let spectrum = spectra.get_mut(&spectrum_id)
        .ok_or(crate::error::NmrError::NotFound)?;
    
    // Apply processing
    if let Some(apod) = processing_params.apodization {
        crate::processing::apodization::apply(spectrum, apod)?;
    }
    
    crate::processing::fft::forward_fft(spectrum)?;
    
    if let Some(phase) = processing_params.phase_correction {
        crate::processing::phasing::apply(spectrum, phase)?;
    }
    
    if processing_params.baseline_correction {
        crate::processing::baseline::correct(spectrum)?;
    }
    
    Ok(spectrum.clone())
}

#[derive(serde::Deserialize)]
pub struct ProcessingParams {
    pub apodization: Option<ApodizationType>,
    pub phase_correction: Option<PhaseParams>,
    pub baseline_correction: bool,
}
```

```rust
// src-tauri/src/commands/density.rs

use tauri::State;
use crate::state::AppState;
use crate::error::Result;
use crate::crystallize::{CrystallizationState, CrystallizationConfig};

/// Initialize density field from sequence
#[tauri::command]
pub async fn initialize_density(
    project_id: uuid::Uuid,
    sequence: String,
    config: Option<CrystallizationConfig>,
    state: State<'_, AppState>,
) -> Result<DensityFieldSummary> {
    let config = config.unwrap_or_default();
    
    // Load BMRB priors
    let bmrb = state.bmrb_cache.lock().await;
    
    // Initialize crystallization state
    let crystal_state = CrystallizationState::<2>::initialize(
        &sequence,
        &bmrb,
        &config,
    );
    
    // Store state
    let mut projects = state.crystallization_states.lock().await;
    projects.insert(project_id, crystal_state);
    
    Ok(DensityFieldSummary {
        particle_count: config.particle_count,
        peak_states: 0,
        crystallization_progress: 0.0,
    })
}

/// Add experiment evidence to density field
#[tauri::command]
pub async fn add_experiment_evidence(
    project_id: uuid::Uuid,
    experiment_id: uuid::Uuid,
    state: State<'_, AppState>,
) -> Result<CrystallizationUpdate> {
    let spectra = state.spectra.lock().await;
    let experiment = spectra.get(&experiment_id)
        .ok_or(crate::error::NmrError::NotFound)?;
    
    let config = CrystallizationConfig::default();
    
    let mut projects = state.crystallization_states.lock().await;
    let crystal_state = projects.get_mut(&project_id)
        .ok_or(crate::error::NmrError::NotFound)?;
    
    // Update with experiment
    crystal_state.update_with_experiment(experiment, &config);
    
    Ok(CrystallizationUpdate {
        new_crystallized: crystal_state.newly_crystallized_count(),
        new_nucleating: crystal_state.newly_nucleating_count(),
        progress: crystal_state.crystallization_progress,
        persistence_diagram: crystal_state.persistence.to_summary(),
    })
}

/// Get current density field for visualization
#[tauri::command]
pub async fn get_density_field(
    project_id: uuid::Uuid,
    region: Option<ChemShiftRegion>,
    resolution: usize,
    state: State<'_, AppState>,
) -> Result<DensityFieldData> {
    let projects = state.crystallization_states.lock().await;
    let crystal_state = projects.get(&project_id)
        .ok_or(crate::error::NmrError::NotFound)?;
    
    // Evaluate density on grid for visualization
    let grid = crystal_state.density.evaluate_on_grid(region, resolution);
    
    Ok(DensityFieldData {
        values: grid.values,
        x_range: grid.x_range,
        y_range: grid.y_range,
        peak_states: crystal_state.peak_states.iter()
            .map(|ps| ps.to_summary())
            .collect(),
    })
}

#[derive(serde::Serialize)]
pub struct DensityFieldSummary {
    pub particle_count: usize,
    pub peak_states: usize,
    pub crystallization_progress: f64,
}

#[derive(serde::Serialize)]
pub struct CrystallizationUpdate {
    pub new_crystallized: usize,
    pub new_nucleating: usize,
    pub progress: f64,
    pub persistence_diagram: PersistenceSummary,
}
```

```rust
// src-tauri/src/commands/assignment.rs

use tauri::State;
use crate::state::AppState;
use crate::error::Result;
use crate::inference::factor_graph::FactorGraph;

/// Run global assignment optimization
#[tauri::command]
pub async fn run_global_assignment(
    project_id: uuid::Uuid,
    config: AssignmentConfig,
    state: State<'_, AppState>,
) -> Result<AssignmentResult> {
    // Build factor graph from all data
    let mut factor_graph = FactorGraph::new();
    
    // Add molecular graph
    let projects = state.projects.lock().await;
    let project = projects.get(&project_id)
        .ok_or(crate::error::NmrError::NotFound)?;
    
    factor_graph.add_molecular_structure(&project.molecule);
    
    // Add BMRB priors
    let bmrb = state.bmrb_cache.lock().await;
    factor_graph.add_bmrb_priors(&bmrb, &project.molecule);
    
    // Add crystallized peaks
    let crystal_states = state.crystallization_states.lock().await;
    if let Some(crystal) = crystal_states.get(&project_id) {
        for peak_state in &crystal.peak_states {
            if let PeakState::Crystallized(peak) = peak_state {
                factor_graph.add_peak_factors(peak);
            }
        }
    }
    
    // Add experiment-specific factors
    let spectra = state.spectra.lock().await;
    for spectrum in spectra.values() {
        factor_graph.add_experiment(spectrum);
    }
    
    // Run inference
    let inference_result = factor_graph.infer(
        config.max_iterations,
        config.convergence_threshold,
    );
    
    // Extract assignments
    let assignments = extract_assignments(&inference_result.marginals);
    
    Ok(AssignmentResult {
        converged: inference_result.converged,
        iterations: inference_result.iterations,
        assignments,
        confidence_scores: compute_confidence_scores(&inference_result.marginals),
    })
}

#[derive(serde::Deserialize)]
pub struct AssignmentConfig {
    pub max_iterations: usize,
    pub convergence_threshold: f64,
    pub use_alphafold_prior: bool,
}

#[derive(serde::Serialize)]
pub struct AssignmentResult {
    pub converged: bool,
    pub iterations: usize,
    pub assignments: Vec<AtomAssignment>,
    pub confidence_scores: HashMap<String, f64>,
}

#[derive(serde::Serialize)]
pub struct AtomAssignment {
    pub atom_id: String,
    pub chemical_shift: f64,
    pub uncertainty: f64,
    pub confidence: f64,
    pub alternative_assignments: Vec<(f64, f64)>,  // (shift, probability)
}
```

---

## Part 7: Frontend Components

### 7.1 Main Application Structure

```tsx
// src/App.tsx

import { useState } from 'react';
import { Sidebar } from './components/common/Sidebar';
import { Toolbar } from './components/common/Toolbar';
import { StatusBar } from './components/common/StatusBar';
import { SpectrumViewer } from './components/spectrum/SpectrumViewer';
import { DensityView } from './components/density/DensityView';
import { AssignmentTable } from './components/assignment/AssignmentTable';
import { SequenceViewer } from './components/molecule/SequenceViewer';
import { useProjectStore } from './stores/projectStore';

export function App() {
  const [activeView, setActiveView] = useState<'spectrum' | 'density' | 'assignment'>('spectrum');
  const project = useProjectStore((s) => s.currentProject);

  return (
    <div className="flex h-screen bg-gray-900 text-gray-100">
      <Sidebar onViewChange={setActiveView} activeView={activeView} />
      
      <div className="flex-1 flex flex-col">
        <Toolbar />
        
        <main className="flex-1 flex overflow-hidden">
          {/* Main view area */}
          <div className="flex-1 p-4">
            {activeView === 'spectrum' && <SpectrumViewer />}
            {activeView === 'density' && <DensityView />}
            {activeView === 'assignment' && <AssignmentTable />}
          </div>
          
          {/* Side panel */}
          <div className="w-80 border-l border-gray-700 p-4 overflow-y-auto">
            <SequenceViewer sequence={project?.sequence} />
          </div>
        </main>
        
        <StatusBar />
      </div>
    </div>
  );
}
```

### 7.2 Density Visualization Component

```tsx
// src/components/density/DensityView.tsx

import { useEffect, useRef, useCallback } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { useDensityStore } from '../../stores/densityStore';
import { CrystallizationProgress } from './CrystallizationProgress';
import { UncertaintyEllipse } from './UncertaintyEllipse';

interface DensityFieldData {
  values: number[][];
  x_range: [number, number];
  y_range: [number, number];
  peak_states: PeakStateSummary[];
}

interface PeakStateSummary {
  state_type: 'diffuse' | 'nucleating' | 'crystallized';
  position: number[];
  covariance?: number[][];
  persistence?: number;
  entropy?: number;
}

export function DensityView() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { projectId, densityField, peakStates, progress, setDensityField } = useDensityStore();

  // Fetch density field data
  const fetchDensity = useCallback(async () => {
    if (!projectId) return;
    
    try {
      const data = await invoke<DensityFieldData>('get_density_field', {
        projectId,
        resolution: 256,
      });
      setDensityField(data);
    } catch (err) {
      console.error('Failed to fetch density:', err);
    }
  }, [projectId, setDensityField]);

  useEffect(() => {
    fetchDensity();
  }, [fetchDensity]);

  // Render density field
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !densityField) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const { values, x_range, y_range, peak_states } = densityField;
    const width = canvas.width;
    const height = canvas.height;

    // Clear canvas
    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(0, 0, width, height);

    // Draw density field as heatmap
    const imageData = ctx.createImageData(width, height);
    const maxVal = Math.max(...values.flat());

    for (let y = 0; y < values.length; y++) {
      for (let x = 0; x < values[0].length; x++) {
        const val = values[y][x] / maxVal;
        const idx = (y * width + x) * 4;
        
        // Color based on density value
        const color = densityColormap(val);
        imageData.data[idx] = color.r;
        imageData.data[idx + 1] = color.g;
        imageData.data[idx + 2] = color.b;
        imageData.data[idx + 3] = 255;
      }
    }
    ctx.putImageData(imageData, 0, 0);

    // Draw peak states
    for (const peak of peak_states) {
      const x = ((peak.position[0] - x_range[0]) / (x_range[1] - x_range[0])) * width;
      const y = ((peak.position[1] - y_range[0]) / (y_range[1] - y_range[0])) * height;

      switch (peak.state_type) {
        case 'crystallized':
          // Solid marker
          ctx.fillStyle = '#00ff88';
          ctx.beginPath();
          ctx.arc(x, y, 6, 0, Math.PI * 2);
          ctx.fill();
          break;
          
        case 'nucleating':
          // Pulsing marker with uncertainty ellipse
          ctx.strokeStyle = '#ffaa00';
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(x, y, 8, 0, Math.PI * 2);
          ctx.stroke();
          
          // Draw uncertainty ellipse if covariance available
          if (peak.covariance) {
            drawUncertaintyEllipse(ctx, x, y, peak.covariance, width, height, x_range, y_range);
          }
          break;
          
        case 'diffuse':
          // Dim indicator
          ctx.fillStyle = 'rgba(255, 255, 255, 0.3)';
          ctx.beginPath();
          ctx.arc(x, y, 4, 0, Math.PI * 2);
          ctx.fill();
          break;
      }
    }
  }, [densityField]);

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold">Density Field</h2>
        <CrystallizationProgress progress={progress} />
      </div>
      
      <div className="flex-1 relative">
        <canvas
          ref={canvasRef}
          width={800}
          height={600}
          className="w-full h-full object-contain"
        />
        
        {/* Legend */}
        <div className="absolute bottom-4 right-4 bg-gray-800 p-2 rounded text-sm">
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-full bg-green-400"></span>
            <span>Crystallized</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-full border-2 border-yellow-400"></span>
            <span>Nucleating</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-full bg-white/30"></span>
            <span>Diffuse</span>
          </div>
        </div>
      </div>
    </div>
  );
}

// Colormap for density visualization
function densityColormap(value: number): { r: number; g: number; b: number } {
  // Viridis-like colormap
  const colors = [
    { r: 68, g: 1, b: 84 },     // 0.0
    { r: 72, g: 40, b: 120 },   // 0.25
    { r: 62, g: 74, b: 137 },   // 0.5
    { r: 49, g: 104, b: 142 },  // 0.75
    { r: 253, g: 231, b: 37 },  // 1.0
  ];
  
  const idx = Math.min(Math.floor(value * (colors.length - 1)), colors.length - 2);
  const t = value * (colors.length - 1) - idx;
  
  return {
    r: Math.round(colors[idx].r + t * (colors[idx + 1].r - colors[idx].r)),
    g: Math.round(colors[idx].g + t * (colors[idx + 1].g - colors[idx].g)),
    b: Math.round(colors[idx].b + t * (colors[idx + 1].b - colors[idx].b)),
  };
}

// Draw uncertainty ellipse from covariance matrix
function drawUncertaintyEllipse(
  ctx: CanvasRenderingContext2D,
  cx: number,
  cy: number,
  cov: number[][],
  canvasWidth: number,
  canvasHeight: number,
  xRange: [number, number],
  yRange: [number, number]
) {
  // Eigendecomposition of 2x2 covariance matrix
  const a = cov[0][0];
  const b = cov[0][1];
  const d = cov[1][1];
  
  const trace = a + d;
  const det = a * d - b * b;
  const discriminant = Math.sqrt(trace * trace / 4 - det);
  
  const lambda1 = trace / 2 + discriminant;
  const lambda2 = trace / 2 - discriminant;
  
  // Semi-axes (2 sigma for 95% confidence)
  const scale = 2.0;
  const rx = Math.sqrt(lambda1) * scale * canvasWidth / (xRange[1] - xRange[0]);
  const ry = Math.sqrt(lambda2) * scale * canvasHeight / (yRange[1] - yRange[0]);
  
  // Rotation angle
  const angle = Math.atan2(2 * b, a - d) / 2;
  
  ctx.save();
  ctx.translate(cx, cy);
  ctx.rotate(angle);
  ctx.strokeStyle = 'rgba(255, 170, 0, 0.5)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.ellipse(0, 0, rx, ry, 0, 0, Math.PI * 2);
  ctx.stroke();
  ctx.restore();
}
```

### 7.3 Zustand Store for Density State

```typescript
// src/stores/densityStore.ts

import { create } from 'zustand';

interface DensityFieldData {
  values: number[][];
  x_range: [number, number];
  y_range: [number, number];
  peak_states: PeakStateSummary[];
}

interface PeakStateSummary {
  state_type: 'diffuse' | 'nucleating' | 'crystallized';
  position: number[];
  covariance?: number[][];
  persistence?: number;
  entropy?: number;
}

interface DensityStore {
  projectId: string | null;
  densityField: DensityFieldData | null;
  peakStates: PeakStateSummary[];
  progress: number;
  
  setProjectId: (id: string) => void;
  setDensityField: (data: DensityFieldData) => void;
  setProgress: (progress: number) => void;
  reset: () => void;
}

export const useDensityStore = create<DensityStore>((set) => ({
  projectId: null,
  densityField: null,
  peakStates: [],
  progress: 0,
  
  setProjectId: (id) => set({ projectId: id }),
  
  setDensityField: (data) => set({
    densityField: data,
    peakStates: data.peak_states,
  }),
  
  setProgress: (progress) => set({ progress }),
  
  reset: () => set({
    projectId: null,
    densityField: null,
    peakStates: [],
    progress: 0,
  }),
}));
```

---

## Part 8: Implementation Roadmap

### Phase 1: Foundation (Weeks 1-3)

**Week 1: Project Setup**
- [ ] Initialize Tauri 2.0 project with React + TypeScript
- [ ] Configure Cargo.toml with all dependencies
- [ ] Set up SQLite with migrations
- [ ] Create error handling infrastructure

**Week 2: Data Layer**
- [ ] Implement Bruker format reader
- [ ] Create spectrum data structures
- [ ] Build molecular graph with petgraph
- [ ] Set up Zarr storage for spectra

**Week 3: Basic Processing**
- [ ] Implement FFT with rustfft
- [ ] Add apodization functions
- [ ] Phase correction algorithms
- [ ] Baseline correction

### Phase 2: Density Crystallization (Weeks 4-6)

**Week 4: Density Representation**
- [ ] Particle cloud implementation
- [ ] Variational GMM
- [ ] Kernel density estimation
- [ ] BMRB prior integration

**Week 5: Persistence Topology**
- [ ] Union-find with persistence tracking
- [ ] Sublevel set filtration
- [ ] Persistence diagram computation
- [ ] Noise threshold estimation

**Week 6: Crystallization Engine**
- [ ] Crystallization criteria
- [ ] Entropy computation
- [ ] MDL model selection
- [ ] Evidence accumulation pipeline

### Phase 3: Inference (Weeks 7-9)

**Week 7: Factor Graph**
- [ ] Variable and factor node types
- [ ] Graph construction from experiments
- [ ] BMRB prior factors
- [ ] Peak consistency factors

**Week 8: Belief Propagation**
- [ ] Message passing implementation
- [ ] Convergence detection
- [ ] Marginal computation
- [ ] Assignment extraction

**Week 9: Multi-Experiment Fusion**
- [ ] HSQC factor integration
- [ ] NOESY distance factors
- [ ] TOCSY spin system factors
- [ ] Global scoring function

### Phase 4: Frontend (Weeks 10-12)

**Week 10: Core UI**
- [ ] Spectrum viewer (1D/2D)
- [ ] WebGL rendering
- [ ] Pan/zoom interactions
- [ ] Peak picking UI

**Week 11: Density Visualization**
- [ ] Density field heatmap
- [ ] Peak state markers
- [ ] Uncertainty ellipses
- [ ] Crystallization progress

**Week 12: Assignment Interface**
- [ ] Assignment table
- [ ] Confidence indicators
- [ ] Sequence coverage map
- [ ] Linked views (brushing)

### Phase 5: Polish (Weeks 13-14)

**Week 13: Testing & Validation**
- [ ] Unit tests for core algorithms
- [ ] Integration tests with real data
- [ ] Benchmark critical paths
- [ ] Numerical validation

**Week 14: Documentation & Release**
- [ ] API documentation
- [ ] User guide
- [ ] Example workflows
- [ ] GitHub release

---

## Part 9: Key Performance Targets

| Operation | Target | Method |
|-----------|--------|--------|
| 1D FFT (16K points) | < 1 ms | rustfft SIMD |
| 2D FFT (2K×2K) | < 50 ms | Parallel rayon |
| Persistence computation | < 100 ms | Optimized union-find |
| Belief propagation (1000 vars) | < 1 s | Message caching |
| Density KDE (10K particles) | < 10 ms | Spatial indexing |
| Spectrum rendering (60 FPS) | 16 ms/frame | WebGL |
| Memory (typical session) | < 500 MB | Streaming, sparse |

---

## Part 10: Extension Points

### 10.1 Future Molecule Types

```rust
// The architecture supports extension to:
pub enum MoleculeType {
    Peptide,           // Initial focus
    Protein,           // Larger sequences
    NaturalProduct,    // Complex ring systems
    SmallMolecule,     // Metabolomics
    Oligonucleotide,   // DNA/RNA
}

// Each type requires:
// 1. Molecular graph construction
// 2. BMRB/database priors
// 3. Experiment-specific factors
// 4. Validation rules
```

### 10.2 Dynamics Integration (Future)

```rust
// Future support for relaxation data
pub struct RelaxationData {
    pub experiment_type: RelaxationType,
    pub field_strength: f64,
    pub residue_data: Vec<ResidueRelaxation>,
}

pub enum RelaxationType {
    T1,
    T2,
    HetNOE,
    CPMG,
    CEST,
}

// Model-free analysis output
pub struct ModelFreeParameters {
    pub s2: f64,           // Order parameter
    pub tau_e: f64,        // Internal motion timescale
    pub rex: Option<f64>,  // Exchange contribution
}
```

### 10.3 AlphaFold Integration

```rust
// Future AlphaFold prior integration
pub struct AlphaFoldPrior {
    pub structure: Vec<AtomPosition>,
    pub plddt: Vec<f64>,  // Per-residue confidence
    
    // Derive chemical shift priors from structure
    pub fn predict_shifts(&self) -> Vec<ShiftPrediction>;
    
    // Add structural restraints to factor graph
    pub fn add_structure_factors(&self, graph: &mut FactorGraph);
}
```

---

## Appendix A: Cargo.toml

```toml
[package]
name = "crystalline"
version = "0.1.0"
edition = "2021"
rust-version = "1.75"
description = "Next-generation NMR spectroscopy platform"
license = "MIT"
repository = "https://github.com/your-org/crystalline"

[build-dependencies]
tauri-build = { version = "2.0", features = [] }

[dependencies]
# Tauri
tauri = { version = "2.9", features = ["tray-icon", "protocol-asset"] }
tauri-plugin-dialog = "2.0"
tauri-plugin-fs = "2.0"
tauri-plugin-shell = "2.0"

# Async
tokio = { version = "1.42", features = ["full"] }

# Serialization
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
bincode = "1.3"

# Arrays and math
ndarray = { version = "0.16", features = ["serde", "rayon"] }
ndarray-linalg = { version = "0.16", features = ["openblas-static"] }
nalgebra = { version = "0.33", features = ["serde-serialize"] }
num = "0.4"
num-complex = { version = "0.4", features = ["serde"] }

# FFT
rustfft = "6.2"
realfft = "3.4"

# Graph structures
petgraph = { version = "0.8", features = ["serde-1"] }

# Database
rusqlite = { version = "0.32", features = ["bundled", "blob", "array"] }
rusqlite_migration = "2.0"

# ML
ort = { version = "2.0", features = ["download-binaries"] }

# File formats
zarr = "0.1"
parquet = { version = "53.0", features = ["async"] }

# Error handling
thiserror = "2.0"
anyhow = "1.0"

# Utilities
uuid = { version = "1.11", features = ["v4", "serde"] }
chrono = { version = "0.4", features = ["serde"] }
rayon = "1.10"
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
blake3 = "1.5"

[dev-dependencies]
proptest = "1.5"
criterion = { version = "0.5", features = ["html_reports"] }
approx = "0.5"
tempfile = "3.14"

[[bench]]
name = "fft_benchmark"
harness = false

[[bench]]
name = "inference_benchmark"
harness = false
```

---

## Appendix B: Quick Start Commands

```bash
# Create project
npm create tauri-app@latest crystalline -- --template react-ts
cd crystalline

# Add Rust dependencies
# (copy Cargo.toml from above)

# Install frontend dependencies
npm install zustand @tauri-apps/api three

# Development
npm run tauri dev

# Build release
npm run tauri build
```

---

This specification provides a complete blueprint for implementing CRYSTALLINE. The document can be fed directly to Claude Code to begin implementation, starting with Phase 1 foundation work.
