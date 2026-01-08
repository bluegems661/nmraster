# Next-generation NMR platform with simultaneous multi-experiment analysis

The paradigm shift from sequential to simultaneous analysis of NMR data represents a fundamental breakthrough in computational spectroscopy. While human analysts process HSQC, NOESY, and TOCSY spectra one at a time, software can reason about all experimental observations together—finding correlations, resolving ambiguities, and detecting patterns invisible to sequential workflows. Modern approaches like ARTINA achieve **91.36% assignment accuracy** by processing 25+ spectrum types simultaneously through deep learning and global optimization. This document provides a complete implementation blueprint for building such a platform in Rust/Tauri.

The core innovation centers on treating NMR analysis as a global optimization problem over a probabilistic graphical model. Every peak in every spectrum becomes a node in an interconnected factor graph, where message-passing algorithms propagate constraints across experiment types. Chemical shifts assigned from an HSQC immediately constrain possible NOESY cross-peak interpretations, which in turn validate or refute TOCSY spin system groupings. This simultaneous reasoning—impossible for human analysts—enables the software to resolve ambiguities that would otherwise require manual intervention.

---

## Alternative Analysis Mode: CRYSTALLINE

**CRYSTALLINE** (Continuous-density Recognition Yielding Structures Through Adaptive Lattice-based Inference in NMR Experiments) provides an alternative analysis paradigm alongside the traditional factor graph approach described above.

### The Density Crystallization Paradigm

Where traditional NMR analysis forces binary peak/no-peak decisions upfront, CRYSTALLINE maintains **continuous probability densities** over chemical shift space. Peaks "crystallize" into discrete entities only when sufficient multi-experiment evidence accumulates—naturally handling crowded regions where traditional methods fail.

**Traditional workflow:**
```
Raw Spectrum → Peak Pick → Assign → Calculate Structure
     ↓              ↓         ↓              ↓
   (data)      (binary)   (manual)      (restraints)
```

**CRYSTALLINE workflow:**
```
All Spectra → Unified Density Field → Evidence Accumulation → Crystallization → Structure
     ↓               ↓                        ↓                    ↓              ↓
 (parallel)    (probabilistic)         (factor graph)         (threshold)    (ensemble)
```

### Key Innovations

1. **Continuous Density Representation**: Peaks exist as probability distributions, not binary entities. Initialized from BMRB priors, refined by experimental evidence.

2. **Topological Persistence for Peak Significance**: Parameter-free peak detection using persistent homology. Noise artifacts have low persistence; true peaks persist across filtration levels.

3. **Three Peak States**:
   - **Diffuse**: Just noise-like density, not yet localized
   - **Nucleating**: Gathering evidence, uncertainty narrowing
   - **Crystallized**: Definite peak with quantified uncertainty

4. **Information-Theoretic Crystallization**: Peaks "emerge" when ALL criteria are met:
   - Entropy below threshold (position well-determined, <0.02 ppm)
   - Persistence above noise (topologically significant)
   - MDL favors peak model (model selection confirms reality)
   - Multiple experiments agree (corroborating evidence)

5. **Uncertainty Propagation**: Full posterior distributions flow through to structure calculation, enabling ensemble generation that properly represents conformational ambiguity.

6. **Graceful Ambiguity Handling**: Crowded regions remain as density until resolvable. No forced wrong decisions—uncertainty propagates honestly to downstream analysis.

---

## The mathematical foundation for simultaneous analysis

The joint probability distribution over all NMR observables factorizes as a product of local factors encoding physical constraints. For a molecule with atoms A, spectra S, and peaks P, the global scoring function takes the form:

```
P(assignments | data) ∝ ∏ᵢ φᵢ(chemical_shift_consistency)
                       × ∏ⱼ φⱼ(NOE_distance_compatibility)
                       × ∏ₖ φₖ(BMRB_statistics)
                       × ∏ₗ φₗ(sequential_connectivity)
```

The FLYA algorithm pioneered this approach by treating assignment as expected-to-observed peak mapping. For each atom with Gaussian prior N(μₐ, σₐ) and observations {xₗᵃ}, the cost function becomes:

```
cost(a, {xₗᵃ}) = -log 𝔼ᵤ~N(μₐ,σₐ) [∏ₗ f(xₗᵃ | μ, σₗ)]
```

This formulation enables evolutionary optimization with local refinement—random initialization from BMRB statistics, followed by genetic algorithm recombination of assignments, achieving **96-99% backbone accuracy** for automated workflows.

Belief propagation on factor graphs provides the inference engine. Messages flow between variable nodes (chemical shifts, atom assignments) and factor nodes (constraint potentials):

```
μₓ→f(x) = ∏_{f'∈n(x)\f} μf'→x(x)
μf→x(x) = Σ_{x'} φf(x, x') ∏_{x'∈n(f)\x} μx'→f(x')
```

For NMR data, loopy belief propagation handles the non-tree graph structure, with convergence typically achieved within 10-50 iterations.

---

## Graph database schema for unified NMR data

The data model must capture relationships between peaks across all experiment types while supporting efficient cross-experiment queries. A hybrid architecture combining SQLite for structured data, Neo4j-style graph relationships, and Zarr for spectral arrays provides optimal query flexibility.

### Core node types (Neo4j representation)

```cypher
(:Molecule {name, sequence_length})
(:Chain {chain_code, polymer_type})
(:Residue {sequence_code, residue_name, one_letter_code})
(:Atom {atom_name, element, isotope_number})
(:ChemicalShift {value, error, ambiguity_code, list_id})
(:Experiment {name, type, date, spectrometer_frequency})
(:Spectrum {dimensions, nucleus_types[], sw[], num_points[]})
(:Peak {position[], intensity, volume, height, line_width[]})
(:DistanceConstraint {lower_bound, upper_bound, target, weight})
(:RelaxationData {type, field_strength, value, error})
```

### Critical relationship types

```cypher
// Molecular hierarchy
(Residue)-[:NEXT]->(Residue)  // Sequence connectivity
(Atom)-[:BONDED_TO]->(Atom)   // Covalent bonds

// Peak relationships
(Peak)-[:ASSIGNED_TO {dimension: int, probability: float}]->(Atom)
(Peak)-[:CORRELATES_WITH {mechanism: "NOESY"|"TOCSY"|"J-coupling"}]->(Peak)
(Peak)-[:SHARES_ASSIGNMENT]->(Peak)  // Cross-experiment validation

// Constraints
(DistanceConstraint)-[:DERIVED_FROM]->(Peak)
(DistanceConstraint)-[:RESTRAINS]->(Atom)
```

The `SHARES_ASSIGNMENT` relationship is critical for simultaneous analysis—it links peaks across HSQC, NOESY, and TOCSY that reference the same atoms, enabling constraint propagation to validate assignments globally.

### SQLite schema for core entities

```sql
CREATE TABLE chemical_shifts (
    shift_id INTEGER PRIMARY KEY,
    list_id INTEGER NOT NULL REFERENCES chemical_shift_lists(list_id),
    atom_id INTEGER NOT NULL REFERENCES atoms(atom_id),
    value REAL NOT NULL,
    error REAL,
    ambiguity_code INTEGER DEFAULT 1,
    confidence REAL DEFAULT 1.0,
    UNIQUE(list_id, atom_id)
);
CREATE INDEX idx_shifts_value ON chemical_shifts(value);

CREATE TABLE peak_assignments (
    assignment_id INTEGER PRIMARY KEY,
    peak_id INTEGER NOT NULL REFERENCES peaks(peak_id),
    dimension_index INTEGER NOT NULL,
    atom_id INTEGER REFERENCES atoms(atom_id),
    probability REAL DEFAULT 1.0,
    is_primary INTEGER DEFAULT 1
);
CREATE INDEX idx_assignments_atom ON peak_assignments(atom_id);

CREATE TABLE distance_constraints (
    constraint_id INTEGER PRIMARY KEY,
    atom1_id INTEGER NOT NULL REFERENCES atoms(atom_id),
    atom2_id INTEGER NOT NULL REFERENCES atoms(atom_id),
    lower_bound REAL,
    upper_bound REAL,
    peak_id INTEGER REFERENCES peaks(peak_id),
    weight REAL DEFAULT 1.0
);
```

---

## Dynamic NMR data integration architecture

Relaxation data (T1, T2, NOE, CPMG, CEST) captures molecular dynamics across timescales from picoseconds to seconds. Integration with static structural data requires a hierarchical model linking per-residue dynamics to atomic coordinates and constraints.

### Lipari-Szabo model-free analysis

The spectral density function for backbone dynamics:

```
J(ω) = (2/5) × [S²τc/(1 + (ωτc)²) + (1-S²)τ'/(1 + (ωτ')²)]
```

where S² is the generalized order parameter (**0-1, higher = more rigid**), τc is the overall correlation time, and τ' = τeτc/(τe + τc) combines internal motion (τe) with overall tumbling.

Experimental relaxation rates connect to spectral densities via:
- **R1** = d²[J(ωH-ωN) + 3J(ωN) + 6J(ωH+ωN)]/4 + c²J(ωN)
- **R2** = d²[4J(0) + J(ωH-ωN) + 3J(ωN) + 6J(ωH) + 6J(ωH+ωN)]/8 + c²[4J(0) + 3J(ωN)]/6
- **NOE** = 1 + (γH/γN)d²[6J(ωH+ωN) - J(ωH-ωN)]/(4R1)

### CPMG dispersion data structure

```json
{
  "residue_id": 45,
  "experiment_type": "CPMG",
  "field_strength_mhz": 800.0,
  "temperature_k": 298.0,
  "dispersion_points": [
    {"nu_cpmg_hz": 50, "r2eff": 15.2, "error": 0.3},
    {"nu_cpmg_hz": 100, "r2eff": 14.8, "error": 0.25},
    {"nu_cpmg_hz": 200, "r2eff": 13.9, "error": 0.22}
  ],
  "fitted_parameters": {
    "model": "CR72",
    "kex": 1500,
    "pB": 0.03,
    "delta_omega_ppm": 2.5,
    "R20": 12.1
  }
}
```

The Carver-Richards equation for two-state exchange:
```
R2eff = R2⁰ + (kex/2) - (1/τCP) × arccosh(D₊cosh(η₊) - D₋cos(η₋))
```

---

## Cross-domain inspiration: Multi-omics and cryo-EM patterns

The "all data at once" paradigm has been solved in several adjacent fields, providing transferable algorithmic patterns.

### MOFA (Multi-Omics Factor Analysis)

Uses Bayesian Group Factor Analysis to identify **K latent factors explaining variance across M data modalities**. Automatic Relevance Determination priors induce sparsity, revealing which factors are modality-specific versus shared. For NMR, this maps to:
- Treat each experiment type (HSQC, NOESY, TOCSY, relaxation) as a "view"
- Latent factors capture underlying structural/dynamic features
- Variance decomposition identifies experiment-specific vs. shared information

### Cryo-EM simultaneous refinement (RELION)

Regularized likelihood optimization with marginalization over orientational AND class assignments:
```
P(data | model) = ∏ᵢ Σclass Σorientation P(imageᵢ | class, orientation, model)
```

Key patterns transferable to NMR:
- **Iterative refinement with recycling**: Feed predictions back through the network
- **Bayesian polishing**: Per-particle (per-peak) error correction using global reference
- **Multi-reference classification**: Conformational heterogeneity as discrete classes

### IMP (Integrative Modeling Platform)

Restraint-based framework combining NMR, SAXS, cryo-EM, crosslinking:
```
Score = Σᵢ wᵢ × forward_model_disagreement(dataᵢ, structure)
```

Each data type encoded as a spatial restraint with:
- Forward model simulating observable from candidate structure
- Likelihood comparing simulated vs. observed
- Bayesian scoring integrating uncertainty

---

## Rust/Tauri project structure for Claude Code

```
nmr-platform/
├── src-tauri/
│   ├── Cargo.toml
│   ├── src/
│   │   ├── lib.rs                      # Tauri entry point
│   │   ├── main.rs                     # Windows subsystem config
│   │   ├── error.rs                    # thiserror error types
│   │   ├── commands/
│   │   │   ├── mod.rs
│   │   │   ├── spectrum.rs             # Load, process spectra
│   │   │   ├── assignment.rs           # Global assignment optimization
│   │   │   ├── database.rs             # CRUD operations
│   │   │   └── analysis.rs             # Peak picking, integration
│   │   ├── state/
│   │   │   ├── mod.rs
│   │   │   └── app_state.rs            # Mutex-wrapped application state
│   │   ├── processing/
│   │   │   ├── mod.rs
│   │   │   ├── fft.rs                  # rustfft integration
│   │   │   ├── phasing.rs              # Phase correction
│   │   │   ├── baseline.rs             # Baseline algorithms
│   │   │   └── apodization.rs          # Window functions
│   │   ├── data/
│   │   │   ├── mod.rs
│   │   │   ├── spectrum.rs             # Spectrum data structures
│   │   │   ├── experiment.rs           # Experiment metadata
│   │   │   ├── molecule.rs             # Molecular graph (petgraph)
│   │   │   └── constraint.rs           # NOE, dihedral constraints
│   │   ├── inference/
│   │   │   ├── mod.rs
│   │   │   ├── factor_graph.rs         # Factor graph construction
│   │   │   ├── belief_propagation.rs   # Message passing
│   │   │   ├── assignment.rs           # FLYA-style optimization
│   │   │   └── scoring.rs              # Multi-experiment scoring
│   │   ├── db/
│   │   │   ├── mod.rs
│   │   │   ├── connection.rs           # SQLite with WAL mode
│   │   │   ├── migrations.rs           # Schema versioning
│   │   │   └── queries.rs              # Prepared statements
│   │   └── ml/
│   │       ├── mod.rs
│   │       ├── model_registry.rs       # Version management
│   │       ├── inference.rs            # ONNX runtime (ort)
│   │       └── cache.rs                # Result caching
├── src/                                # React frontend
│   ├── main.tsx
│   ├── App.tsx
│   ├── components/
│   │   ├── spectrum/
│   │   │   ├── SpectrumViewer.tsx      # Main viewer component
│   │   │   ├── SpectrumCanvas.tsx      # WebGL/Canvas rendering
│   │   │   └── PeakList.tsx            # Interactive peak table
│   │   ├── assignment/
│   │   │   ├── AssignmentTable.tsx     # Chemical shift list
│   │   │   └── SpinSystemView.tsx      # Grouped spin systems
│   │   └── molecule/
│   │       └── SequenceViewer.tsx      # Sequence with annotations
│   ├── stores/
│   │   ├── spectrumStore.ts            # Zustand store
│   │   └── assignmentStore.ts
│   ├── hooks/
│   │   ├── useSpectrum.ts
│   │   └── useTauriCommand.ts
│   └── lib/
│       └── tauri.ts                    # Invoke wrappers
├── models/                             # ML model storage
│   ├── peak_picker/v1.0.0/model.onnx
│   └── assignment/v1.0.0/model.onnx
└── package.json
```

---

## Cargo.toml with specific dependencies

```toml
[package]
name = "nmr-platform"
version = "0.1.0"
edition = "2021"
rust-version = "1.75"

[build-dependencies]
tauri-build = { version = "2.0", features = [] }

[dependencies]
# Tauri Core
tauri = { version = "2.9", features = ["tray-icon", "protocol-asset"] }
tauri-plugin-dialog = "2.0"
tauri-plugin-fs = "2.0"

# Async Runtime
tokio = { version = "1.42", features = ["full", "sync", "rt-multi-thread"] }

# Serialization
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
bincode = "1.3"

# Matrix/Array Operations
ndarray = { version = "0.16", features = ["serde"] }
nalgebra = { version = "0.33", features = ["serde-serialize"] }

# Scientific Computing
num = "0.4"
num-complex = { version = "0.4", features = ["serde"] }
rustfft = "6.2"
realfft = "3.4"

# Database
rusqlite = { version = "0.32", features = ["bundled", "blob", "array"] }
rusqlite_migration = "2.0"

# Graph Data Structures
petgraph = { version = "0.8", features = ["serde-1"] }

# ML Integration
ort = { version = "2.0", features = ["download-binaries"] }

# Error Handling
thiserror = "2.0"
anyhow = "1.0"

# Utilities
uuid = { version = "1.11", features = ["v4", "serde"] }
chrono = { version = "0.4", features = ["serde"] }
rayon = "1.10"
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }

[dev-dependencies]
proptest = "1.5"
criterion = { version = "0.5", features = ["html_reports"] }
approx = "0.5"
tempfile = "3.14"
```

---

## Factor graph implementation for global assignment

```rust
// src-tauri/src/inference/factor_graph.rs
use petgraph::graph::{DiGraph, NodeIndex};
use ndarray::Array1;
use std::collections::HashMap;

#[derive(Clone)]
pub enum FactorNode {
    Variable {
        id: String,
        domain: Vec<f64>,          // Possible values (chemical shifts)
    },
    Factor {
        potential: FactorPotential,
        connected_vars: Vec<NodeIndex>,
    },
}

#[derive(Clone)]
pub enum FactorPotential {
    ChemicalShiftPrior {
        mean: f64,
        std: f64,
        atom_type: String,
    },
    PeakConsistency {
        peak_position: f64,
        tolerance: f64,
    },
    SequentialConnectivity {
        expected_shift_difference: f64,
        tolerance: f64,
    },
    NOEDistance {
        distance_bounds: (f64, f64),  // (lower, upper) in Angstroms
    },
}

pub struct FactorGraph {
    graph: DiGraph<FactorNode, f64>,
    var_indices: HashMap<String, NodeIndex>,
    messages: HashMap<(NodeIndex, NodeIndex), Array1<f64>>,
}

impl FactorGraph {
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            var_indices: HashMap::new(),
            messages: HashMap::new(),
        }
    }

    pub fn add_chemical_shift_variable(
        &mut self,
        atom_id: &str,
        possible_shifts: Vec<f64>,
    ) -> NodeIndex {
        let node = FactorNode::Variable {
            id: atom_id.to_string(),
            domain: possible_shifts,
        };
        let idx = self.graph.add_node(node);
        self.var_indices.insert(atom_id.to_string(), idx);
        idx
    }

    pub fn add_bmrb_prior_factor(
        &mut self,
        atom_id: &str,
        atom_type: &str,
        mean: f64,
        std: f64,
    ) {
        let var_idx = self.var_indices[atom_id];
        let factor = FactorNode::Factor {
            potential: FactorPotential::ChemicalShiftPrior {
                mean,
                std,
                atom_type: atom_type.to_string(),
            },
            connected_vars: vec![var_idx],
        };
        let factor_idx = self.graph.add_node(factor);
        self.graph.add_edge(var_idx, factor_idx, 1.0);
        self.graph.add_edge(factor_idx, var_idx, 1.0);
    }

    pub fn add_peak_factor(
        &mut self,
        atom_ids: Vec<&str>,
        peak_positions: Vec<f64>,
        tolerances: Vec<f64>,
    ) {
        let var_indices: Vec<NodeIndex> = atom_ids
            .iter()
            .map(|id| self.var_indices[*id])
            .collect();
        
        for (i, (&pos, &tol)) in peak_positions.iter().zip(tolerances.iter()).enumerate() {
            let factor = FactorNode::Factor {
                potential: FactorPotential::PeakConsistency {
                    peak_position: pos,
                    tolerance: tol,
                },
                connected_vars: vec![var_indices[i]],
            };
            let factor_idx = self.graph.add_node(factor);
            self.graph.add_edge(var_indices[i], factor_idx, 1.0);
            self.graph.add_edge(factor_idx, var_indices[i], 1.0);
        }
    }

    pub fn run_belief_propagation(&mut self, max_iterations: usize, tolerance: f64) {
        // Initialize messages to uniform
        self.initialize_messages();
        
        for iteration in 0..max_iterations {
            let max_delta = self.propagate_one_step();
            if max_delta < tolerance {
                tracing::info!("BP converged at iteration {}", iteration);
                break;
            }
        }
    }

    fn propagate_one_step(&mut self) -> f64 {
        let mut max_delta = 0.0f64;
        
        for edge in self.graph.edge_indices() {
            let (from, to) = self.graph.edge_endpoints(edge).unwrap();
            let new_message = self.compute_message(from, to);
            
            let key = (from, to);
            if let Some(old_message) = self.messages.get(&key) {
                let delta = (&new_message - old_message)
                    .mapv(|x| x.abs())
                    .sum();
                max_delta = max_delta.max(delta);
            }
            self.messages.insert(key, new_message);
        }
        
        max_delta
    }

    fn compute_message(&self, from: NodeIndex, to: NodeIndex) -> Array1<f64> {
        match &self.graph[from] {
            FactorNode::Variable { domain, .. } => {
                // Variable to factor message: product of incoming messages except from target
                let mut msg = Array1::ones(domain.len());
                for neighbor in self.graph.neighbors(from) {
                    if neighbor != to {
                        if let Some(incoming) = self.messages.get(&(neighbor, from)) {
                            msg *= incoming;
                        }
                    }
                }
                normalize_log_message(msg)
            }
            FactorNode::Factor { potential, connected_vars } => {
                // Factor to variable message: marginalize potential
                self.compute_factor_message(potential, connected_vars, to)
            }
        }
    }

    fn compute_factor_message(
        &self,
        potential: &FactorPotential,
        connected_vars: &[NodeIndex],
        target: NodeIndex,
    ) -> Array1<f64> {
        let target_domain = match &self.graph[target] {
            FactorNode::Variable { domain, .. } => domain,
            _ => panic!("Target must be variable"),
        };
        
        let mut message = Array1::zeros(target_domain.len());
        
        match potential {
            FactorPotential::ChemicalShiftPrior { mean, std, .. } => {
                for (i, &value) in target_domain.iter().enumerate() {
                    let log_prob = -0.5 * ((value - mean) / std).powi(2);
                    message[i] = log_prob;
                }
            }
            FactorPotential::PeakConsistency { peak_position, tolerance } => {
                for (i, &value) in target_domain.iter().enumerate() {
                    let diff = (value - peak_position).abs();
                    let log_prob = if diff < *tolerance {
                        -0.5 * (diff / tolerance).powi(2)
                    } else {
                        -10.0  // Strong penalty for violations
                    };
                    message[i] = log_prob;
                }
            }
            _ => {
                // Other potentials...
            }
        }
        
        normalize_log_message(message)
    }

    pub fn get_marginals(&self) -> HashMap<String, Array1<f64>> {
        let mut marginals = HashMap::new();
        
        for (id, &idx) in &self.var_indices {
            if let FactorNode::Variable { domain, .. } = &self.graph[idx] {
                let mut marginal = Array1::zeros(domain.len());
                for neighbor in self.graph.neighbors(idx) {
                    if let Some(msg) = self.messages.get(&(neighbor, idx)) {
                        marginal += msg;
                    }
                }
                marginals.insert(id.clone(), softmax(&marginal));
            }
        }
        
        marginals
    }
}

fn normalize_log_message(mut msg: Array1<f64>) -> Array1<f64> {
    let max_val = msg.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    msg -= max_val;
    msg
}

fn softmax(log_probs: &Array1<f64>) -> Array1<f64> {
    let max_val = log_probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_vals = log_probs.mapv(|x| (x - max_val).exp());
    let sum = exp_vals.sum();
    exp_vals / sum
}
```

---

## CRYSTALLINE: Density field data structures

The CRYSTALLINE mode represents peaks as continuous probability densities that evolve through three states. These structures supplement the factor graph approach above.

### Density field representations

```rust
// src-tauri/src/data/density.rs

use serde::{Deserialize, Serialize};
use crate::data::spectrum::ExperimentType;

/// Chemical shift position in N-dimensional space
pub type ChemShift<const D: usize> = [f64; D];

/// Particle in the density field (Sequential Monte Carlo representation)
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

/// Variational Gaussian Mixture Model (Dirichlet Process)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariationalGMM<const D: usize> {
    pub components: Vec<GaussianComponent<D>>,
    pub concentration: f64,  // Dirichlet concentration parameter
    pub converged: bool,
    pub elbo: f64,  // Evidence lower bound
}

/// Density field representation (multiple options for different use cases)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DensityField<const D: usize> {
    /// Particle cloud for online updates (Sequential Monte Carlo)
    Particles(ParticleCloud<D>),
    /// Variational GMM for final extraction
    GMM(VariationalGMM<D>),
    /// Hybrid: particles for updates, GMM for extraction
    Hybrid {
        particles: ParticleCloud<D>,
        gmm: VariationalGMM<D>,
    },
}
```

### Peak states during crystallization

```rust
// Peak at various crystallization stages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PeakState<const D: usize> {
    /// Diffuse density - not yet a peak
    Diffuse {
        region_id: usize,
        center_estimate: ChemShift<D>,
        spread: f64,  // Approximate extent
    },

    /// Nucleating - gathering evidence, uncertainty narrowing
    Nucleating {
        mean: ChemShift<D>,
        covariance: [[f64; D]; D],
        persistence: f64,           // Topological persistence
        entropy: f64,               // Shannon entropy of distribution
        evidence_sources: Vec<ExperimentType>,
    },

    /// Crystallized - definite peak with quantified uncertainty
    Crystallized(CrystallinePeak<D>),
}

/// Fully crystallized peak with uncertainty quantification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrystallinePeak<const D: usize> {
    pub id: uuid::Uuid,
    pub position: ChemShift<D>,
    pub covariance: [[f64; D]; D],
    pub intensity: f64,
    pub volume: Option<f64>,
    pub line_width: [f64; D],

    // Crystallization metadata
    pub persistence: f64,              // How significant (topologically)
    pub crystallization_entropy: f64,  // Entropy at crystallization
    pub confidence: f64,               // 1.0 - exp(entropy)
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

---

## Zarr chunk sizes for NMR data types

| Data Type | Typical Shape | Recommended Chunks | Rationale |
|-----------|---------------|-------------------|-----------|
| 1D spectrum | (32768,) | (4096,) | Fits L2 cache, ~32KB |
| 2D HSQC | (2048, 512) | (256, 64) | ~65KB chunks, good for random access |
| 2D NOESY | (4096, 4096) | (512, 512) | Larger for sequential processing |
| 3D HNCO | (128, 64, 512) | (32, 16, 128) | Balance all dimensions |
| 4D HNCACO | (64, 32, 32, 256) | (16, 8, 8, 64) | ~32KB chunks |
| Relaxation series | (N_residues, N_delays) | (50, N_delays) | Full time series per residue |
| CPMG dispersion | (N_residues, N_νCPMG) | (50, N_νCPMG) | Group residues |

Compression: Use **blosc with lz4** for spectral data (fast decompression), **zstd level 3** for archival.

---

## Parquet schema for peak lists with uncertainty

```python
import pyarrow as pa

peak_schema = pa.schema([
    ('peak_id', pa.int64()),
    ('spectrum_id', pa.string()),
    ('experiment_type', pa.string()),
    
    # Positions with uncertainty
    ('positions_ppm', pa.list_(pa.float32())),
    ('position_errors', pa.list_(pa.float32())),
    
    # Intensities
    ('intensity', pa.float32()),
    ('volume', pa.float32()),
    ('signal_noise_ratio', pa.float32()),
    
    # Assignments (nested for ambiguity)
    ('assignments', pa.list_(pa.struct([
        ('dimension', pa.int8()),
        ('chain_code', pa.string()),
        ('residue_code', pa.string()),
        ('atom_name', pa.string()),
        ('probability', pa.float32())  # For ambiguous assignments
    ]))),
    
    # Cross-experiment linking
    ('linked_peak_ids', pa.list_(pa.int64())),  # Peaks sharing assignment
    ('constraint_ids', pa.list_(pa.int64())),    # Derived constraints
    
    # Metadata
    ('figure_of_merit', pa.float32()),
    ('is_artifact', pa.bool_()),
    ('annotation', pa.string())
])
```

---

## ML pipeline architecture with ONNX

```rust
// src-tauri/src/ml/inference.rs
use ort::{session::Session, value::TensorRef};
use ndarray::{Array2, ArrayViewD};
use std::sync::Arc;
use parking_lot::RwLock;
use std::collections::HashMap;

pub struct ModelRegistry {
    models: RwLock<HashMap<String, Arc<ONNXModel>>>,
    model_dir: std::path::PathBuf,
}

pub struct ONNXModel {
    session: Session,
    pub model_version: String,
    pub model_hash: String,
    pub input_shape: Vec<i64>,
    pub output_shape: Vec<i64>,
}

impl ONNXModel {
    pub fn load(model_path: &str, version: &str) -> Result<Self, crate::error::NmrError> {
        let session = Session::builder()?
            .with_optimization_level(ort::session::GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(model_path)?;
        
        let model_hash = compute_model_hash(model_path);
        
        // Extract input/output shapes from model metadata
        let input_shape = session.inputs[0]
            .input_type
            .tensor_dimensions()
            .unwrap()
            .to_vec();
        let output_shape = session.outputs[0]
            .output_type
            .tensor_dimensions()
            .unwrap()
            .to_vec();
        
        Ok(Self {
            session,
            model_version: version.to_string(),
            model_hash,
            input_shape,
            output_shape,
        })
    }

    pub fn infer(&self, input: Array2<f32>) -> Result<Array2<f32>, crate::error::NmrError> {
        let input_tensor = TensorRef::try_from(&input)?;
        let outputs = self.session.run(ort::inputs!["input" => input_tensor]?)?;
        
        let output = outputs["output"]
            .try_extract_tensor::<f32>()?
            .view()
            .to_owned();
        
        Ok(output.into_dimensionality()?)
    }
}

impl ModelRegistry {
    pub fn new(model_dir: std::path::PathBuf) -> Self {
        Self {
            models: RwLock::new(HashMap::new()),
            model_dir,
        }
    }

    pub fn get_or_load(&self, model_id: &str, version: &str) -> Result<Arc<ONNXModel>, crate::error::NmrError> {
        let key = format!("{}:{}", model_id, version);
        
        if let Some(model) = self.models.read().get(&key) {
            return Ok(Arc::clone(model));
        }
        
        let path = self.model_dir
            .join(model_id)
            .join(version)
            .join("model.onnx");
        
        let model = Arc::new(ONNXModel::load(
            path.to_str().unwrap(),
            version,
        )?);
        
        self.models.write().insert(key, Arc::clone(&model));
        Ok(model)
    }
}

fn compute_model_hash(path: &str) -> String {
    use std::fs::File;
    use std::io::Read;
    
    let mut file = File::open(path).unwrap();
    let mut hasher = blake3::Hasher::new();
    let mut buffer = [0u8; 8192];
    
    loop {
        let bytes_read = file.read(&mut buffer).unwrap();
        if bytes_read == 0 { break; }
        hasher.update(&buffer[..bytes_read]);
    }
    
    hasher.finalize().to_hex().to_string()
}
```

---

## CRYSTALLINE: Topological persistence for peak detection

Persistent homology provides parameter-free peak detection by tracking which density features persist across filtration levels. True peaks have high persistence; noise artifacts die quickly.

```rust
// src-tauri/src/topology/persistence.rs

/// Persistence diagram storing birth-death pairs
#[derive(Debug, Clone)]
pub struct PersistenceDiagram {
    pub pairs: Vec<BirthDeathPair>,
}

#[derive(Debug, Clone)]
pub struct BirthDeathPair {
    pub birth: f64,         // Filtration level at which feature appears
    pub death: f64,         // Filtration level at which feature merges
    pub location: Vec<f64>, // Chemical shift location of the feature
    pub dimension: usize,   // Homological dimension (0 for peaks/maxima)
}

impl BirthDeathPair {
    /// Persistence = death - birth. Higher = more significant feature.
    pub fn persistence(&self) -> f64 {
        self.death - self.birth
    }
}

impl PersistenceDiagram {
    /// Compute persistence from density field using superlevel set filtration
    pub fn from_density<const D: usize>(density: &KernelDensityEstimate<D>) -> Self {
        let values = density.evaluate_on_grid();
        let mut sorted_indices: Vec<usize> = (0..values.len()).collect();

        // Sort descending for superlevel sets (peaks = local maxima)
        sorted_indices.sort_by(|&a, &b| {
            values[b].partial_cmp(&values[a]).unwrap()
        });

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
                // New component born (local maximum)
                component_birth[idx] = Some(value);
            } else {
                // Merge components - older (higher value) survives
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

        PersistenceDiagram { pairs }
    }

    /// Get features above persistence threshold (significant peaks)
    pub fn significant_features(&self, threshold: f64) -> Vec<&BirthDeathPair> {
        self.pairs
            .iter()
            .filter(|p| p.persistence() > threshold)
            .collect()
    }

    /// Estimate noise threshold from persistence distribution
    pub fn estimate_noise_threshold(&self, significance: f64) -> f64 {
        let mut persistences: Vec<f64> = self.pairs
            .iter()
            .map(|p| p.persistence())
            .filter(|&p| p.is_finite() && p > 0.0)
            .collect();

        if persistences.is_empty() {
            return 0.0;
        }

        persistences.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let idx = ((1.0 - significance) * persistences.len() as f64) as usize;
        persistences[idx.min(persistences.len() - 1)]
    }
}

/// Union-Find with path compression for persistence computation
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
        if px == py { return; }
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

## CRYSTALLINE: Crystallization algorithm

The crystallization engine tracks density evolution and determines when peaks should crystallize based on multiple criteria.

```rust
// src-tauri/src/crystallize/mod.rs

use crate::data::density::{DensityField, PeakState, CrystallinePeak, ParticleCloud};
use crate::data::spectrum::ExperimentType;
use crate::topology::persistence::PersistenceDiagram;

/// Configuration for crystallization criteria
#[derive(Debug, Clone)]
pub struct CrystallizationConfig {
    pub entropy_threshold: f64,       // Default: ln(0.02) ≈ -3.91 for 0.02 ppm
    pub persistence_threshold: f64,   // Default: 3.0 (3× noise level)
    pub mdl_threshold: f64,           // Default: 2.0 bits
    pub min_evidence_sources: usize,  // Default: 2 experiments
    pub particle_count: usize,        // Default: 10000
    pub resample_threshold: f64,      // Default: 0.5 (ESS ratio)
}

impl Default for CrystallizationConfig {
    fn default() -> Self {
        Self {
            entropy_threshold: -3.91,
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
        let particles = sample_from_bmrb_prior(sequence, bmrb_priors, config.particle_count);

        Self {
            density: DensityField::Particles(particles),
            peak_states: Vec::new(),
            persistence: PersistenceDiagram::empty(),
            evidence_history: Vec::new(),
            crystallization_progress: 0.0,
        }
    }

    /// Update density with new experiment evidence
    pub fn update_with_experiment(
        &mut self,
        experiment: &Experiment<D>,
        config: &CrystallizationConfig,
    ) {
        let observations = extract_observations(experiment);

        // Update particle weights with likelihood
        if let DensityField::Particles(ref mut particles) = self.density {
            for obs in &observations {
                let likelihood = compute_likelihood(obs, experiment.experiment_type);
                particles.update_weights(&likelihood);
            }

            // Resample if effective sample size too low
            if particles.effective_sample_size() <
               config.resample_threshold * config.particle_count as f64 {
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

        // Update persistence and check crystallization
        self.update_persistence();
        self.check_crystallization(config);
    }

    /// Check if any density regions should crystallize
    fn check_crystallization(&mut self, config: &CrystallizationConfig) {
        let candidates = self.persistence.significant_features(config.persistence_threshold);

        for candidate in candidates {
            let region = self.extract_region(&candidate);

            // Check ALL crystallization criteria
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
            } else if candidate.persistence() > config.persistence_threshold * 0.5 {
                // Mark as nucleating (gathering evidence)
                self.peak_states.push(PeakState::Nucleating {
                    mean: region.center,
                    covariance: region.covariance,
                    persistence: candidate.persistence(),
                    entropy,
                    evidence_sources: region.evidence_sources,
                });
            }
        }

        self.crystallization_progress = self.compute_progress();
    }

    /// Crystallize a region into a definite peak
    fn crystallize_region(&self, region: DensityRegion<D>) -> CrystallinePeak<D> {
        CrystallinePeak {
            id: uuid::Uuid::new_v4(),
            position: region.center,
            covariance: region.covariance,
            intensity: region.total_weight,
            volume: Some(region.volume),
            line_width: region.estimate_linewidth(),
            persistence: region.persistence,
            crystallization_entropy: region.entropy,
            confidence: 1.0 - region.entropy.exp(),
            evidence_sources: region.evidence_sources,
            assignments: Vec::new(),
        }
    }
}
```

---

## Tauri command for global assignment

```rust
// src-tauri/src/commands/assignment.rs
use tauri::State;
use crate::state::AppState;
use crate::inference::factor_graph::FactorGraph;
use crate::error::Result;

#[derive(serde::Serialize, serde::Deserialize)]
pub struct AssignmentResult {
    pub atom_id: String,
    pub assigned_shift: f64,
    pub confidence: f64,
    pub alternative_shifts: Vec<(f64, f64)>,  // (shift, probability)
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct GlobalAssignmentRequest {
    pub peak_lists: Vec<PeakListData>,
    pub sequence: String,
    pub bmrb_statistics: Vec<BMRBPrior>,
    pub tolerances: ToleranceSettings,
}

#[tauri::command]
pub async fn run_global_assignment(
    request: GlobalAssignmentRequest,
    state: State<'_, AppState>,
) -> Result<Vec<AssignmentResult>> {
    // Build factor graph from all experiments
    let mut graph = FactorGraph::new();
    
    // Add chemical shift variables for each assignable atom
    let atoms = parse_sequence_atoms(&request.sequence);
    for atom in &atoms {
        let possible_shifts = get_possible_shifts(&atom.atom_type, &request.bmrb_statistics);
        graph.add_chemical_shift_variable(&atom.id, possible_shifts);
    }
    
    // Add BMRB prior factors
    for atom in &atoms {
        if let Some(prior) = find_bmrb_prior(&atom.atom_type, &request.bmrb_statistics) {
            graph.add_bmrb_prior_factor(
                &atom.id,
                &atom.atom_type,
                prior.mean,
                prior.std,
            );
        }
    }
    
    // Add peak consistency factors from ALL experiments
    for peak_list in &request.peak_lists {
        for peak in &peak_list.peaks {
            let atom_ids: Vec<&str> = peak.possible_assignments
                .iter()
                .map(|a| a.atom_id.as_str())
                .collect();
            
            graph.add_peak_factor(
                atom_ids,
                peak.positions.clone(),
                vec![request.tolerances.get(&peak_list.experiment_type); peak.positions.len()],
            );
        }
    }
    
    // Add sequential connectivity factors
    add_sequential_connectivity_factors(&mut graph, &atoms);
    
    // Add NOE distance factors
    for peak_list in request.peak_lists.iter().filter(|pl| pl.experiment_type == "NOESY") {
        add_noe_factors(&mut graph, peak_list, &request.tolerances);
    }
    
    // Run belief propagation
    graph.run_belief_propagation(100, 1e-6);
    
    // Extract results
    let marginals = graph.get_marginals();
    let results: Vec<AssignmentResult> = atoms
        .iter()
        .filter_map(|atom| {
            marginals.get(&atom.id).map(|marginal| {
                let (best_idx, &best_prob) = marginal
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())?;
                
                let possible_shifts = get_possible_shifts(&atom.atom_type, &request.bmrb_statistics);
                
                Some(AssignmentResult {
                    atom_id: atom.id.clone(),
                    assigned_shift: possible_shifts[best_idx],
                    confidence: best_prob,
                    alternative_shifts: marginal
                        .iter()
                        .enumerate()
                        .filter(|(i, p)| *i != best_idx && **p > 0.05)
                        .map(|(i, &p)| (possible_shifts[i], p))
                        .collect(),
                })
            })?
        })
        .collect();
    
    Ok(results)
}
```

---

## CRYSTALLINE: Density field Tauri commands

Commands for the CRYSTALLINE density crystallization mode.

```rust
// src-tauri/src/commands/density.rs

use tauri::State;
use crate::state::AppState;
use crate::error::Result;
use crate::crystallize::{CrystallizationState, CrystallizationConfig};

/// Initialize density field from sequence and BMRB priors
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

    // Initialize crystallization state (2D for HSQC-like data)
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

/// Get all peak states (diffuse, nucleating, crystallized)
#[tauri::command]
pub async fn get_peak_states(
    project_id: uuid::Uuid,
    state: State<'_, AppState>,
) -> Result<Vec<PeakStateSummary>> {
    let projects = state.crystallization_states.lock().await;
    let crystal_state = projects.get(&project_id)
        .ok_or(crate::error::NmrError::NotFound)?;

    Ok(crystal_state.peak_states.iter()
        .map(|ps| ps.to_summary())
        .collect())
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

#[derive(serde::Serialize)]
pub struct DensityFieldData {
    pub values: Vec<Vec<f64>>,
    pub x_range: (f64, f64),
    pub y_range: (f64, f64),
    pub peak_states: Vec<PeakStateSummary>,
}

#[derive(serde::Serialize)]
pub struct PeakStateSummary {
    pub state_type: String,  // "diffuse" | "nucleating" | "crystallized"
    pub position: Vec<f64>,
    pub covariance: Option<Vec<Vec<f64>>>,
    pub persistence: Option<f64>,
    pub entropy: Option<f64>,
    pub confidence: Option<f64>,
}
```

---

## React component for spectrum visualization

```tsx
// src/components/spectrum/SpectrumViewer.tsx
import { useRef, useEffect, useCallback, useMemo } from 'react';
import { useSpectrumStore } from '../../stores/spectrumStore';

interface SpectrumViewerProps {
  width: number;
  height: number;
  showPeaks?: boolean;
}

export function SpectrumViewer({ width, height, showPeaks = true }: SpectrumViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { spectrum, peaks, selectedPeak, setSelectedPeak } = useSpectrumStore();
  
  const renderSpectrum = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !spectrum) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);
    
    // Clear
    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(0, 0, width, height);
    
    // Calculate scales
    const { real, ppmAxis } = spectrum;
    const xScale = width / real.length;
    const yMin = Math.min(...real);
    const yMax = Math.max(...real);
    const yRange = yMax - yMin || 1;
    const yScale = (height - 40) / yRange;
    
    // Draw spectrum
    ctx.strokeStyle = '#00ff88';
    ctx.lineWidth = 1;
    ctx.beginPath();
    
    for (let i = 0; i < real.length; i++) {
      const x = i * xScale;
      const y = height - 20 - (real[i] - yMin) * yScale;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    
    // Draw peaks
    if (showPeaks && peaks) {
      for (const peak of peaks) {
        const x = (peak.index / real.length) * width;
        const y = height - 20 - (peak.intensity - yMin) * yScale;
        
        ctx.fillStyle = peak.id === selectedPeak ? '#ff4444' : '#ffaa00';
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.fill();
        
        // Peak label
        if (peak.assignment) {
          ctx.fillStyle = '#ffffff';
          ctx.font = '10px monospace';
          ctx.fillText(peak.assignment, x - 10, y - 8);
        }
      }
    }
    
    // Draw ppm axis
    ctx.strokeStyle = '#666';
    ctx.fillStyle = '#aaa';
    ctx.font = '10px sans-serif';
    ctx.beginPath();
    ctx.moveTo(0, height - 20);
    ctx.lineTo(width, height - 20);
    ctx.stroke();
    
    const ppmMin = ppmAxis[0];
    const ppmMax = ppmAxis[ppmAxis.length - 1];
    for (let ppm = Math.ceil(ppmMin); ppm <= Math.floor(ppmMax); ppm++) {
      const x = ((ppm - ppmMin) / (ppmMax - ppmMin)) * width;
      ctx.fillText(`${ppm}`, x - 5, height - 5);
    }
  }, [spectrum, peaks, selectedPeak, width, height, showPeaks]);
  
  useEffect(() => {
    renderSpectrum();
  }, [renderSpectrum]);
  
  const handleClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!peaks || !spectrum) return;
    
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;
    
    const x = e.clientX - rect.left;
    const clickIndex = Math.round((x / width) * spectrum.real.length);
    
    // Find nearest peak
    const nearest = peaks.reduce((closest, peak) => {
      const dist = Math.abs(peak.index - clickIndex);
      return dist < closest.dist ? { peak, dist } : closest;
    }, { peak: null as any, dist: Infinity });
    
    if (nearest.dist < 20) {
      setSelectedPeak(nearest.peak.id);
    }
  }, [peaks, spectrum, width, setSelectedPeak]);
  
  return (
    <canvas
      ref={canvasRef}
      style={{ width, height, cursor: 'crosshair' }}
      onClick={handleClick}
    />
  );
}
```

---

## Step-by-step implementation guidance

### Phase 1: Foundation (Weeks 1-2)
1. Initialize Tauri 2.0 project with `npm create tauri-app@latest`
2. Configure Cargo.toml with core dependencies (rusqlite, ndarray, rustfft)
3. Implement error.rs with thiserror error types
4. Set up SQLite with WAL mode and migrations
5. Create basic spectrum data structures
6. Implement FFT processing with rustfft

### Phase 2: Data layer (Weeks 3-4)
1. Implement molecular graph using petgraph
2. Create peak list data structures with uncertainty
3. Build Zarr reader for spectral data
4. Implement Parquet writer for peak lists
5. Create database queries for cross-experiment lookups

### Phase 3: Inference engine (Weeks 5-6)
1. Implement factor graph construction
2. Add belief propagation message passing
3. Create BMRB prior integration
4. Implement peak consistency factors
5. Add NOE distance factors
6. Build consensus scoring across experiments

### Phase 4: Frontend (Weeks 7-8)
1. Set up React with Zustand stores
2. Create spectrum canvas renderer
3. Implement peak picking UI
4. Build assignment table with probability display
5. Add cross-experiment linking visualization

### Phase 5: ML integration (Weeks 9-10)
1. Set up ONNX runtime with ort crate
2. Create model registry with versioning
3. Implement inference caching
4. Add peak picking model integration
5. Build assignment prediction pipeline

### Phase 6: Testing and polish (Weeks 11-12)
1. Property-based testing with proptest
2. Benchmark critical paths with criterion
3. Numerical validation with known datasets
4. Cross-platform testing (Windows, macOS, Linux)
5. Documentation and examples

### Phase 7: CRYSTALLINE density crystallization (Alternative mode)
1. Implement `data/density.rs` with PeakState, DensityField types
2. Create `density/` module:
   - `particle.rs` - Particle cloud representation
   - `gmm.rs` - Variational Gaussian Mixture Model
   - `kde.rs` - Kernel density estimation
3. Build `topology/` module:
   - `union_find.rs` - Union-find with persistence tracking
   - `persistence.rs` - Persistent homology computation
4. Implement `crystallize/` module:
   - `criteria.rs` - Crystallization thresholds
   - `entropy.rs` - Entropy computation
   - Main crystallization state machine
5. Add density Tauri commands (initialize_density, add_experiment_evidence, get_density_field, get_peak_states)
6. Create frontend density visualization:
   - `densityStore.ts` - Zustand state management
   - `DensityView.tsx` - Density field heatmap
   - `CrystallizationProgress.tsx` - Progress indicator
   - `UncertaintyEllipse.tsx` - Confidence region display

**Performance targets for CRYSTALLINE:**
| Operation | Target |
|-----------|--------|
| Persistence computation (10K particles) | < 100 ms |
| Particle cloud update | < 10 ms |
| Density field evaluation (256×256) | < 50 ms |
| Crystallization check | < 5 ms |

---

## Conclusion: The simultaneous analysis advantage

The revolutionary aspect of this platform lies not in any single algorithm, but in the unified factor graph that connects all experimental observations. When a human analyst examines an HSQC spectrum, they cannot simultaneously consider the implications for NOESY assignments, relaxation dynamics, and structural constraints. The software can.

By encoding every peak as a node in a probabilistic graphical model, constraint propagation automatically resolves ambiguities that would require human intuition. A weak HSQC assignment becomes strong when corroborated by NOESY cross-peaks and consistent with TOCSY spin systems. Contradictory constraints from different experiments are handled through soft potentials rather than hard failures.

The implementation strategy prioritizes incremental development—each phase delivers working functionality while building toward the complete simultaneous analysis system. The modular architecture ensures that improvements to any component (better peak picking ML, more sophisticated factor potentials, faster inference) immediately benefit the entire platform.

### Two Analysis Modes

**Traditional Factor Graph Mode:** Binary peak picking followed by global assignment via belief propagation. Fast, well-understood, suitable for clean spectra with good separation.

**CRYSTALLINE Mode:** Density crystallization with continuous probability fields, topological persistence, and information-theoretic crystallization criteria. Handles crowded regions gracefully, maintains honest uncertainty quantification, but computationally more intensive.

Users can choose the appropriate mode based on their data quality and requirements. Both modes share the underlying data structures and can interoperate—peaks crystallized from density can feed into the factor graph for final assignment refinement.

This dual-approach represents the natural evolution of NMR software from sequential human-mimicking workflows to truly parallel machine reasoning—finding patterns in data that no human analyst could perceive.