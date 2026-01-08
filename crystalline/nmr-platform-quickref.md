# CRYSTALLINE: Quick Reference & Key Concepts

## The Big Idea

**Traditional NMR**: Sequential binary decisions
```
Spectrum → Peak or No Peak? → Assign → Structure
          (threshold-based)  (manual)  (restraints)
```

**CRYSTALLINE**: Continuous probabilistic reasoning
```
All Data → Density Field → Crystallization → Structure
           (probability)   (when certain)    (ensemble)
```

---

## Core Innovation: Density Crystallization

### What It Solves
- **Crowded regions**: Traditional peak picking forces wrong decisions in overlapped areas
- **Lost uncertainty**: Binary peak/no-peak loses information about confidence
- **Sequential bottleneck**: Human analysts can only look at one spectrum at a time

### How It Works

1. **Initial State**: Instead of peaks, we have a *probability density* over chemical shift space
   - Initialized from BMRB statistics for each atom type
   - Represents "somewhere here there might be a signal"

2. **Evidence Accumulation**: Each experiment updates the density
   - HSQC narrows down H/N positions
   - NOESY adds distance constraints  
   - TOCSY groups spin systems
   - All contribute simultaneously via factor graph

3. **Crystallization Criteria**: Density becomes a peak when:
   - **Entropy < threshold**: Position is well-determined (< 0.02 ppm uncertainty)
   - **Persistence > noise**: Topologically significant (not noise artifact)
   - **MDL favors peak**: Model selection says "yes, there's really a peak here"

4. **Graceful Handling of Ambiguity**: Crowded regions stay as density until resolvable
   - No forced wrong decisions
   - Uncertainty propagates to structure calculation

---

## Mathematical Framework Summary

### Density Representation
```
Option A: Particle Cloud (Sequential Monte Carlo)
- 10,000+ weighted particles in chemical shift space
- Updated by importance resampling
- Good for online updates

Option B: Variational GMM (Dirichlet Process)  
- Automatic number of components
- Uncertainty quantification built-in
- Better for final extraction
```

### Crystallization Criterion
A density region crystallizes when ALL conditions met:

```
1. H(X | data) < -3.91        # Entropy below threshold (≈0.02 ppm)
2. persistence > 3σ_noise     # Topologically significant  
3. MDL(peak) < MDL(no peak)   # Model selection favors peak
4. evidence_sources ≥ 2       # Multiple experiments agree
```

### Factor Graph for Multi-Experiment Fusion
```
Variables: Chemical shifts, peak existence, assignments
Factors:   BMRB priors, peak consistency, NOE distances, spin systems

Inference: Loopy belief propagation
Output:    Marginal distributions over all assignments
```

---

## Technology Stack

```
Frontend:  React + TypeScript + WebGL
Bridge:    Tauri 2.0 IPC
Backend:   Rust
  - rustfft (FFT processing)
  - ndarray (array operations)
  - petgraph (molecular/factor graphs)
  - rusqlite (database)
  - ort (ONNX ML inference)
  - rayon (parallelism)

Data:
  - Zarr (spectral data storage)
  - Parquet (peak lists)
  - SQLite (metadata, provenance)
```

---

## Key Data Structures

### Peak States
```rust
enum PeakState {
    Diffuse {           // Just noise-like density
        region_id,
        center_estimate,
        spread,
    },
    Nucleating {        // Gathering evidence
        mean,
        covariance,
        persistence,
        entropy,
        evidence_sources,
    },
    Crystallized {      // Definite peak
        position,
        covariance,     // Uncertainty!
        confidence,
        assignments,
    },
}
```

### Experiment Evidence
```rust
struct EvidenceRecord {
    experiment_type,     // HSQC, NOESY, TOCSY, etc.
    timestamp,
    num_observations,
    log_likelihood,
}
```

---

## Algorithm Pipeline

```
1. INITIALIZE
   └─> Sample particles from BMRB priors for sequence

2. FOR each experiment (HSQC, NOESY, TOCSY, ...):
   ├─> Extract observations
   ├─> Update particle weights (Bayesian update)
   ├─> Resample if effective sample size low
   ├─> Compute persistence diagram
   └─> Check crystallization criteria
       └─> If met: extract peak, remove particles

3. FINAL PASS
   ├─> Fit variational GMM to remaining particles
   └─> Crystallize any remaining significant components

4. GLOBAL ASSIGNMENT
   ├─> Build factor graph with all peaks + constraints
   ├─> Run belief propagation
   └─> Extract assignments with probabilities

5. STRUCTURE CALCULATION
   └─> Generate ensemble with uncertainty from posteriors
```

---

## File Organization

```
crystalline/
├── src-tauri/src/
│   ├── data/           # Core types (spectrum, molecule, peak)
│   ├── io/             # File format readers (Bruker, NMR-STAR)
│   ├── processing/     # FFT, phasing, baseline
│   ├── density/        # Particle cloud, GMM, KDE
│   ├── topology/       # Persistent homology
│   ├── inference/      # Factor graph, belief propagation
│   ├── crystallize/    # Crystallization criteria & extraction
│   ├── structure/      # Restraint generation, geometry
│   └── commands/       # Tauri API endpoints
│
├── src/                # React frontend
│   ├── components/
│   │   ├── spectrum/   # Spectrum viewer
│   │   ├── density/    # Density visualization
│   │   └── assignment/ # Assignment table
│   └── stores/         # Zustand state management
│
└── models/             # ONNX ML models
```

---

## Implementation Phases

| Phase | Weeks | Focus |
|-------|-------|-------|
| 1. Foundation | 1-3 | Bruker I/O, FFT, database |
| 2. Crystallization | 4-6 | Density, persistence, criteria |
| 3. Inference | 7-9 | Factor graph, belief propagation |
| 4. Frontend | 10-12 | Visualization, UI |
| 5. Polish | 13-14 | Testing, docs, release |

---

## Performance Targets

| Operation | Target |
|-----------|--------|
| 1D FFT (16K) | < 1 ms |
| 2D FFT (2K×2K) | < 50 ms |
| Persistence | < 100 ms |
| Belief propagation | < 1 s |
| Spectrum render | 60 FPS |
| Memory | < 500 MB |

---

## Extension Points

### Future Molecule Types
- Natural products (complex coupling networks)
- Oligonucleotides (different shift ranges)
- Small molecules (metabolomics)

### Future: Dynamics
- T1/T2/NOE relaxation as additional evidence
- CPMG dispersion
- Model-free analysis integrated

### Future: AlphaFold
- Structure predictions as chemical shift priors
- Validation of assignments against predicted contacts

---

## Key Insight

> "The software sees all experiments at once and finds patterns humans cannot perceive. A weak HSQC assignment becomes strong when corroborated by NOESY cross-peaks, consistent with TOCSY spin systems, and supported by BMRB statistics. The density crystallization framework handles this naturally through probabilistic inference."

---

## Getting Started with Claude Code

```bash
# Start with:
"Initialize the CRYSTALLINE NMR platform project following 
the specification in nmr-platform-spec.md. Begin with Phase 1:
1. Create Tauri project structure
2. Implement Bruker format reader
3. Set up SQLite database with migrations"
```
