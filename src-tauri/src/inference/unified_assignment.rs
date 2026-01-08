//! Unified factor graph for simultaneous spin system building, typing, and assignment.
//!
//! **Key Vision**: ALL evidence in ONE factor graph, solved SIMULTANEOUSLY.
//!
//! Instead of:
//!   1. Build spin systems (TOCSY only) → 2. Type (carbons) → 3. Assign (sequence)
//!
//! We do:
//!   All peaks → unified factor graph → BP → assignments
//!
//! Variables:
//!   - Each peak (15N-HSQC or 13C-HSQC) gets assigned to a residue position
//!   - Domain: 0 (unassigned), 1..N (residue positions)
//!
//! Factors (all contribute SIMULTANEOUSLY):
//!   1. TOCSY correlation: peaks with TOCSY correlation → same residue
//!   2. Carbon typing: peak's carbon shift → compatible residue types (BMRB)
//!   3. Sequential NOESY: connected peaks → sequential residues
//!   4. Backbone uniqueness: each residue has at most one backbone NH
//!
//! The correct solution emerges because ALL factors must agree.

use std::collections::{HashMap, HashSet};
use ndarray::Array1;
use uuid::Uuid;

use crate::data::{PeakExperimentType, UnlabeledPeak, TransferPathway, ResidueOffset, AtomConstraint};
use crate::data::spectrum::NucleusType;
use crate::inference::scoring::{KDEScorer, ShiftScorer};
use crate::testdata::{BMRBDatabase, KDEDatabase};

// =============================================================================
// NEW: Unified Observation Model - Every Peak is Equal
// =============================================================================

/// A single observation from ANY experiment type.
/// This is the unified representation where HSQC, TOCSY, NOESY, etc. are all first-class.
///
/// Now includes physics-based metadata for factor reasoning:
/// - `transfer_pathway`: How magnetization moved (DirectBond, ThroughBond, ThroughSpace, BackboneSequential)
/// - `is_sequential_evidence`: Whether this observation provides inter-residue connectivity
#[derive(Debug, Clone)]
pub struct Observation {
    pub id: Uuid,
    /// What was observed (any combination of dimensions with physics metadata)
    pub dimensions: Vec<ObservedDimension>,
    /// Which experiment type produced this observation (METADATA ONLY - not for factor logic)
    pub experiment_type: PeakExperimentType,
    /// Original intensity for weighting
    pub intensity: f64,
    /// Ground truth for testing (optional)
    pub ground_truth: Option<GroundTruth>,

    // === PHYSICS-BASED FIELDS ===

    /// How magnetization was transferred to produce this observation.
    /// Factor logic should use this instead of experiment_type.
    pub transfer_pathway: TransferPathway,
    /// Whether this observation provides sequential connectivity evidence.
    /// True for inter-residue observations (e.g., weak HNCA, CBCACONH).
    pub is_sequential_evidence: bool,
}

/// A single dimension of an observation with its nucleus type, chemical shift, and physics metadata.
#[derive(Debug, Clone)]
pub struct ObservedDimension {
    /// Nucleus type (H1, C13, N15, etc.)
    pub nucleus: NucleusType,
    /// Chemical shift in ppm
    pub shift: f64,
    /// Atom name hint (e.g., "HN", "HA", "CA") - used for typing
    /// DEPRECATED: Prefer atom_constraint for new code
    pub atom_hint: Option<String>,

    // === PHYSICS-BASED FIELDS ===

    /// Residue relationship: is this dimension observing i, i-1, or unknown?
    pub residue_offset: ResidueOffset,
    /// Which atoms are physically possible for this dimension?
    pub atom_constraint: AtomConstraint,
}

/// Ground truth for testing purposes.
#[derive(Debug, Clone)]
pub struct GroundTruth {
    /// 1-indexed residue position
    pub residue_position: usize,
    /// Atom names for each dimension (e.g., ["HN", "N"] for 15N-HSQC)
    pub atom_names: Vec<String>,
}

/// Tolerance schedule for annealing during belief propagation.
#[derive(Debug, Clone, Copy)]
pub enum ToleranceSchedule {
    /// Fixed tolerance throughout BP
    Fixed,
    /// Linear interpolation from start_mult to end_mult
    Linear { start_mult: f64, end_mult: f64 },
    /// Exponential decay: base * exp(-decay * progress)
    Exponential { decay: f64 },
}

impl ToleranceSchedule {
    /// Get the multiplier at a given progress (0.0 to 1.0)
    pub fn multiplier(&self, iteration: usize, max_iterations: usize) -> f64 {
        let progress = iteration as f64 / max_iterations.max(1) as f64;
        match self {
            ToleranceSchedule::Fixed => 1.0,
            ToleranceSchedule::Linear { start_mult, end_mult } => {
                start_mult + (end_mult - start_mult) * progress
            }
            ToleranceSchedule::Exponential { decay } => {
                (-decay * progress).exp()
            }
        }
    }
}

/// Per-nucleus adaptive tolerance parameters with annealing schedule.
/// Named NucleusToleranceParams to distinguish from the simpler AdaptiveToleranceParams.
#[derive(Debug, Clone)]
pub struct NucleusToleranceParams {
    /// Base tolerance for H1 (ppm)
    pub h_tolerance_base: f64,
    /// Base tolerance for C13 (ppm)
    pub c_tolerance_base: f64,
    /// Base tolerance for N15 (ppm)
    pub n_tolerance_base: f64,
    /// Annealing schedule for tolerances
    pub tolerance_schedule: ToleranceSchedule,
}

impl Default for NucleusToleranceParams {
    fn default() -> Self {
        Self {
            h_tolerance_base: 0.03,   // 0.03 ppm for protons (~15 Hz at 500 MHz)
            c_tolerance_base: 0.4,    // 0.4 ppm for carbons
            n_tolerance_base: 0.4,    // 0.4 ppm for nitrogen
            tolerance_schedule: ToleranceSchedule::Linear {
                start_mult: 2.0,  // Start at 2x base
                end_mult: 1.0,    // End at 1x base
            },
        }
    }
}

impl NucleusToleranceParams {
    /// Get tolerance for a specific nucleus at a given BP iteration.
    pub fn tolerance_for(&self, nucleus: NucleusType, iteration: usize, max_iterations: usize) -> f64 {
        let base = match nucleus {
            NucleusType::H1 => self.h_tolerance_base,
            NucleusType::C13 => self.c_tolerance_base,
            NucleusType::N15 => self.n_tolerance_base,
            _ => self.h_tolerance_base,  // Default to proton tolerance
        };

        let mult = self.tolerance_schedule.multiplier(iteration, max_iterations);
        base * mult
    }
}

impl Observation {
    /// Create a new observation from raw peak data, populating physics metadata.
    ///
    /// Physics fields (transfer_pathway, residue_offset, atom_constraint, is_sequential_evidence)
    /// are populated based on experiment type AT CREATION TIME. After this, factor logic should
    /// use these physics fields instead of experiment_type.
    pub fn from_unlabeled_peak(peak: &UnlabeledPeak) -> Option<Self> {
        use PeakExperimentType::*;

        let (dimensions, transfer_pathway, is_sequential_evidence) = match peak.experiment_type {
            // === DIRECT BOND: HSQC experiments ===
            // One-bond J-coupling. Both nuclei are at the same residue (intra).
            Hsqc15N => {
                let dims = vec![
                    ObservedDimension {
                        nucleus: NucleusType::N15,
                        shift: peak.position_ppm[0],
                        atom_hint: Some("N".to_string()),
                        residue_offset: ResidueOffset::Intra,
                        atom_constraint: AtomConstraint::Exact("N".to_string()),
                    },
                    ObservedDimension {
                        nucleus: NucleusType::H1,
                        shift: peak.position_ppm[1],
                        atom_hint: Some("HN".to_string()),
                        residue_offset: ResidueOffset::Intra,
                        atom_constraint: AtomConstraint::Exact("H".to_string()),
                    },
                ];
                (dims, TransferPathway::DirectBond, false)
            }

            Hsqc13C => {
                let dims = vec![
                    ObservedDimension {
                        nucleus: NucleusType::C13,
                        shift: peak.position_ppm[0],
                        atom_hint: None,
                        residue_offset: ResidueOffset::Intra,
                        // Could be any aliphatic carbon
                        atom_constraint: AtomConstraint::OneOf(vec![
                            "CA".into(), "CB".into(), "CG".into(), "CG1".into(), "CG2".into(),
                            "CD".into(), "CD1".into(), "CD2".into(), "CE".into(), "CE1".into(), "CE2".into(),
                        ]),
                    },
                    ObservedDimension {
                        nucleus: NucleusType::H1,
                        shift: peak.position_ppm[1],
                        atom_hint: None,
                        residue_offset: ResidueOffset::Intra,
                        atom_constraint: AtomConstraint::Any,
                    },
                ];
                (dims, TransferPathway::DirectBond, false)
            }

            // === THROUGH-BOND: TOCSY experiments ===
            // Multi-bond J-coupling. Both protons are in the same spin system (same residue).
            Tocsy => {
                let dims = vec![
                    ObservedDimension {
                        nucleus: NucleusType::H1,
                        shift: peak.position_ppm[0],
                        atom_hint: None,
                        residue_offset: ResidueOffset::Intra,
                        atom_constraint: AtomConstraint::Any,
                    },
                    ObservedDimension {
                        nucleus: NucleusType::H1,
                        shift: peak.position_ppm[1],
                        atom_hint: None,
                        residue_offset: ResidueOffset::Intra,
                        atom_constraint: AtomConstraint::Any,
                    },
                ];
                (dims, TransferPathway::ThroughBond, false)
            }

            // === THROUGH-SPACE: NOESY experiments ===
            // Dipolar coupling. Protons may be from different residues.
            Noesy => {
                let dims = vec![
                    ObservedDimension {
                        nucleus: NucleusType::H1,
                        shift: peak.position_ppm[0],
                        atom_hint: None,
                        residue_offset: ResidueOffset::Unknown,
                        atom_constraint: AtomConstraint::Any,
                    },
                    ObservedDimension {
                        nucleus: NucleusType::H1,
                        shift: peak.position_ppm[1],
                        atom_hint: None,
                        residue_offset: ResidueOffset::Unknown,
                        atom_constraint: AtomConstraint::Any,
                    },
                ];
                (dims, TransferPathway::ThroughSpace, false)
            }

            // === HSQC-TOCSY: Hybrid experiments ===
            // DirectBond anchor + ThroughBond extension
            HsqcTocsy15N => {
                let dims = vec![
                    ObservedDimension {
                        nucleus: NucleusType::N15,
                        shift: peak.position_ppm[0],
                        atom_hint: Some("N".to_string()),
                        residue_offset: ResidueOffset::Intra,
                        atom_constraint: AtomConstraint::Exact("N".to_string()),
                    },
                    ObservedDimension {
                        nucleus: NucleusType::H1,
                        shift: peak.position_ppm[1],
                        atom_hint: None,
                        residue_offset: ResidueOffset::Intra,
                        atom_constraint: AtomConstraint::Any,
                    },
                ];
                (dims, TransferPathway::ThroughBond, false)
            }

            HsqcTocsy13C => {
                let dims = vec![
                    ObservedDimension {
                        nucleus: NucleusType::C13,
                        shift: peak.position_ppm[0],
                        atom_hint: None,
                        residue_offset: ResidueOffset::Intra,
                        atom_constraint: AtomConstraint::Any,
                    },
                    ObservedDimension {
                        nucleus: NucleusType::H1,
                        shift: peak.position_ppm[1],
                        atom_hint: None,
                        residue_offset: ResidueOffset::Intra,
                        atom_constraint: AtomConstraint::Any,
                    },
                ];
                (dims, TransferPathway::ThroughBond, false)
            }

            // === BACKBONE SEQUENTIAL: Triple-resonance experiments ===
            // Specific backbone magnetization transfer paths.
            // Intra vs inter determined by intensity.

            Hnca => {
                if peak.position_ppm.len() < 3 { return None; }
                // Strong intensity = intra (i), weak = inter (i-1)
                let is_intra = peak.intensity > 0.5;
                let carbon_offset = if is_intra { ResidueOffset::Intra } else { ResidueOffset::PrecedingResidue };
                let is_seq = !is_intra;

                let dims = vec![
                    ObservedDimension {
                        nucleus: NucleusType::H1,
                        shift: peak.position_ppm[0],
                        atom_hint: Some("HN".to_string()),
                        residue_offset: ResidueOffset::Intra,
                        atom_constraint: AtomConstraint::Exact("H".to_string()),
                    },
                    ObservedDimension {
                        nucleus: NucleusType::N15,
                        shift: peak.position_ppm[1],
                        atom_hint: Some("N".to_string()),
                        residue_offset: ResidueOffset::Intra,
                        atom_constraint: AtomConstraint::Exact("N".to_string()),
                    },
                    ObservedDimension {
                        nucleus: NucleusType::C13,
                        shift: peak.position_ppm[2],
                        atom_hint: Some("CA".to_string()),
                        residue_offset: carbon_offset,
                        atom_constraint: AtomConstraint::Exact("CA".to_string()),
                    },
                ];
                (dims, TransferPathway::BackboneSequential, is_seq)
            }

            Hncacb => {
                if peak.position_ppm.len() < 3 { return None; }
                // Positive intensity = intra (i), negative = inter (i-1)
                let is_intra = peak.intensity > 0.0;
                let carbon_offset = if is_intra { ResidueOffset::Intra } else { ResidueOffset::PrecedingResidue };
                let is_seq = !is_intra;

                let dims = vec![
                    ObservedDimension {
                        nucleus: NucleusType::H1,
                        shift: peak.position_ppm[0],
                        atom_hint: Some("HN".to_string()),
                        residue_offset: ResidueOffset::Intra,
                        atom_constraint: AtomConstraint::Exact("H".to_string()),
                    },
                    ObservedDimension {
                        nucleus: NucleusType::N15,
                        shift: peak.position_ppm[1],
                        atom_hint: Some("N".to_string()),
                        residue_offset: ResidueOffset::Intra,
                        atom_constraint: AtomConstraint::Exact("N".to_string()),
                    },
                    ObservedDimension {
                        nucleus: NucleusType::C13,
                        shift: peak.position_ppm[2],
                        atom_hint: None,
                        residue_offset: carbon_offset,
                        atom_constraint: AtomConstraint::OneOf(vec!["CA".into(), "CB".into()]),
                    },
                ];
                (dims, TransferPathway::BackboneSequential, is_seq)
            }

            Cbcaconh => {
                if peak.position_ppm.len() < 3 { return None; }
                // CBCACONH ALWAYS shows i-1 carbons (inter-residue only)
                let dims = vec![
                    ObservedDimension {
                        nucleus: NucleusType::H1,
                        shift: peak.position_ppm[0],
                        atom_hint: Some("HN".to_string()),
                        residue_offset: ResidueOffset::Intra,
                        atom_constraint: AtomConstraint::Exact("H".to_string()),
                    },
                    ObservedDimension {
                        nucleus: NucleusType::N15,
                        shift: peak.position_ppm[1],
                        atom_hint: Some("N".to_string()),
                        residue_offset: ResidueOffset::Intra,
                        atom_constraint: AtomConstraint::Exact("N".to_string()),
                    },
                    ObservedDimension {
                        nucleus: NucleusType::C13,
                        shift: peak.position_ppm[2],
                        atom_hint: None,
                        residue_offset: ResidueOffset::PrecedingResidue,
                        atom_constraint: AtomConstraint::OneOf(vec!["CA".into(), "CB".into()]),
                    },
                ];
                (dims, TransferPathway::BackboneSequential, true)
            }

            Hnco => {
                if peak.position_ppm.len() < 3 { return None; }
                // HNCO shows CO(i-1)
                let dims = vec![
                    ObservedDimension {
                        nucleus: NucleusType::H1,
                        shift: peak.position_ppm[0],
                        atom_hint: Some("HN".to_string()),
                        residue_offset: ResidueOffset::Intra,
                        atom_constraint: AtomConstraint::Exact("H".to_string()),
                    },
                    ObservedDimension {
                        nucleus: NucleusType::N15,
                        shift: peak.position_ppm[1],
                        atom_hint: Some("N".to_string()),
                        residue_offset: ResidueOffset::Intra,
                        atom_constraint: AtomConstraint::Exact("N".to_string()),
                    },
                    ObservedDimension {
                        nucleus: NucleusType::C13,
                        shift: peak.position_ppm[2],
                        atom_hint: Some("C".to_string()),
                        residue_offset: ResidueOffset::PrecedingResidue,
                        atom_constraint: AtomConstraint::Exact("C".to_string()),
                    },
                ];
                (dims, TransferPathway::BackboneSequential, true)
            }

            Hbhaconh => {
                if peak.position_ppm.len() < 3 { return None; }
                // HBHACONH shows HA/HB(i-1)
                let dims = vec![
                    ObservedDimension {
                        nucleus: NucleusType::H1,
                        shift: peak.position_ppm[0],
                        atom_hint: Some("HN".to_string()),
                        residue_offset: ResidueOffset::Intra,
                        atom_constraint: AtomConstraint::Exact("H".to_string()),
                    },
                    ObservedDimension {
                        nucleus: NucleusType::N15,
                        shift: peak.position_ppm[1],
                        atom_hint: Some("N".to_string()),
                        residue_offset: ResidueOffset::Intra,
                        atom_constraint: AtomConstraint::Exact("N".to_string()),
                    },
                    ObservedDimension {
                        nucleus: NucleusType::H1,
                        shift: peak.position_ppm[2],
                        atom_hint: None,
                        residue_offset: ResidueOffset::PrecedingResidue,
                        atom_constraint: AtomConstraint::OneOf(vec!["HA".into(), "HA2".into(), "HA3".into(), "HB".into(), "HB2".into(), "HB3".into()]),
                    },
                ];
                (dims, TransferPathway::BackboneSequential, true)
            }

            // 3D HSQC-TOCSY variants
            HsqcTocsy15N3D => {
                if peak.position_ppm.len() < 3 { return None; }
                let dims = vec![
                    ObservedDimension {
                        nucleus: NucleusType::N15,
                        shift: peak.position_ppm[0],
                        atom_hint: Some("N".to_string()),
                        residue_offset: ResidueOffset::Intra,
                        atom_constraint: AtomConstraint::Exact("N".to_string()),
                    },
                    ObservedDimension {
                        nucleus: NucleusType::H1,
                        shift: peak.position_ppm[1],
                        atom_hint: Some("HN".to_string()),
                        residue_offset: ResidueOffset::Intra,
                        atom_constraint: AtomConstraint::Exact("H".to_string()),
                    },
                    ObservedDimension {
                        nucleus: NucleusType::H1,
                        shift: peak.position_ppm[2],
                        atom_hint: None,
                        residue_offset: ResidueOffset::Intra,
                        atom_constraint: AtomConstraint::Any,
                    },
                ];
                (dims, TransferPathway::ThroughBond, false)
            }

            HsqcTocsy13C3D => {
                if peak.position_ppm.len() < 3 { return None; }
                let dims = vec![
                    ObservedDimension {
                        nucleus: NucleusType::C13,
                        shift: peak.position_ppm[0],
                        atom_hint: None,
                        residue_offset: ResidueOffset::Intra,
                        atom_constraint: AtomConstraint::Any,
                    },
                    ObservedDimension {
                        nucleus: NucleusType::H1,
                        shift: peak.position_ppm[1],
                        atom_hint: None,
                        residue_offset: ResidueOffset::Intra,
                        atom_constraint: AtomConstraint::Any,
                    },
                    ObservedDimension {
                        nucleus: NucleusType::H1,
                        shift: peak.position_ppm[2],
                        atom_hint: None,
                        residue_offset: ResidueOffset::Intra,
                        atom_constraint: AtomConstraint::Any,
                    },
                ];
                (dims, TransferPathway::ThroughBond, false)
            }
        };

        Some(Self {
            id: peak.id,
            dimensions,
            experiment_type: peak.experiment_type,
            intensity: peak.intensity,
            ground_truth: None,
            transfer_pathway,
            is_sequential_evidence,
        })
    }

    /// Check if this observation has a proton dimension.
    pub fn has_proton(&self) -> bool {
        self.dimensions.iter().any(|d| d.nucleus == NucleusType::H1)
    }

    /// Get all proton shifts in this observation.
    pub fn proton_shifts(&self) -> Vec<f64> {
        self.dimensions
            .iter()
            .filter(|d| d.nucleus == NucleusType::H1)
            .map(|d| d.shift)
            .collect()
    }

    /// Get the heavy atom shift (C13 or N15) if present.
    pub fn heavy_shift(&self) -> Option<f64> {
        self.dimensions
            .iter()
            .find(|d| d.nucleus == NucleusType::C13 || d.nucleus == NucleusType::N15)
            .map(|d| d.shift)
    }

    /// Get the heavy atom nucleus type if present.
    pub fn heavy_nucleus(&self) -> Option<NucleusType> {
        self.dimensions
            .iter()
            .find(|d| d.nucleus == NucleusType::C13 || d.nucleus == NucleusType::N15)
            .map(|d| d.nucleus)
    }
}

// =============================================================================
// END: Unified Observation Model
// =============================================================================

/// Configuration for the unified assignment algorithm.
#[derive(Debug, Clone)]
pub struct UnifiedAssignmentParams {
    /// Initial proton tolerance (loose) for TOCSY matching (ppm)
    pub h_tolerance_initial: f64,
    /// Final proton tolerance (tight) for TOCSY matching (ppm)
    pub h_tolerance_final: f64,
    /// Initial carbon tolerance (loose) for typing (ppm)
    pub c_tolerance_initial: f64,
    /// Final carbon tolerance (tight) for typing (ppm)
    pub c_tolerance_final: f64,
    /// Initial TOCSY weight (exploration phase)
    pub tocsy_weight_initial: f64,
    /// Final TOCSY weight (refinement phase)
    pub tocsy_weight_final: f64,
    /// Initial typing weight (exploration phase)
    pub typing_weight_initial: f64,
    /// Final typing weight (refinement phase)
    pub typing_weight_final: f64,
    /// Strength of sequential NOESY factor
    pub sequential_weight: f64,
    /// Strength of sequence-type constraint (type → valid positions)
    pub sequence_type_weight: f64,
    /// Confidence threshold for applying sequence-type constraint
    pub sequence_type_confidence_threshold: f64,
    /// BP convergence threshold
    pub convergence_threshold: f64,
    /// Maximum BP iterations
    pub max_iterations: usize,
    /// Damping factor for BP (0-1)
    pub damping: f64,
    /// Annealing schedule: fraction of iterations to use for exploration
    pub exploration_fraction: f64,
    /// Enable verbose output to see what the algorithm is doing
    pub verbose: bool,
    // === 3D Triple-resonance adaptive tolerances ===
    /// Initial H/N tolerance for linking 3D peaks to backbone (loose, ppm)
    pub triple_res_hn_tolerance_initial: f64,
    /// Final H/N tolerance for linking 3D peaks to backbone (tight, ppm)
    pub triple_res_hn_tolerance_final: f64,
    /// Initial carbon tolerance for CA/CB sequential matching (loose, ppm)
    pub triple_res_c_tolerance_initial: f64,
    /// Final carbon tolerance for CA/CB sequential matching (tight, ppm)
    pub triple_res_c_tolerance_final: f64,
    /// Weight for 3D sequential factor
    pub triple_res_sequential_weight: f64,
}

impl Default for UnifiedAssignmentParams {
    fn default() -> Self {
        // Optimal parameters from grid search (A7B3C1 configuration)
        // Key insight: "TOCSY is King" - trust grouping over typing
        Self {
            // Tight tolerances reduce false positive matches
            h_tolerance_initial: 0.03,  // 15 Hz at 500 MHz
            h_tolerance_final: 0.01,    // 5 Hz - very tight
            c_tolerance_initial: 2.0,   // Tighter carbon matching
            c_tolerance_final: 0.5,     // Very tight carbon matching
            // TOCSY dominant: grouping evidence is unambiguous
            // Carbon typing has overlap between amino acids (VAL/THR, etc.)
            tocsy_weight_initial: 0.5,  // Reduced: TOCSY has spurious correlations
            tocsy_weight_final: 0.5,    // Keep low to avoid pulling backbones wrong
            typing_weight_initial: 1.0, // Typing as tie-breaker only
            typing_weight_final: 2.0,   // Slight increase but still secondary
            sequential_weight: 5.0,     // Strong sequential links from triple-resonance
            sequence_type_weight: 8.0,  // Moderate: typed X → must be at X position
            sequence_type_confidence_threshold: 0.5,  // Apply when confidence > 50%
            convergence_threshold: 1e-6,
            max_iterations: 100,
            damping: 0.5,               // Standard damping
            exploration_fraction: 0.3,  // 30% exploration, 70% refinement
            verbose: false,             // Set to true for detailed output
            // 3D triple-resonance: adaptive tolerances
            // Start loose to capture all potential matches, tighten to keep best
            triple_res_hn_tolerance_initial: 0.05,  // Loose H tolerance for 3D→backbone linking
            triple_res_hn_tolerance_final: 0.01,    // Tight H tolerance
            triple_res_c_tolerance_initial: 1.0,    // Loose CA/CB matching (1 ppm)
            triple_res_c_tolerance_final: 0.2,      // Tight CA/CB matching (0.2 ppm)
            triple_res_sequential_weight: 3.0,      // Strong weight for sequential links
        }
    }
}

/// Interpolated parameters at a given iteration.
#[derive(Debug, Clone)]
pub struct InterpolatedParams {
    pub h_tolerance: f64,
    pub c_tolerance: f64,
    pub tocsy_weight: f64,
    pub typing_weight: f64,
    pub sequential_weight: f64,
    pub sequence_type_weight: f64,
    pub sequence_type_threshold: f64,
    pub verbose: bool,
    // 3D triple-resonance adaptive tolerances
    pub triple_res_hn_tolerance: f64,
    pub triple_res_c_tolerance: f64,
    pub triple_res_sequential_weight: f64,
}

impl UnifiedAssignmentParams {
    /// Get interpolated parameters for a given iteration progress (0.0 to 1.0).
    pub fn interpolate(&self, progress: f64) -> InterpolatedParams {
        // Use sigmoid-like schedule for smoother transition
        let t = if progress < self.exploration_fraction {
            // Exploration phase: stay loose
            0.0
        } else {
            // Refinement phase: linear tightening
            (progress - self.exploration_fraction) / (1.0 - self.exploration_fraction)
        };

        // Exponential interpolation for tolerances (tighten faster at end)
        let t_exp = t * t;  // Quadratic: slow start, fast finish

        InterpolatedParams {
            h_tolerance: self.h_tolerance_initial * (1.0 - t_exp) + self.h_tolerance_final * t_exp,
            c_tolerance: self.c_tolerance_initial * (1.0 - t_exp) + self.c_tolerance_final * t_exp,
            tocsy_weight: self.tocsy_weight_initial * (1.0 - t) + self.tocsy_weight_final * t,
            typing_weight: self.typing_weight_initial * (1.0 - t) + self.typing_weight_final * t,
            sequential_weight: self.sequential_weight,
            // Sequence-type constraint strengthens during refinement
            sequence_type_weight: self.sequence_type_weight * t,  // Start at 0, ramp up
            sequence_type_threshold: self.sequence_type_confidence_threshold,
            verbose: self.verbose,
            // 3D tolerances: same quadratic tightening schedule
            triple_res_hn_tolerance: self.triple_res_hn_tolerance_initial * (1.0 - t_exp)
                + self.triple_res_hn_tolerance_final * t_exp,
            triple_res_c_tolerance: self.triple_res_c_tolerance_initial * (1.0 - t_exp)
                + self.triple_res_c_tolerance_final * t_exp,
            triple_res_sequential_weight: self.triple_res_sequential_weight,
        }
    }
}

/// A peak in the unified graph with its observed data.
#[derive(Debug, Clone)]
pub struct ObservedPeak {
    pub id: Uuid,
    pub peak_type: PeakType,
    pub h_shift: f64,           // Proton chemical shift
    pub heavy_shift: Option<f64>, // N for 15N-HSQC, C for 13C-HSQC
    pub intensity: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PeakType {
    Backbone,   // 15N-HSQC (has both H and N)
    Carbon,     // 13C-HSQC (has C and attached H)
}

/// Result of unified assignment.
#[derive(Debug, Clone)]
pub struct UnifiedAssignmentResult {
    pub peak_id: Uuid,
    pub assigned_residue: i32,  // 0 = unassigned, 1..N = residue position
    pub confidence: f64,
    pub peak_type: PeakType,
}

/// Carbon observation from a 3D triple-resonance experiment, linked to a backbone NH.
/// Used for both typing (CA/CB shifts are diagnostic) and sequential assignment.
#[derive(Debug, Clone)]
struct TripleResCarbonObs {
    /// Index of the backbone peak this observation is linked to (via H/N matching)
    backbone_idx: usize,
    /// Carbon chemical shift (CA, CB, or CO)
    carbon_shift: f64,
    /// True = CA, False = CB (for experiments that distinguish)
    is_ca: bool,
    /// True = intra-residue (i), False = inter-residue (i-1)
    is_intra: bool,
    /// Source experiment type
    source: PeakExperimentType,
}

/// Sequential connectivity evidence from matching CA/CB shifts across backbone peaks.
/// When CA(i) from HNCA at backbone A matches CA(i-1) from CBCACONH at backbone B,
/// we have strong evidence that A's residue precedes B's residue in sequence.
#[derive(Debug, Clone)]
struct TripleResSequentialLink {
    /// Backbone peak index for position i (the preceding residue)
    from_backbone_idx: usize,
    /// Backbone peak index for position i+1 (the following residue)
    to_backbone_idx: usize,
    /// Best CA ppm difference (absolute value) - quality computed dynamically from tolerance
    ca_ppm_diff: f64,
    /// Best CB ppm difference if available (absolute value)
    cb_ppm_diff: Option<f64>,
}

impl TripleResSequentialLink {
    /// Compute quality based on current tolerance (0 = perfect match, 1 = at tolerance limit)
    /// Returns None if link is outside current tolerance
    fn quality(&self, c_tolerance: f64) -> Option<f64> {
        if self.ca_ppm_diff > c_tolerance {
            return None;  // CA mismatch - link not valid at this tolerance
        }
        let ca_quality = 1.0 - (self.ca_ppm_diff / c_tolerance);
        let cb_quality = self.cb_ppm_diff.and_then(|diff| {
            if diff <= c_tolerance {
                Some(1.0 - (diff / c_tolerance))
            } else {
                None
            }
        });
        // CA quality + 50% bonus for CB match
        Some(ca_quality + cb_quality.unwrap_or(0.0) * 0.5)
    }
}

/// Spin system proton profile for amino acid typing based on proton count/pattern.
/// Built from HSQC-TOCSY data - counts unique protons per backbone anchor.
#[derive(Debug, Clone)]
struct SpinSystemProtonProfile {
    /// Index of backbone peak this profile corresponds to
    backbone_idx: usize,
    /// Number of distinct protons in this spin system
    proton_count: usize,
    /// All proton shifts in this system
    proton_shifts: Vec<f64>,
    /// Has methyl protons (< 1.5 ppm, typically multiple degenerate peaks)
    has_methyl: bool,
    /// Has aromatic protons (6.5-8.0 ppm range, excluding backbone amide)
    has_aromatic: bool,
}

/// The unified factor graph.
pub struct UnifiedFactorGraph {
    /// All peaks (backbone + carbon)
    peaks: Vec<ObservedPeak>,
    /// Sequence (one-letter codes)
    sequence: String,
    /// Residue types for each position (3-letter codes)
    residue_types: Vec<String>,
    /// Domain size (0 = unassigned, 1..N = residues)
    domain_size: usize,
    /// TOCSY correlation pairs: (peak_idx_i, peak_idx_j, proton_diff_ppm)
    /// Stores raw proton difference so strength can be recalculated adaptively
    tocsy_correlations: Vec<(usize, usize, f64)>,
    /// HSQC-TOCSY correlation pairs: (peak_idx_i, peak_idx_j, heavy_diff_ppm)
    /// Peaks sharing same heavy atom anchor belong to same residue.
    /// Higher confidence than regular TOCSY since heavy-atom-anchored.
    hsqc_tocsy_correlations: Vec<(usize, usize, f64)>,
    /// Raw NOESY cross-peaks: (backbone_idx, carbon_idx, quality)
    /// Quality = proton mismatch in ppm (smaller = better match)
    /// Sequential relationships computed dynamically from beliefs during BP
    noesy_backbone_carbon: Vec<(usize, usize, f64)>,
    /// Carbon typing scores: peak_idx -> residue_pos -> score
    carbon_typing_scores: Vec<Vec<f64>>,
    /// Beliefs (marginals) for each peak
    beliefs: Vec<Array1<f64>>,
    /// Messages between peaks (for pairwise factors)
    messages: HashMap<(usize, usize), Array1<f64>>,
    /// BMRB database reference
    bmrb: BMRBDatabase,
    /// Cached KDE scorer (avoids reloading database on every score call)
    kde_scorer: KDEScorer,
    /// KDE database for proton pattern typing
    kde_database: KDEDatabase,
    /// Valid positions for each amino acid type: type_name -> [positions]
    /// e.g., "SER" -> [15] means SER only at position 15
    type_to_positions: HashMap<String, Vec<usize>>,
    /// Verbose mode - print detailed information about the algorithm
    verbose: bool,

    // === 3D Triple-Resonance Data (processed for BP) ===
    /// Carbon observations from 3D experiments, linked to backbone peaks
    triple_res_carbons: Vec<TripleResCarbonObs>,
    /// Sequential links derived from CA/CB matching across backbone peaks
    triple_res_sequential: Vec<TripleResSequentialLink>,

    // === HSQC-TOCSY Proton Profiles (for amino acid typing) ===
    /// Proton count profiles for each backbone peak, built from HSQC-TOCSY
    proton_profiles: Vec<SpinSystemProtonProfile>,
}

impl UnifiedFactorGraph {
    /// Create a new unified factor graph from peaks and sequence.
    pub fn new(
        hsqc_15n: &[UnlabeledPeak],
        hsqc_13c: &[UnlabeledPeak],
        tocsy: &[UnlabeledPeak],
        noesy: &[UnlabeledPeak],
        hsqc_tocsy_15n: &[UnlabeledPeak],
        hsqc_tocsy_13c: &[UnlabeledPeak],
        // 3D HSQC-TOCSY experiments (stored for future BP integration)
        _hsqc_tocsy_15n_3d: &[UnlabeledPeak],
        _hsqc_tocsy_13c_3d: &[UnlabeledPeak],
        // 3D triple-resonance experiments
        hnco: &[UnlabeledPeak],
        hnca: &[UnlabeledPeak],
        hncacb: &[UnlabeledPeak],
        cbcaconh: &[UnlabeledPeak],
        hbhaconh: &[UnlabeledPeak],
        sequence: &str,
        params: &UnifiedAssignmentParams,
    ) -> Self {
        let bmrb = BMRBDatabase::load_embedded();

        // Convert sequence to residue types
        let residue_types: Vec<String> = sequence.chars()
            .map(|c| one_letter_to_three(&c))
            .collect();

        // Domain: 0 (unassigned) + N residues
        let domain_size = sequence.len() + 1;

        // Collect all peaks
        let mut peaks = Vec::new();

        // Add backbone peaks (15N-HSQC)
        for peak in hsqc_15n {
            if peak.experiment_type == PeakExperimentType::Hsqc15N {
                peaks.push(ObservedPeak {
                    id: peak.id,
                    peak_type: PeakType::Backbone,
                    h_shift: peak.position_ppm[1],  // H is second dimension
                    heavy_shift: Some(peak.position_ppm[0]),  // N is first dimension
                    intensity: peak.intensity,
                });
            }
        }

        // Add carbon peaks (13C-HSQC)
        for peak in hsqc_13c {
            if peak.experiment_type == PeakExperimentType::Hsqc13C {
                peaks.push(ObservedPeak {
                    id: peak.id,
                    peak_type: PeakType::Carbon,
                    h_shift: peak.position_ppm[1],  // H is second dimension
                    heavy_shift: Some(peak.position_ppm[0]),  // C is first dimension
                    intensity: peak.intensity,
                });
            }
        }

        // === HSQC-TOCSY as Primary Data Source ===
        // If no primary HSQC peaks, create virtual peaks from HSQC-TOCSY

        // Create virtual backbone peaks from 15N-HSQC-TOCSY
        let backbone_count = peaks.iter().filter(|p| p.peak_type == PeakType::Backbone).count();
        if backbone_count == 0 && !hsqc_tocsy_15n.is_empty() {
            // Extract unique N shifts and find corresponding backbone H
            use std::collections::HashSet;
            let mut seen_n: HashSet<i32> = HashSet::new();  // Discretized N shift

            for peak in hsqc_tocsy_15n {
                if peak.experiment_type != PeakExperimentType::HsqcTocsy15N {
                    continue;
                }
                let n_shift = peak.position_ppm[0];
                let h_shift = peak.position_ppm[1];

                // Discretize N to 0.1 ppm bins for uniqueness check
                let n_bin = (n_shift * 10.0).round() as i32;

                // Only create one backbone peak per unique N
                if seen_n.contains(&n_bin) {
                    continue;
                }

                // Check if this H is in backbone amide range (6.5-10.0 ppm)
                if h_shift >= 6.5 && h_shift <= 10.0 {
                    seen_n.insert(n_bin);
                    peaks.push(ObservedPeak {
                        id: peak.id,  // Use the HSQC-TOCSY peak ID
                        peak_type: PeakType::Backbone,
                        h_shift,
                        heavy_shift: Some(n_shift),
                        intensity: peak.intensity,
                    });
                }
            }

            if params.verbose {
                println!("\n[HSQC-TOCSY Primary] Created {} virtual backbone peaks from 15N-HSQC-TOCSY",
                    peaks.iter().filter(|p| p.peak_type == PeakType::Backbone).count());
            }
        }

        // Create virtual carbon peaks from 13C-HSQC-TOCSY
        let carbon_count = peaks.iter().filter(|p| p.peak_type == PeakType::Carbon).count();
        if carbon_count == 0 && !hsqc_tocsy_13c.is_empty() {
            // Extract unique (C, H_attached) pairs
            use std::collections::HashSet;
            let mut seen_ch: HashSet<(i32, i32)> = HashSet::new();  // Discretized (C, H)

            for peak in hsqc_tocsy_13c {
                if peak.experiment_type != PeakExperimentType::HsqcTocsy13C {
                    continue;
                }
                let c_shift = peak.position_ppm[0];
                let h_shift = peak.position_ppm[1];

                // Discretize to 0.1 ppm bins
                let c_bin = (c_shift * 10.0).round() as i32;
                let h_bin = (h_shift * 10.0).round() as i32;

                // Only create one carbon peak per unique (C, H) pair
                if seen_ch.contains(&(c_bin, h_bin)) {
                    continue;
                }

                // Skip backbone amide protons (we want aliphatic C-H correlations)
                if h_shift >= 6.5 && h_shift <= 10.0 {
                    continue;
                }

                seen_ch.insert((c_bin, h_bin));
                peaks.push(ObservedPeak {
                    id: peak.id,
                    peak_type: PeakType::Carbon,
                    h_shift,
                    heavy_shift: Some(c_shift),
                    intensity: peak.intensity,
                });
            }

            if params.verbose {
                println!("[HSQC-TOCSY Primary] Created {} virtual carbon peaks from 13C-HSQC-TOCSY",
                    peaks.iter().filter(|p| p.peak_type == PeakType::Carbon).count());
            }
        }

        // Build TOCSY correlations
        let tocsy_correlations = build_tocsy_correlations(&peaks, tocsy, params);

        // Build HSQC-TOCSY correlations (peaks sharing same heavy atom anchor)
        let hsqc_tocsy_correlations = build_hsqc_tocsy_correlations(
            &peaks, hsqc_tocsy_15n, hsqc_tocsy_13c, params
        );

        // Build raw NOESY backbone-carbon connections (sequential logic computed dynamically in BP)
        let noesy_backbone_carbon = build_noesy_backbone_carbon(&peaks, noesy, params);

        // Compute carbon typing scores
        let carbon_typing_scores = compute_carbon_typing_scores(&peaks, &residue_types, &bmrb, params);

        // Initialize beliefs to uniform
        let beliefs: Vec<Array1<f64>> = peaks.iter()
            .map(|_| Array1::from_elem(domain_size, 1.0 / domain_size as f64))
            .collect();

        // Create KDE scorer and database once (expensive to load, so cache it)
        let kde_scorer = KDEScorer::new();
        let kde_database = KDEDatabase::load_embedded();

        // Build type → positions map for sequence-type constraint
        // e.g., if sequence is "ACDEFGHIKLMNQRSTVWY":
        //   "ALA" -> [1], "CYS" -> [2], "ASP" -> [3], ...
        // If a type appears multiple times, it has multiple valid positions
        let mut type_to_positions: HashMap<String, Vec<usize>> = HashMap::new();
        for (i, res_type) in residue_types.iter().enumerate() {
            type_to_positions
                .entry(res_type.clone())
                .or_insert_with(Vec::new)
                .push(i + 1);  // Position is 1-indexed
        }

        // Verbose: print input summary
        if params.verbose {
            println!("\n═══════════════════════════════════════════════════════════════════");
            println!("                    UNIFIED ASSIGNMENT - INPUT SUMMARY              ");
            println!("═══════════════════════════════════════════════════════════════════");
            println!("\nSequence: {} ({} residues)", sequence, sequence.len());
            println!("Residue types: {:?}", residue_types);

            println!("\n--- TYPE → POSITIONS MAP ---");
            for (aa_type, positions) in &type_to_positions {
                println!("  {} appears at positions: {:?}", aa_type, positions);
            }

            println!("\n--- PEAKS ({} total) ---", peaks.len());
            let bb_count = peaks.iter().filter(|p| p.peak_type == PeakType::Backbone).count();
            let c_count = peaks.iter().filter(|p| p.peak_type == PeakType::Carbon).count();
            println!("  Backbone (15N-HSQC): {}", bb_count);
            println!("  Carbon (13C-HSQC): {}", c_count);

            for (idx, peak) in peaks.iter().enumerate() {
                let type_str = match peak.peak_type {
                    PeakType::Backbone => "BB",
                    PeakType::Carbon => "C ",
                };
                println!("  Peak {:2} [{}]: H={:.3} ppm, heavy={:.1} ppm",
                    idx, type_str, peak.h_shift,
                    peak.heavy_shift.unwrap_or(0.0));
            }

            println!("\n--- TOCSY CORRELATIONS ({}) ---", tocsy_correlations.len());
            for (i, j, diff) in &tocsy_correlations {
                let pi = &peaks[*i];
                let pj = &peaks[*j];
                println!("  Peak {} (H={:.3}) ↔ Peak {} (H={:.3}) | Δppm={:.4}",
                    i, pi.h_shift, j, pj.h_shift, diff);
            }

            println!("\n--- NOESY BACKBONE-CARBON CORRELATIONS ({}) ---", noesy_backbone_carbon.len());
            println!("  (Sequential relationships computed dynamically from beliefs during BP)");
            for (bb_idx, c_idx, quality) in &noesy_backbone_carbon {
                let bb = &peaks[*bb_idx];
                let c = &peaks[*c_idx];
                println!("  Backbone {} (H={:.3}) ↔ Carbon {} (H={:.3}) | quality={:.4}",
                    bb_idx, bb.h_shift, c_idx, c.h_shift, quality);
            }

            println!("\n--- CARBON TYPING SCORES (KDE-based) ---");
            for (peak_idx, peak) in peaks.iter().enumerate() {
                if peak.peak_type == PeakType::Carbon {
                    if let Some(c_shift) = peak.heavy_shift {
                        println!("  Peak {} (C={:.1} ppm):", peak_idx, c_shift);
                        // Show top 3 residue types for this carbon
                        let mut scores: Vec<(usize, &str, f64)> = residue_types.iter()
                            .enumerate()
                            .map(|(r, res_type)| (r + 1, res_type.as_str(), carbon_typing_scores[peak_idx][r + 1]))
                            .collect();
                        scores.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
                        for (pos, res_type, score) in scores.iter().take(3) {
                            println!("    {} at pos {}: score={:.4}", res_type, pos, score);
                        }
                    }
                }
            }
            println!("═══════════════════════════════════════════════════════════════════\n");
        }

        // === Process 3D triple-resonance experiments ===
        // Link 3D peaks to backbone peaks by matching H/N coordinates
        // Use INITIAL (loose) tolerances for discovering potential links
        // Quality is computed dynamically during BP using current tolerance
        let hn_tolerance = params.triple_res_hn_tolerance_initial;
        let n_tolerance = params.triple_res_hn_tolerance_initial * 10.0;  // N tolerance ~10x H tolerance
        // Use loose tolerance to discover ALL potential CA/CB matches
        // Links outside current tolerance are filtered during BP
        let carbon_match_tolerance = params.triple_res_c_tolerance_initial;

        // Collect backbone peak info for matching: (bb_idx, H_shift, N_shift)
        let backbone_info: Vec<(usize, f64, f64)> = peaks.iter()
            .enumerate()
            .filter(|(_, p)| p.peak_type == PeakType::Backbone)
            .filter_map(|(idx, p)| {
                p.heavy_shift.map(|n| (idx, p.h_shift, n))
            })
            .collect();

        // Helper to find backbone peak matching H/N
        let find_backbone = |h: f64, n: f64| -> Option<usize> {
            backbone_info.iter()
                .find(|(_, bh, bn)| (h - bh).abs() < hn_tolerance && (n - bn).abs() < n_tolerance)
                .map(|(idx, _, _)| *idx)
        };

        let mut triple_res_carbons: Vec<TripleResCarbonObs> = Vec::new();

        // Process HNCO: (H, N, CO) - CO(i-1) from NH(i)
        for peak in hnco {
            if peak.position_ppm.len() >= 3 {
                let (h, n, co) = (peak.position_ppm[0], peak.position_ppm[1], peak.position_ppm[2]);
                if let Some(bb_idx) = find_backbone(h, n) {
                    triple_res_carbons.push(TripleResCarbonObs {
                        backbone_idx: bb_idx,
                        carbon_shift: co,
                        is_ca: false,  // CO is neither CA nor CB
                        is_intra: false,  // CO is always from i-1
                        source: PeakExperimentType::Hnco,
                    });
                }
            }
        }

        // Process HNCA: (H, N, CA) - CA(i) strong, CA(i-1) weak
        // Distinguish by intensity: strong (>0.5) = intra, weak = inter
        for peak in hnca {
            if peak.position_ppm.len() >= 3 {
                let (h, n, ca) = (peak.position_ppm[0], peak.position_ppm[1], peak.position_ppm[2]);
                if let Some(bb_idx) = find_backbone(h, n) {
                    let is_intra = peak.intensity > 0.5;  // Strong = intra-residue
                    triple_res_carbons.push(TripleResCarbonObs {
                        backbone_idx: bb_idx,
                        carbon_shift: ca,
                        is_ca: true,
                        is_intra,
                        source: PeakExperimentType::Hnca,
                    });
                }
            }
        }

        // Process HNCACB: (H, N, CA/CB) - sign encodes intra(+) vs inter(-)
        // Positive intensity = intra-residue, negative = inter-residue
        // CA typically 50-65 ppm, CB typically 15-45 ppm (except THR/SER ~60-70)
        for peak in hncacb {
            if peak.position_ppm.len() >= 3 {
                let (h, n, c) = (peak.position_ppm[0], peak.position_ppm[1], peak.position_ppm[2]);
                if let Some(bb_idx) = find_backbone(h, n) {
                    let is_intra = peak.intensity > 0.0;  // Sign convention
                    // CA is typically 50-65 ppm, CB is typically 15-45 ppm
                    // Exception: THR/SER CB ~60-70 ppm, but we'll use shift threshold
                    let is_ca = c > 45.0;  // Simple heuristic
                    triple_res_carbons.push(TripleResCarbonObs {
                        backbone_idx: bb_idx,
                        carbon_shift: c,
                        is_ca,
                        is_intra,
                        source: PeakExperimentType::Hncacb,
                    });
                }
            }
        }

        // Process CBCACONH: (H, N, CA/CB) - i-1 only
        for peak in cbcaconh {
            if peak.position_ppm.len() >= 3 {
                let (h, n, c) = (peak.position_ppm[0], peak.position_ppm[1], peak.position_ppm[2]);
                if let Some(bb_idx) = find_backbone(h, n) {
                    let is_ca = c > 45.0;
                    triple_res_carbons.push(TripleResCarbonObs {
                        backbone_idx: bb_idx,
                        carbon_shift: c,
                        is_ca,
                        is_intra: false,  // CBCACONH always shows i-1
                        source: PeakExperimentType::Cbcaconh,
                    });
                }
            }
        }

        // Process HBHACONH: (H, N, HA/HB) - i-1 protons (for future use)
        // Currently not used for BP but stored for potential future enhancement
        let _hbhaconh_count = hbhaconh.len();  // Acknowledge but don't store

        // === Build sequential links from CA/CB matching ===
        // For each backbone peak, collect its intra-residue CA(i) and inter-residue CA(i-1)
        // Then match: if backbone A has CA(i) that matches backbone B's CA(i-1), A precedes B
        let mut triple_res_sequential: Vec<TripleResSequentialLink> = Vec::new();

        // Group carbon observations by backbone index
        let mut intra_ca_by_bb: HashMap<usize, Vec<f64>> = HashMap::new();
        let mut inter_ca_by_bb: HashMap<usize, Vec<f64>> = HashMap::new();
        let mut intra_cb_by_bb: HashMap<usize, Vec<f64>> = HashMap::new();
        let mut inter_cb_by_bb: HashMap<usize, Vec<f64>> = HashMap::new();

        for obs in &triple_res_carbons {
            if obs.is_ca {
                if obs.is_intra {
                    intra_ca_by_bb.entry(obs.backbone_idx).or_default().push(obs.carbon_shift);
                } else {
                    inter_ca_by_bb.entry(obs.backbone_idx).or_default().push(obs.carbon_shift);
                }
            } else if obs.source != PeakExperimentType::Hnco {  // Exclude CO
                if obs.is_intra {
                    intra_cb_by_bb.entry(obs.backbone_idx).or_default().push(obs.carbon_shift);
                } else {
                    inter_cb_by_bb.entry(obs.backbone_idx).or_default().push(obs.carbon_shift);
                }
            }
        }

        // Find sequential matches: CA(i) at backbone A matches CA(i-1) at backbone B
        // Store raw ppm differences - quality computed dynamically based on current tolerance
        // Use a HashMap to deduplicate and keep only the best (smallest diff) link per (from, to) pair
        let mut best_links: HashMap<(usize, usize), TripleResSequentialLink> = HashMap::new();

        for (&bb_a, ca_intra_shifts) in &intra_ca_by_bb {
            for (&bb_b, ca_inter_shifts) in &inter_ca_by_bb {
                if bb_a == bb_b { continue; }  // Can't be sequential with itself

                // Find the smallest CA difference between A's intra and B's inter
                let mut best_ca_diff = f64::MAX;
                for &ca_a in ca_intra_shifts {
                    for &ca_b in ca_inter_shifts {
                        let ca_diff = (ca_a - ca_b).abs();
                        if ca_diff < carbon_match_tolerance {
                            best_ca_diff = best_ca_diff.min(ca_diff);
                        }
                    }
                }

                if best_ca_diff < f64::MAX {
                    // Find smallest CB difference as additional evidence
                    let best_cb_diff = intra_cb_by_bb.get(&bb_a)
                        .and_then(|cb_intra| {
                            inter_cb_by_bb.get(&bb_b).and_then(|cb_inter| {
                                let mut min_cb = f64::MAX;
                                for &cb_a in cb_intra {
                                    for &cb_b in cb_inter {
                                        let cb_diff = (cb_a - cb_b).abs();
                                        if cb_diff < carbon_match_tolerance {
                                            min_cb = min_cb.min(cb_diff);
                                        }
                                    }
                                }
                                if min_cb < f64::MAX { Some(min_cb) } else { None }
                            })
                        });

                    let link = TripleResSequentialLink {
                        from_backbone_idx: bb_a,
                        to_backbone_idx: bb_b,
                        ca_ppm_diff: best_ca_diff,
                        cb_ppm_diff: best_cb_diff,
                    };

                    // Keep the link with smallest CA diff for this (from, to) pair
                    let key = (bb_a, bb_b);
                    if let Some(existing) = best_links.get(&key) {
                        // Smaller ppm diff = better match
                        if best_ca_diff < existing.ca_ppm_diff {
                            best_links.insert(key, link);
                        }
                    } else {
                        best_links.insert(key, link);
                    }
                }
            }
        }

        let triple_res_sequential: Vec<TripleResSequentialLink> = best_links.into_values().collect();

        if params.verbose && (!triple_res_carbons.is_empty() || !triple_res_sequential.is_empty()) {
            println!("\n--- 3D TRIPLE-RESONANCE DATA ---");
            println!("  Carbon observations linked to backbone: {}", triple_res_carbons.len());
            println!("  Initial tolerances: H/N={:.3} ppm, CA/CB={:.2} ppm",
                hn_tolerance, carbon_match_tolerance);
            // Show intra and inter CA shifts for each backbone
            println!("\n  CA shifts by backbone (for sequential matching):");
            for bb_idx in 0..peaks.iter().filter(|p| p.peak_type == PeakType::Backbone).count() {
                let intra_cas: Vec<f64> = intra_ca_by_bb.get(&bb_idx).map_or(vec![], |v| v.clone());
                let inter_cas: Vec<f64> = inter_ca_by_bb.get(&bb_idx).map_or(vec![], |v| v.clone());
                if !intra_cas.is_empty() || !inter_cas.is_empty() {
                    println!("    BB {}: intra CA(i)={:?}, inter CA(i-1)={:?}",
                        bb_idx,
                        intra_cas.iter().map(|x| format!("{:.1}", x)).collect::<Vec<_>>(),
                        inter_cas.iter().map(|x| format!("{:.1}", x)).collect::<Vec<_>>());
                }
            }
            println!("\n  Sequential links from CA/CB matching (ppm diffs): {}", triple_res_sequential.len());
            for link in &triple_res_sequential {
                let cb_str = link.cb_ppm_diff.map_or("none".to_string(), |d| format!("{:.3}", d));
                println!("    BB {} → BB {} (CA Δppm={:.3}, CB Δppm={})",
                    link.from_backbone_idx, link.to_backbone_idx, link.ca_ppm_diff, cb_str);
            }
        }

        // Build proton profiles from HSQC-TOCSY data
        let proton_profiles = build_proton_profiles(&peaks, hsqc_tocsy_15n, params);

        if params.verbose && !proton_profiles.is_empty() {
            println!("\n--- PROTON PROFILES (from HSQC-TOCSY) ---");
            for profile in &proton_profiles {
                let bb_peak = &peaks[profile.backbone_idx];
                // Format proton shifts (sort for readability)
                let mut shifts = profile.proton_shifts.clone();
                shifts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let shifts_str: Vec<String> = shifts.iter().map(|s| format!("{:.2}", s)).collect();
                println!("  BB {} (N={:.1}): {} protons [{}], methyl={}, aromatic={}",
                    profile.backbone_idx,
                    bb_peak.heavy_shift.unwrap_or(0.0),
                    profile.proton_count,
                    shifts_str.join(", "),
                    profile.has_methyl,
                    profile.has_aromatic
                );
            }
        }

        Self {
            peaks,
            sequence: sequence.to_string(),
            residue_types,
            domain_size,
            tocsy_correlations,
            hsqc_tocsy_correlations,
            noesy_backbone_carbon,
            carbon_typing_scores,
            beliefs,
            messages: HashMap::new(),
            bmrb,
            kde_scorer,
            kde_database,
            type_to_positions,
            verbose: params.verbose,
            triple_res_carbons,
            triple_res_sequential,
            proton_profiles,
        }
    }

    /// Run belief propagation with adaptive tolerances.
    pub fn run_bp(&mut self, params: &UnifiedAssignmentParams) {
        // Initialize messages
        self.initialize_messages();

        if self.verbose {
            println!("\n═══════════════════════════════════════════════════════════════════");
            println!("                    BELIEF PROPAGATION - STARTING                    ");
            println!("═══════════════════════════════════════════════════════════════════");
            println!("Max iterations: {}", params.max_iterations);
            println!("Exploration fraction: {:.0}%", params.exploration_fraction * 100.0);
            println!("Convergence threshold: {:e}", params.convergence_threshold);
        }

        let mut last_progress_log = 0;

        for iter in 0..params.max_iterations {
            // Compute progress (0.0 to 1.0)
            let progress = iter as f64 / params.max_iterations as f64;

            // Get interpolated parameters for this iteration
            let interp = params.interpolate(progress);

            // Recompute carbon typing scores with current tolerance
            self.update_carbon_typing_scores(&interp);

            // Run one BP iteration with current parameters
            let max_change = self.bp_iteration_adaptive(&interp, params);

            // Verbose: show detailed progress at key iterations
            let progress_pct = (progress * 100.0) as i32;
            if self.verbose && (iter == 0 || progress_pct >= last_progress_log + 25 || iter == params.max_iterations - 1) {
                let phase = if progress < params.exploration_fraction { "EXPLORE" } else { "REFINE" };
                println!("\n┌─── Iteration {} ({:.0}% - {}) ───────────────────────────────", iter, progress * 100.0, phase);
                println!("│ Parameters:");
                println!("│   H tolerance: {:.4} ppm", interp.h_tolerance);
                println!("│   C tolerance: {:.2} ppm", interp.c_tolerance);
                println!("│   TOCSY weight: {:.2}", interp.tocsy_weight);
                println!("│   Typing weight: {:.2}", interp.typing_weight);
                println!("│   Seq-type weight: {:.2}", interp.sequence_type_weight);
                println!("│ Max belief change: {:.6}", max_change);

                // Show current beliefs for backbone peaks
                println!("│");
                println!("│ Current backbone peak beliefs (showing top 3):");
                for (idx, peak) in self.peaks.iter().enumerate() {
                    if peak.peak_type == PeakType::Backbone {
                        let belief = &self.beliefs[idx];
                        // Get top 3 positions
                        let mut positions: Vec<(usize, f64)> = belief.iter().enumerate()
                            .skip(1)  // Skip unassigned
                            .map(|(r, &p)| (r, p))
                            .collect();
                        positions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                        let best_type = self.residue_types.get(positions[0].0 - 1).map(|s| s.as_str()).unwrap_or("?");
                        println!("│   Peak {:2} (H={:.3}, N={:.1}): pos {} ({}) {:.1}%",
                            idx, peak.h_shift, peak.heavy_shift.unwrap_or(0.0),
                            positions[0].0, best_type, positions[0].1 * 100.0);
                        if positions.len() > 1 && positions[1].1 > 0.05 {
                            let type2 = self.residue_types.get(positions[1].0 - 1).map(|s| s.as_str()).unwrap_or("?");
                            println!("│              ↳ runner-up: pos {} ({}) {:.1}%",
                                positions[1].0, type2, positions[1].1 * 100.0);
                        }
                    }
                }
                println!("└───────────────────────────────────────────────────────────────");
                last_progress_log = progress_pct;
            } else if progress_pct >= last_progress_log + 25 {
                tracing::debug!(
                    "Adaptive BP {}%: h_tol={:.3}, c_tol={:.2}, tocsy_w={:.1}, typing_w={:.1}",
                    progress_pct, interp.h_tolerance, interp.c_tolerance,
                    interp.tocsy_weight, interp.typing_weight
                );
                last_progress_log = progress_pct;
            }

            // Early convergence only in refinement phase
            if progress > params.exploration_fraction && max_change < params.convergence_threshold {
                if self.verbose {
                    println!("\n✓ CONVERGED after {} iterations (change={:.8} < threshold)", iter + 1, max_change);
                }
                tracing::info!("Adaptive BP converged after {} iterations", iter + 1);
                return;
            }
        }

        if self.verbose {
            println!("\n✓ Completed all {} iterations", params.max_iterations);
        }
        tracing::info!("Adaptive BP completed {} iterations", params.max_iterations);
    }

    /// Update carbon typing scores with current tolerance.
    fn update_carbon_typing_scores(&mut self, _interp: &InterpolatedParams) {
        for peak_idx in 0..self.peaks.len() {
            if self.peaks[peak_idx].peak_type == PeakType::Carbon {
                if let Some(c_shift) = self.peaks[peak_idx].heavy_shift {
                    for r in 0..self.residue_types.len() {
                        let score = score_carbon_for_residue_kde(
                            c_shift, &self.residue_types[r], &self.kde_scorer
                        );
                        self.carbon_typing_scores[peak_idx][r + 1] = score;
                    }
                }
            }
        }
    }

    /// Initialize all messages to uniform.
    fn initialize_messages(&mut self) {
        let uniform = Array1::from_elem(self.domain_size, 1.0 / self.domain_size as f64);

        // Messages for TOCSY pairwise factors
        // NOESY sequential is computed directly from beliefs, no messages needed
        for &(i, j, _) in &self.tocsy_correlations {
            self.messages.insert((i, j), uniform.clone());
            self.messages.insert((j, i), uniform.clone());
        }

        // Messages for HSQC-TOCSY pairwise factors
        for &(i, j, _) in &self.hsqc_tocsy_correlations {
            self.messages.insert((i, j), uniform.clone());
            self.messages.insert((j, i), uniform.clone());
        }
    }

    /// One iteration of belief propagation with adaptive parameters.
    fn bp_iteration_adaptive(&mut self, interp: &InterpolatedParams, params: &UnifiedAssignmentParams) -> f64 {
        let mut max_change = 0.0f64;

        // Update beliefs for each peak by combining all factors
        for peak_idx in 0..self.peaks.len() {
            let new_belief = self.compute_belief_adaptive(peak_idx, interp);

            // Track change
            let change: f64 = (&new_belief - &self.beliefs[peak_idx])
                .mapv(|x| x.abs())
                .sum();
            max_change = max_change.max(change);

            // Apply damping
            self.beliefs[peak_idx] = &self.beliefs[peak_idx] * params.damping
                + &new_belief * (1.0 - params.damping);

            // Normalize
            let sum: f64 = self.beliefs[peak_idx].iter().sum();
            if sum > 0.0 {
                self.beliefs[peak_idx] /= sum;
            }
        }

        // Update messages for pairwise factors
        self.update_pairwise_messages_adaptive(interp);

        max_change
    }

    /// Compute belief for a single peak by combining all factors.
    fn compute_belief(&self, peak_idx: usize, params: &UnifiedAssignmentParams) -> Array1<f64> {
        let peak = &self.peaks[peak_idx];
        let mut log_belief = Array1::zeros(self.domain_size);

        // Factor 1: Carbon typing (for 13C-HSQC peaks)
        if peak.peak_type == PeakType::Carbon {
            for r in 0..self.domain_size {
                log_belief[r] += params.typing_weight_initial * self.carbon_typing_scores[peak_idx][r].ln().max(-10.0);
            }
        }

        // Factor 2: Incoming messages from TOCSY-correlated peaks
        for &(i, j, strength) in &self.tocsy_correlations {
            let (other_idx, this_idx) = if i == peak_idx {
                (j, i)
            } else if j == peak_idx {
                (i, j)
            } else {
                continue;
            };

            if let Some(msg) = self.messages.get(&(other_idx, this_idx)) {
                // TOCSY factor: encourage same residue assignment
                for r in 0..self.domain_size {
                    // Weighted by correlation strength and incoming message
                    let tocsy_factor = params.tocsy_weight_initial * strength * msg[r];
                    log_belief[r] += tocsy_factor.max(1e-10).ln();
                }
            }
        }

        // Factor 3: Sequential NOESY (deprecated - use compute_belief_adaptive instead)
        // This old code path is kept for reference but not used
        let _ = &self.noesy_backbone_carbon;  // Acknowledge field exists

        // Factor 4: Backbone uniqueness (soft constraint via other backbone beliefs)
        if peak.peak_type == PeakType::Backbone {
            for (other_idx, other_peak) in self.peaks.iter().enumerate() {
                if other_idx != peak_idx && other_peak.peak_type == PeakType::Backbone {
                    // Penalize assigning to same residue as another backbone peak
                    for r in 1..self.domain_size {
                        let other_prob = self.beliefs[other_idx][r];
                        // Soft exclusion: reduce probability proportionally
                        log_belief[r] -= other_prob * 2.0;
                    }
                }
            }
        }

        // Convert log to probability
        let max_log = log_belief.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut belief = log_belief.mapv(|x| (x - max_log).exp());

        // Normalize
        let sum: f64 = belief.iter().sum();
        if sum > 0.0 {
            belief /= sum;
        }

        belief
    }

    /// Compute belief for a single peak with adaptive parameters.
    fn compute_belief_adaptive(&self, peak_idx: usize, interp: &InterpolatedParams) -> Array1<f64> {
        let peak = &self.peaks[peak_idx];
        let mut log_belief = Array1::zeros(self.domain_size);

        // Factor 0: N-terminus exclusion for backbone peaks
        // Position 1 (first residue) has no backbone NH, so backbone peaks can't be assigned there
        if peak.peak_type == PeakType::Backbone && self.domain_size > 1 {
            log_belief[1] = -100.0;  // Effectively zero probability
        }

        // Factor 1: Carbon typing (for 13C-HSQC peaks) - uses adaptive typing_weight
        if peak.peak_type == PeakType::Carbon {
            for r in 0..self.domain_size {
                // Carbon typing scores are already updated with adaptive tolerance
                log_belief[r] += interp.typing_weight * self.carbon_typing_scores[peak_idx][r].ln().max(-10.0);
            }
        }

        // Factor 1b: Backbone N/H chemical shift typing (for backbone peaks)
        // Uses KDE database to score observed N and H shifts against expected shifts for each residue type
        // This is critical for HSQC-TOCSY-only mode where no carbon peaks are available for typing
        if peak.peak_type == PeakType::Backbone {
            if let Some(n_shift) = peak.heavy_shift {
                let h_shift = peak.h_shift;

                for r in 1..self.domain_size {
                    let res_type = &self.residue_types[r - 1];

                    // Get N shift density from KDE
                    let n_density = self.kde_database.density(res_type, "N", n_shift);
                    let n_score = if n_density > 0.0 { n_density } else { 1e-10 };

                    // Get H shift density from KDE
                    let h_density = self.kde_database.density(res_type, "H", h_shift);
                    let h_score = if h_density > 0.0 { h_density } else { 1e-10 };

                    // Combined backbone typing score (product of densities)
                    let backbone_score = (n_score * h_score).sqrt(); // Geometric mean
                    log_belief[r] += interp.typing_weight * backbone_score.max(1e-10).ln();
                }
            }
        }

        // Factor 1c: 3D Triple-resonance carbon typing (for backbone peaks)
        // Uses CA/CB shifts from HNCACB, CBCACONH, HNCA to help type backbone peaks
        // Key insight: INTRA-residue carbons (is_intra=true) directly tell us CA/CB of THIS residue
        // This is the UNIFIED model: ALL carbon evidence contributes to typing, not just 13C-HSQC
        if peak.peak_type == PeakType::Backbone {
            for obs in &self.triple_res_carbons {
                // Only use carbons linked to THIS backbone peak
                if obs.backbone_idx != peak_idx {
                    continue;
                }

                // Only use INTRA-residue carbons for typing THIS residue
                // Inter-residue carbons (is_intra=false) tell us about i-1, not useful for typing self
                if !obs.is_intra {
                    continue;
                }

                let atom_name = if obs.is_ca { "CA" } else { "CB" };

                for r in 1..self.domain_size {
                    let res_type = &self.residue_types[r - 1];

                    // Get carbon shift density from KDE
                    let c_density = self.kde_database.density(res_type, atom_name, obs.carbon_shift);
                    let c_score = if c_density > 0.0 { c_density } else { 1e-10 };

                    // Add to typing score (same weight as other typing factors)
                    log_belief[r] += interp.typing_weight * c_score.max(1e-10).ln();
                }
            }
        }

        // Factor 2: Incoming messages from TOCSY-correlated peaks - uses adaptive tocsy_weight
        // NORMALIZED: count correlations first, then average their influence
        let mut num_correlations = 0;
        let mut aggregated_msg: Array1<f64> = Array1::zeros(self.domain_size);

        for &(i, j, proton_diff) in &self.tocsy_correlations {
            let (other_idx, this_idx) = if i == peak_idx {
                (j, i)
            } else if j == peak_idx {
                (i, j)
            } else {
                continue;
            };

            // ADAPTIVE STRENGTH: compute based on current tolerance
            let strength = (-0.5 * (proton_diff / interp.h_tolerance).powi(2)).exp();

            if strength < 0.01 {
                continue;
            }

            if let Some(msg) = self.messages.get(&(other_idx, this_idx)) {
                // Aggregate messages (weighted by strength)
                for r in 0..self.domain_size {
                    aggregated_msg[r] += msg[r] * strength;
                }
                num_correlations += 1;
            }
        }

        // DON'T normalize - more correlated carbons = stronger joint evidence
        // (Same insight as sidechain proton scoring: joint probability, not average)
        if num_correlations > 0 {
            // Scale weight by number of correlations to avoid runaway scores
            let per_corr_weight = interp.tocsy_weight / (1.0 + (num_correlations as f64).ln());
            for r in 0..self.domain_size {
                let tocsy_factor: f64 = per_corr_weight * aggregated_msg[r];
                log_belief[r] += tocsy_factor.max(1e-10).ln();
            }
        }

        // Factor 2b: HSQC-TOCSY correlations - same logic as TOCSY but with 1.5x weight
        // HSQC-TOCSY is more reliable than regular TOCSY because peaks are heavy-atom-anchored
        let mut num_ht_correlations = 0;
        let mut ht_aggregated_msg: Array1<f64> = Array1::zeros(self.domain_size);

        for &(i, j, proton_diff) in &self.hsqc_tocsy_correlations {
            let (other_idx, this_idx) = if i == peak_idx {
                (j, i)
            } else if j == peak_idx {
                (i, j)
            } else {
                continue;
            };

            // ADAPTIVE STRENGTH: compute based on current tolerance
            let strength = (-0.5 * (proton_diff / interp.h_tolerance).powi(2)).exp();

            if strength < 0.01 {
                continue;
            }

            if let Some(msg) = self.messages.get(&(other_idx, this_idx)) {
                // Aggregate messages (weighted by strength)
                for r in 0..self.domain_size {
                    ht_aggregated_msg[r] += msg[r] * strength;
                }
                num_ht_correlations += 1;
            }
        }

        // Apply HSQC-TOCSY factor - DON'T normalize (joint probability of spin system)
        if num_ht_correlations > 0 {
            // Scale weight by log(correlations) to avoid runaway scores while preserving joint evidence
            let per_corr_weight = (interp.tocsy_weight * 1.5) / (1.0 + (num_ht_correlations as f64).ln());
            for r in 0..self.domain_size {
                let ht_factor: f64 = per_corr_weight * ht_aggregated_msg[r];
                log_belief[r] += ht_factor.max(1e-10).ln();
            }
        }

        // Factor 3: Sequential NOESY - DYNAMICALLY computed from current beliefs
        // Key insight: dαN NOE means backbone H(i) correlates with HA(i-1)
        // If NOESY shows backbone B correlates with carbon C, and C believes it's at position R,
        // then B should be at position R+1 (B follows C's residue)
        //
        // IMPORTANT:
        // 1. Skip intra-residue correlations (carbon TOCSY-correlated with this backbone)
        // 2. Only BOOST positions where carbon has above-uniform belief
        // 3. NOESY provides positive evidence ("B follows C"), not negative evidence
        if peak.peak_type == PeakType::Backbone {
            let uniform_prob = 1.0 / self.domain_size as f64;

            for &(bb_idx, c_idx, quality) in &self.noesy_backbone_carbon {
                if bb_idx != peak_idx { continue; }

                // Skip if carbon is TOCSY-correlated with this backbone (intra-residue, not sequential)
                let is_intra_residue = self.tocsy_correlations.iter()
                    .any(|&(i, j, _)| (i == peak_idx && j == c_idx) || (j == peak_idx && i == c_idx));
                if is_intra_residue { continue; }

                // Adaptive strength from proton match quality
                let strength = (-0.5 * (quality / interp.h_tolerance).powi(2)).exp();
                if strength < 0.1 { continue; }

                // Use carbon's current belief to inform backbone's position
                // Only boost, never penalize - NOESY is positive evidence
                let carbon_belief = &self.beliefs[c_idx];
                for r in 1..self.domain_size {
                    if r + 1 < self.domain_size {
                        let carbon_prob = carbon_belief[r];
                        // Only apply factor when carbon has STRONG concentrated belief
                        // This avoids noise from diffuse/uncertain carbon assignments
                        if carbon_prob > uniform_prob * 3.0 {
                            // Boost = how much more likely than uniform (as log ratio)
                            let boost = (carbon_prob / uniform_prob).ln();
                            log_belief[r + 1] += interp.sequential_weight * strength * boost;
                        }
                    }
                }
            }
        }

        // Factor 4: Backbone uniqueness - DISABLED during BP
        //
        // Rationale: Uniqueness is an EXTRACTION constraint, not a BP constraint.
        // With limited data (e.g., HSQC-TOCSY only), beliefs are diffuse and mutual
        // suppression causes everything to collapse to zero. Instead:
        // - Let BP converge to the best beliefs given the evidence
        // - Enforce uniqueness during greedy extraction (which already happens)
        //
        // This allows the algorithm to work "organically" with limited datasets,
        // making its best guess even when uncertain, rather than giving up entirely.
        //
        // Old code kept for reference:
        // if peak.peak_type == PeakType::Backbone {
        //     for (other_idx, other_peak) in self.peaks.iter().enumerate() {
        //         if other_idx != peak_idx && other_peak.peak_type == PeakType::Backbone {
        //             for r in 1..self.domain_size {
        //                 let other_prob = self.beliefs[other_idx][r];
        //                 if other_prob > 0.3 {
        //                     log_belief[r] -= (other_prob - 0.3) * 15.0;
        //                 }
        //             }
        //         }
        //     }
        // }

        // Factor 5: Sequence-type constraint
        // If we're confident about the amino acid type, constrain to valid positions
        // For carbon peaks: use direct carbon typing scores
        // For backbone peaks: aggregate from TOCSY-correlated carbons
        let type_scores = self.get_aggregate_type_scores(peak_idx, interp);
        if let Some((best_type, confidence)) = self.get_best_type(&type_scores) {
            if confidence > interp.sequence_type_threshold {
                // Get valid positions for this type
                if let Some(valid_positions) = self.type_to_positions.get(&best_type) {
                    // Penalize positions that don't match the typed amino acid
                    for r in 1..self.domain_size {
                        if !valid_positions.contains(&r) {
                            // Strong penalty for invalid positions
                            log_belief[r] -= interp.sequence_type_weight * confidence;
                        }
                    }
                }
            }
        }

        // Factor 6: 3D Triple-resonance sequential links (ADAPTIVE TOLERANCE)
        // When CA(i) from backbone A matches CA(i-1) from backbone B, A precedes B
        // This provides definitive sequential connectivity from 3D experiments
        //
        // Quality is computed dynamically based on CURRENT tolerance:
        // - Early iterations: loose tolerance → more links considered
        // - Late iterations: tight tolerance → only best matches remain
        if peak.peak_type == PeakType::Backbone {
            let triple_res_weight = interp.triple_res_sequential_weight;
            let c_tolerance = interp.triple_res_c_tolerance;

            // Check links where THIS peak is the "from" backbone (precedes another)
            for link in &self.triple_res_sequential {
                if link.from_backbone_idx == peak_idx {
                    // Compute quality using CURRENT tolerance (adaptive)
                    if let Some(quality) = link.quality(c_tolerance) {
                        // If I'm at position R, the "to" backbone should be at R+1
                        let to_belief = &self.beliefs[link.to_backbone_idx];

                        for r in 1..self.domain_size - 1 {
                            let to_prob = to_belief[r + 1];
                            // Apply boost proportional to other backbone's belief at R+1
                            // This creates mutual reinforcement between sequential backbones
                            let boost = to_prob * quality * triple_res_weight;
                            log_belief[r] += boost;
                        }
                    }
                }
            }

            // Check links where THIS peak is the "to" backbone (follows another)
            for link in &self.triple_res_sequential {
                if link.to_backbone_idx == peak_idx {
                    // Compute quality using CURRENT tolerance (adaptive)
                    if let Some(quality) = link.quality(c_tolerance) {
                        // If I'm at position R, the "from" backbone should be at R-1
                        let from_belief = &self.beliefs[link.from_backbone_idx];

                        for r in 2..self.domain_size {
                            let from_prob = from_belief[r - 1];
                            // Apply boost proportional to other backbone's belief at R-1
                            let boost = from_prob * quality * triple_res_weight;
                            log_belief[r] += boost;
                        }
                    }
                }
            }
        }

        // Factor 7: Sidechain proton typing from HSQC-TOCSY
        // Scores each OBSERVED proton shift against KDE distributions for sidechain atoms
        // Key insight: Don't count protons (peaks overlap, low S/N causes missing peaks)
        // Instead: Score what we DO observe - each proton shift matches some atom type
        if peak.peak_type == PeakType::Backbone && !self.proton_profiles.is_empty() {
            if let Some(profile) = self.proton_profiles.iter()
                .find(|p| p.backbone_idx == peak_idx)
            {
                // Score against each residue type
                for r in 1..self.domain_size {
                    let res_type = &self.residue_types[r - 1];

                    // Score each observed proton shift against this AA type
                    let mut total_log_score = 0.0;
                    let mut num_scored = 0;

                    for &h_shift in &profile.proton_shifts {
                        // Skip backbone amide proton (7-10 ppm) - already scored in Factor 1b
                        if h_shift > 6.5 && h_shift < 10.0 {
                            continue;
                        }

                        // Find best-matching sidechain proton atom for this AA type
                        let best_density = score_proton_shift_for_residue(
                            h_shift, res_type, &self.kde_database
                        );

                        // Add log density (use floor to avoid -infinity)
                        total_log_score += best_density.max(1e-6).ln();
                        num_scored += 1;
                    }

                    // Apply score - use SUM not average (joint probability of spin system)
                    // More matching protons = stronger evidence for that AA type
                    if num_scored > 0 {
                        let proton_weight = interp.typing_weight * 0.5; // Per-proton weight (accumulates)
                        log_belief[r] += proton_weight * total_log_score;
                    }
                }
            }
        }

        // Convert log to probability
        let max_log = log_belief.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut belief = log_belief.mapv(|x| (x - max_log).exp());

        // Normalize
        let sum: f64 = belief.iter().sum();
        if sum > 0.0 {
            belief /= sum;
        }

        belief
    }

    /// Get aggregate type scores for a peak.
    /// For carbon peaks: use direct carbon typing scores
    /// For backbone peaks: aggregate from TOCSY-correlated carbon peaks
    fn get_aggregate_type_scores(&self, peak_idx: usize, interp: &InterpolatedParams) -> HashMap<String, f64> {
        let peak = &self.peaks[peak_idx];
        let mut type_scores: HashMap<String, f64> = HashMap::new();

        if peak.peak_type == PeakType::Carbon {
            // Direct carbon typing: scores are already per-position, convert to per-type
            for (r, res_type) in self.residue_types.iter().enumerate() {
                let score = self.carbon_typing_scores[peak_idx][r + 1];
                *type_scores.entry(res_type.clone()).or_insert(0.0) += score;
            }
        } else if peak.peak_type == PeakType::Backbone {
            // Aggregate from TOCSY-correlated carbon peaks
            let mut num_carbons = 0;
            for &(i, j, proton_diff) in &self.tocsy_correlations {
                let carbon_idx = if i == peak_idx { j } else if j == peak_idx { i } else { continue };

                // Only use carbon peaks
                if self.peaks[carbon_idx].peak_type != PeakType::Carbon {
                    continue;
                }

                // Weight by TOCSY correlation strength
                let strength = (-0.5 * (proton_diff / interp.h_tolerance).powi(2)).exp();
                if strength < 0.1 {
                    continue;
                }

                // Aggregate carbon typing scores
                for (r, res_type) in self.residue_types.iter().enumerate() {
                    let score = self.carbon_typing_scores[carbon_idx][r + 1] * strength;
                    *type_scores.entry(res_type.clone()).or_insert(0.0) += score;
                }
                num_carbons += 1;
            }

            // DON'T normalize - more matching carbons = stronger joint evidence for AA type
            // (Use log-scaling to avoid runaway scores while preserving multiplicative benefit)
            if num_carbons > 1 {
                let scale = 1.0 / (1.0 + (num_carbons as f64).ln());
                for score in type_scores.values_mut() {
                    *score *= scale;
                }
            }
        }

        type_scores
    }

    /// Get the best typed amino acid and confidence from type scores.
    fn get_best_type(&self, type_scores: &HashMap<String, f64>) -> Option<(String, f64)> {
        if type_scores.is_empty() {
            return None;
        }

        // Find best type
        let (best_type, best_score) = type_scores.iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))?;

        // Compute confidence (best score / total scores)
        let total: f64 = type_scores.values().sum();
        let confidence = if total > 0.0 { best_score / total } else { 0.0 };

        Some((best_type.clone(), confidence))
    }

    /// Update pairwise messages with adaptive parameters.
    fn update_pairwise_messages_adaptive(&mut self, interp: &InterpolatedParams) {
        // TOCSY messages: encourage same assignment with adaptive strength
        let tocsy_corr = self.tocsy_correlations.clone();
        for (i, j, proton_diff) in tocsy_corr {
            // ADAPTIVE STRENGTH based on current tolerance
            let strength = (-0.5 * (proton_diff / interp.h_tolerance).powi(2)).exp();

            // Skip very weak correlations
            if strength < 0.01 {
                continue;
            }

            let msg_i_to_j = self.compute_tocsy_message_adaptive(i, j, strength, interp);
            self.messages.insert((i, j), msg_i_to_j);

            let msg_j_to_i = self.compute_tocsy_message_adaptive(j, i, strength, interp);
            self.messages.insert((j, i), msg_j_to_i);
        }

        // HSQC-TOCSY messages: encourage same assignment (uses same message computation as TOCSY)
        let hsqc_tocsy_corr = self.hsqc_tocsy_correlations.clone();
        for (i, j, proton_diff) in hsqc_tocsy_corr {
            // ADAPTIVE STRENGTH based on current tolerance
            let strength = (-0.5 * (proton_diff / interp.h_tolerance).powi(2)).exp();

            // Skip very weak correlations
            if strength < 0.01 {
                continue;
            }

            let msg_i_to_j = self.compute_tocsy_message_adaptive(i, j, strength, interp);
            self.messages.insert((i, j), msg_i_to_j);

            let msg_j_to_i = self.compute_tocsy_message_adaptive(j, i, strength, interp);
            self.messages.insert((j, i), msg_j_to_i);
        }

        // Note: NOESY sequential is now computed directly from beliefs in compute_belief_adaptive
        // No message passing needed - it uses carbon beliefs dynamically
    }

    /// Compute TOCSY message with adaptive weighting.
    fn compute_tocsy_message_adaptive(&self, from: usize, _to: usize, strength: f64, _interp: &InterpolatedParams) -> Array1<f64> {
        let mut msg = self.beliefs[from].clone();

        for r in 0..self.domain_size {
            msg[r] = msg[r] * strength + (1.0 - strength) / self.domain_size as f64;
        }

        let sum: f64 = msg.iter().sum();
        if sum > 0.0 {
            msg /= sum;
        }

        msg
    }

    /// Compute sequential message (forward) with adaptive parameters.
    fn compute_sequential_message_forward_adaptive(&self, from: usize, _to: usize, strength: f64, _interp: &InterpolatedParams) -> Array1<f64> {
        let mut msg = Array1::zeros(self.domain_size);

        for r in 1..self.domain_size {
            if r + 1 < self.domain_size {
                msg[r + 1] += self.beliefs[from][r] * strength;
            }
        }

        let uniform = (1.0 - strength) / self.domain_size as f64;
        msg += uniform;

        let sum: f64 = msg.iter().sum();
        if sum > 0.0 {
            msg /= sum;
        }

        msg
    }

    /// Compute sequential message (backward) with adaptive parameters.
    fn compute_sequential_message_backward_adaptive(&self, from: usize, _to: usize, strength: f64, _interp: &InterpolatedParams) -> Array1<f64> {
        let mut msg = Array1::zeros(self.domain_size);

        for r in 2..self.domain_size {
            msg[r - 1] += self.beliefs[from][r] * strength;
        }

        let uniform = (1.0 - strength) / self.domain_size as f64;
        msg += uniform;

        let sum: f64 = msg.iter().sum();
        if sum > 0.0 {
            msg /= sum;
        }

        msg
    }

    /// Update pairwise messages.
    fn update_pairwise_messages(&mut self, params: &UnifiedAssignmentParams) {
        // TOCSY messages: encourage same assignment
        let tocsy_corr = self.tocsy_correlations.clone();
        for (i, j, strength) in tocsy_corr {
            // Message from i to j
            let msg_i_to_j = self.compute_tocsy_message(i, j, strength, params);
            self.messages.insert((i, j), msg_i_to_j);

            // Message from j to i
            let msg_j_to_i = self.compute_tocsy_message(j, i, strength, params);
            self.messages.insert((j, i), msg_j_to_i);
        }

        // NOESY is now computed directly from beliefs, no message passing needed
        let _ = &self.noesy_backbone_carbon;
    }

    /// Compute TOCSY message (encourages same residue).
    fn compute_tocsy_message(&self, from: usize, _to: usize, strength: f64, _params: &UnifiedAssignmentParams) -> Array1<f64> {
        // Message says: "I think we should both be assigned to residue r"
        // Weighted by my belief and correlation strength
        let mut msg = self.beliefs[from].clone();

        // Scale by strength
        for r in 0..self.domain_size {
            msg[r] = msg[r] * strength + (1.0 - strength) / self.domain_size as f64;
        }

        // Normalize
        let sum: f64 = msg.iter().sum();
        if sum > 0.0 {
            msg /= sum;
        }

        msg
    }

    /// Compute sequential message (forward: i precedes j).
    fn compute_sequential_message_forward(&self, from: usize, _to: usize, strength: f64, _params: &UnifiedAssignmentParams) -> Array1<f64> {
        // If I'm residue r, you should be residue r+1
        let mut msg = Array1::zeros(self.domain_size);

        for r in 1..self.domain_size {
            if r + 1 < self.domain_size {
                msg[r + 1] += self.beliefs[from][r] * strength;
            }
        }

        // Add uniform background
        let uniform = (1.0 - strength) / self.domain_size as f64;
        msg += uniform;

        // Normalize
        let sum: f64 = msg.iter().sum();
        if sum > 0.0 {
            msg /= sum;
        }

        msg
    }

    /// Compute sequential message (backward: j follows i).
    fn compute_sequential_message_backward(&self, from: usize, _to: usize, strength: f64, _params: &UnifiedAssignmentParams) -> Array1<f64> {
        // If I'm residue r, you should be residue r-1
        let mut msg = Array1::zeros(self.domain_size);

        for r in 2..self.domain_size {
            msg[r - 1] += self.beliefs[from][r] * strength;
        }

        // Add uniform background
        let uniform = (1.0 - strength) / self.domain_size as f64;
        msg += uniform;

        // Normalize
        let sum: f64 = msg.iter().sum();
        if sum > 0.0 {
            msg /= sum;
        }

        msg
    }

    /// Extract final assignments from beliefs.
    pub fn extract_assignments(&self) -> Vec<UnifiedAssignmentResult> {
        if self.verbose {
            println!("\n═══════════════════════════════════════════════════════════════════");
            println!("                    EXTRACTING FINAL ASSIGNMENTS                     ");
            println!("═══════════════════════════════════════════════════════════════════");
        }

        let mut results = Vec::new();
        let mut assigned_residues: HashSet<i32> = HashSet::new();

        // Sort peaks by confidence (highest first)
        let mut peak_confidences: Vec<(usize, f64, i32)> = self.peaks.iter()
            .enumerate()
            .map(|(idx, _)| {
                let (best_r, &best_prob) = self.beliefs[idx].iter()
                    .enumerate()
                    .skip(1)  // Skip "unassigned" (r=0)
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .unwrap_or((0, &0.0));
                (idx, best_prob, best_r as i32)
            })
            .collect();

        peak_confidences.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        if self.verbose {
            println!("\nPeaks sorted by confidence (greedy assignment order):");
        }

        // Greedy assignment with uniqueness for backbone peaks
        for (peak_idx, _, _) in peak_confidences {
            let peak = &self.peaks[peak_idx];
            let belief = &self.beliefs[peak_idx];

            // Find best available residue
            let best = belief.iter()
                .enumerate()
                .skip(1)  // Skip unassigned
                .filter(|(r, _)| {
                    // For backbone peaks, enforce uniqueness
                    if peak.peak_type == PeakType::Backbone {
                        !assigned_residues.contains(&(*r as i32))
                    } else {
                        true
                    }
                })
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap());

            if let Some((best_r, &best_prob)) = best {
                if best_prob > 0.01 {  // Threshold for assignment
                    if peak.peak_type == PeakType::Backbone {
                        assigned_residues.insert(best_r as i32);
                    }

                    let type_str = match peak.peak_type {
                        PeakType::Backbone => "BB",
                        PeakType::Carbon => "C ",
                    };
                    let res_type = self.residue_types.get(best_r - 1).map(|s| s.as_str()).unwrap_or("?");

                    if self.verbose {
                        println!("  Peak {:2} [{}] H={:.3} → pos {:2} ({}) conf={:.1}%",
                            peak_idx, type_str, peak.h_shift,
                            best_r, res_type, best_prob * 100.0);
                    }

                    results.push(UnifiedAssignmentResult {
                        peak_id: peak.id,
                        assigned_residue: best_r as i32,
                        confidence: best_prob,
                        peak_type: peak.peak_type,
                    });
                }
            }
        }

        if self.verbose {
            // Summary
            let bb_count = results.iter().filter(|r| r.peak_type == PeakType::Backbone).count();
            let c_count = results.iter().filter(|r| r.peak_type == PeakType::Carbon).count();
            println!("\n--- ASSIGNMENT SUMMARY ---");
            println!("  Backbone assignments: {}/{}", bb_count, self.sequence.len());
            println!("  Carbon assignments: {}", c_count);
            println!("═══════════════════════════════════════════════════════════════════\n");
        }

        results
    }

    /// Get backbone assignments only (for spin system compatibility).
    pub fn get_backbone_assignments(&self) -> Vec<(Uuid, i32, f64)> {
        self.extract_assignments()
            .into_iter()
            .filter(|r| r.peak_type == PeakType::Backbone)
            .map(|r| (r.peak_id, r.assigned_residue, r.confidence))
            .collect()
    }

    /// Debug: print beliefs for all peaks.
    pub fn print_beliefs(&self) {
        for (idx, peak) in self.peaks.iter().enumerate() {
            let belief = &self.beliefs[idx];
            let (best_r, &best_prob) = belief.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap();

            let type_str = match peak.peak_type {
                PeakType::Backbone => "BB",
                PeakType::Carbon => "C ",
            };

            println!("Peak {} {} H={:.3}: best={} ({:.2}) heavy={:?}",
                idx, type_str, peak.h_shift, best_r, best_prob,
                peak.heavy_shift.map(|v| format!("{:.1}", v)));
        }
    }
}

/// Build TOCSY correlations between peaks.
/// Returns (peak_i, peak_j, proton_diff_ppm) - raw proton difference for adaptive strength calculation.
fn build_tocsy_correlations(
    peaks: &[ObservedPeak],
    tocsy: &[UnlabeledPeak],
    params: &UnifiedAssignmentParams,
) -> Vec<(usize, usize, f64)> {
    let mut correlations = Vec::new();

    // Build lookup: discretized H shift -> TOCSY connections (actual ppm values)
    let discretize = |h: f64| (h * 1000.0).round() as i32;
    // Use initial (loose) tolerance for FINDING potential correlations
    let tolerance_bins = (params.h_tolerance_initial * 1000.0).round() as i32;

    // Store TOCSY connections with actual ppm values for precise matching
    let mut tocsy_connections: Vec<(f64, f64)> = Vec::new();
    for peak in tocsy {
        if peak.experiment_type != PeakExperimentType::Tocsy {
            continue;
        }
        let h1 = peak.position_ppm[0];
        let h2 = peak.position_ppm[1];

        // Skip diagonal
        if (h1 - h2).abs() <= params.h_tolerance_initial {
            continue;
        }

        tocsy_connections.push((h1, h2));
        tocsy_connections.push((h2, h1));  // Symmetric
    }

    // Find correlated peak pairs
    for i in 0..peaks.len() {
        let h_i = peaks[i].h_shift;

        for j in (i + 1)..peaks.len() {
            let h_j = peaks[j].h_shift;

            // Find best matching TOCSY connection
            let mut best_diff = f64::MAX;
            for &(tocsy_h1, tocsy_h2) in &tocsy_connections {
                // Check if this TOCSY links peaks i and j
                let diff1 = (tocsy_h1 - h_i).abs();
                let diff2 = (tocsy_h2 - h_j).abs();

                // Both dimensions must match within loose tolerance
                if diff1 <= params.h_tolerance_initial && diff2 <= params.h_tolerance_initial {
                    // Total mismatch for this TOCSY connection
                    let total_diff = diff1.max(diff2);  // Worst dimension determines quality
                    best_diff = best_diff.min(total_diff);
                }
            }

            if best_diff < f64::MAX {
                // Store raw proton difference (in ppm) for adaptive strength calculation
                correlations.push((i, j, best_diff));
            }
        }
    }

    tracing::debug!("Built {} TOCSY correlations between {} peaks", correlations.len(), peaks.len());
    correlations
}

/// Build HSQC-TOCSY correlations.
///
/// HSQC-TOCSY provides strong spin system evidence:
/// - All peaks in 15N-HSQC-TOCSY sharing the same N shift belong to same residue
/// - All peaks in 13C-HSQC-TOCSY sharing the same C shift belong to same residue
///
/// This is higher confidence than regular TOCSY since peaks are heavy-atom-anchored.
fn build_hsqc_tocsy_correlations(
    peaks: &[ObservedPeak],
    hsqc_tocsy_15n: &[UnlabeledPeak],
    hsqc_tocsy_13c: &[UnlabeledPeak],
    params: &UnifiedAssignmentParams,
) -> Vec<(usize, usize, f64)> {
    let mut correlations = Vec::new();

    // Tolerance for matching peaks to HSQC-TOCSY anchor shifts
    let h_tol = params.h_tolerance_initial;
    let n_tol = 0.5;  // N tolerance in ppm (15N has typical range ~15 ppm, so 0.5 is reasonable)
    let c_tol = 1.0;  // C tolerance in ppm (13C has larger range)

    // For 15N-HSQC-TOCSY: group peaks by N shift anchor
    // Peak format: (N_ppm, H_tocsy_ppm)
    if !hsqc_tocsy_15n.is_empty() {
        // Get backbone peaks and their N shifts
        let backbone_peaks: Vec<(usize, f64, f64)> = peaks.iter()
            .enumerate()
            .filter(|(_, p)| p.peak_type == PeakType::Backbone)
            .filter_map(|(i, p)| p.heavy_shift.map(|n| (i, n, p.h_shift)))
            .collect();

        // For each backbone peak, find HSQC-TOCSY peaks anchored at its N shift
        for &(bb_idx, bb_n, bb_h) in &backbone_peaks {
            let mut anchored_protons: Vec<f64> = Vec::new();

            for ht_peak in hsqc_tocsy_15n {
                if ht_peak.experiment_type != PeakExperimentType::HsqcTocsy15N {
                    continue;
                }
                let ht_n = ht_peak.position_ppm[0];
                let ht_h = ht_peak.position_ppm[1];

                // Check if this HSQC-TOCSY peak is anchored at this backbone's N
                if (ht_n - bb_n).abs() <= n_tol {
                    anchored_protons.push(ht_h);
                }
            }

            // Now find which CARBON peaks match these anchored protons
            // (Only correlate backbone with carbons, not backbone with backbone)
            for (other_idx, other_peak) in peaks.iter().enumerate() {
                if other_idx == bb_idx {
                    continue;
                }
                // Skip other backbone peaks - we only want backbone-to-carbon correlations
                if other_peak.peak_type == PeakType::Backbone {
                    continue;
                }

                // Check if this carbon peak's H shift matches any anchored proton
                for &anchored_h in &anchored_protons {
                    if (other_peak.h_shift - anchored_h).abs() <= h_tol {
                        // Found correlation: backbone and carbon peak belong to same residue
                        let quality = (other_peak.h_shift - anchored_h).abs();
                        correlations.push((bb_idx.min(other_idx), bb_idx.max(other_idx), quality));
                        break;
                    }
                }
            }
        }
    }

    // For 13C-HSQC-TOCSY: similar logic but for carbon peaks
    // Peak format: (C_ppm, H_tocsy_ppm)
    if !hsqc_tocsy_13c.is_empty() {
        // Get carbon peaks and their C shifts
        let carbon_peaks: Vec<(usize, f64, f64)> = peaks.iter()
            .enumerate()
            .filter(|(_, p)| p.peak_type == PeakType::Carbon)
            .filter_map(|(i, p)| p.heavy_shift.map(|c| (i, c, p.h_shift)))
            .collect();

        // For each carbon peak, find HSQC-TOCSY peaks anchored at its C shift
        for &(c_idx, c_shift, c_h) in &carbon_peaks {
            let mut anchored_protons: Vec<f64> = Vec::new();

            for ht_peak in hsqc_tocsy_13c {
                if ht_peak.experiment_type != PeakExperimentType::HsqcTocsy13C {
                    continue;
                }
                let ht_c = ht_peak.position_ppm[0];
                let ht_h = ht_peak.position_ppm[1];

                // Check if this HSQC-TOCSY peak is anchored at this carbon's C shift
                if (ht_c - c_shift).abs() <= c_tol {
                    anchored_protons.push(ht_h);
                }
            }

            // Now find which other peaks (backbone or carbon) match these anchored protons
            for (other_idx, other_peak) in peaks.iter().enumerate() {
                if other_idx == c_idx {
                    continue;
                }

                // Check if this peak's H shift matches any anchored proton
                for &anchored_h in &anchored_protons {
                    if (other_peak.h_shift - anchored_h).abs() <= h_tol {
                        // Found correlation: carbon and other peak belong to same residue
                        let quality = (other_peak.h_shift - anchored_h).abs();
                        correlations.push((c_idx.min(other_idx), c_idx.max(other_idx), quality));
                        break;
                    }
                }
            }
        }
    }

    // Remove duplicates (same pair might be found from both sides)
    correlations.sort_by(|a, b| {
        a.0.cmp(&b.0).then(a.1.cmp(&b.1))
    });
    correlations.dedup_by(|a, b| a.0 == b.0 && a.1 == b.1);

    tracing::debug!(
        "Built {} HSQC-TOCSY correlations from {} 15N-HSQC-TOCSY and {} 13C-HSQC-TOCSY peaks",
        correlations.len(),
        hsqc_tocsy_15n.len(),
        hsqc_tocsy_13c.len()
    );
    correlations
}

/// Build raw NOESY backbone-carbon correlations.
///
/// Stores (backbone_idx, carbon_idx, quality) where quality = proton mismatch.
/// Sequential relationships are computed DYNAMICALLY during BP using current beliefs.
/// This is the key insight: don't pre-compute which carbon belongs to which backbone,
/// let the beliefs evolve and weight NOESY evidence accordingly.
fn build_noesy_backbone_carbon(
    peaks: &[ObservedPeak],
    noesy: &[UnlabeledPeak],
    params: &UnifiedAssignmentParams,
) -> Vec<(usize, usize, f64)> {
    let mut correlations = Vec::new();
    // Use same tolerance as initial - adaptive BP will handle refinement
    let tolerance = params.h_tolerance_initial;

    // Get backbone and carbon peak indices
    let backbone_indices: Vec<usize> = peaks.iter()
        .enumerate()
        .filter(|(_, p)| p.peak_type == PeakType::Backbone)
        .map(|(i, _)| i)
        .collect();

    let carbon_indices: Vec<usize> = peaks.iter()
        .enumerate()
        .filter(|(_, p)| p.peak_type == PeakType::Carbon)
        .map(|(i, _)| i)
        .collect();

    if backbone_indices.is_empty() || carbon_indices.is_empty() {
        return correlations;
    }

    // For each NOESY cross-peak, find backbone-carbon pairs
    for noesy_peak in noesy {
        if noesy_peak.experiment_type != PeakExperimentType::Noesy { continue; }

        let h1 = noesy_peak.position_ppm[0];
        let h2 = noesy_peak.position_ppm[1];
        if (h1 - h2).abs() < 0.5 { continue; }  // Skip diagonal

        // One dimension should be backbone-like (7-10 ppm), other aliphatic (0-5 ppm)
        let (bb_h, c_h) = if h1 > 6.0 && h2 < 6.0 {
            (h1, h2)
        } else if h2 > 6.0 && h1 < 6.0 {
            (h2, h1)
        } else {
            continue;  // Not a backbone-aliphatic correlation
        };

        // Find matching backbone peak
        for &bb_idx in &backbone_indices {
            let bb_diff = (peaks[bb_idx].h_shift - bb_h).abs();
            if bb_diff > tolerance { continue; }

            // Find matching carbon peak
            for &c_idx in &carbon_indices {
                let c_diff = (peaks[c_idx].h_shift - c_h).abs();
                if c_diff > tolerance { continue; }

                // Store backbone-carbon correlation with quality
                let quality = bb_diff.max(c_diff);
                let corr = (bb_idx, c_idx, quality);
                if !correlations.iter().any(|(a, b, _)| *a == bb_idx && *b == c_idx) {
                    correlations.push(corr);
                }
            }
        }
    }

    correlations
}

/// Build proton profiles for each backbone peak from HSQC-TOCSY data.
///
/// For each backbone peak, finds all 15N-HSQC-TOCSY peaks with matching N shift,
/// extracts unique proton shifts, and characterizes the spin system by:
/// - Total proton count
/// - Presence of methyls (< 1.5 ppm)
/// - Presence of aromatics (6.5-8.0 ppm, excluding backbone amide)
///
/// IMPORTANT: Each HSQC-TOCSY peak is assigned to its CLOSEST backbone peak only,
/// preventing overlap when backbone N shifts are close together.
///
/// This enables amino acid typing based on spin system proton patterns.
fn build_proton_profiles(
    peaks: &[ObservedPeak],
    hsqc_tocsy_15n: &[UnlabeledPeak],
    _params: &UnifiedAssignmentParams,
) -> Vec<SpinSystemProtonProfile> {
    let mut profiles = Vec::new();

    if hsqc_tocsy_15n.is_empty() {
        return profiles;
    }

    // Get all backbone peaks with their N shifts
    let backbone_peaks: Vec<(usize, f64, f64)> = peaks.iter()
        .enumerate()
        .filter(|(_, p)| p.peak_type == PeakType::Backbone)
        .filter_map(|(i, p)| p.heavy_shift.map(|n| (i, n, p.h_shift)))
        .collect();

    if backbone_peaks.is_empty() {
        return profiles;
    }

    // For each backbone peak, collect protons assigned to it
    let mut bb_protons: HashMap<usize, Vec<f64>> = HashMap::new();
    for (bb_idx, _, _) in &backbone_peaks {
        bb_protons.insert(*bb_idx, Vec::new());
    }

    // Assign each HSQC-TOCSY peak to its CLOSEST backbone peak
    // This prevents overlap when backbone N shifts are close together
    let max_n_tolerance = 2.0;  // Maximum N difference to consider (ppm)

    for ht_peak in hsqc_tocsy_15n {
        if ht_peak.experiment_type != PeakExperimentType::HsqcTocsy15N {
            continue;
        }
        let ht_n = ht_peak.position_ppm[0];
        let ht_h = ht_peak.position_ppm[1];

        // Find the CLOSEST backbone peak
        let mut best_bb: Option<usize> = None;
        let mut best_diff = max_n_tolerance;

        for &(bb_idx, bb_n, _) in &backbone_peaks {
            let diff = (ht_n - bb_n).abs();
            if diff < best_diff {
                best_diff = diff;
                best_bb = Some(bb_idx);
            }
        }

        // Assign to closest backbone if within tolerance
        if let Some(bb_idx) = best_bb {
            bb_protons.get_mut(&bb_idx).unwrap().push(ht_h);
        }
    }

    // Build profiles from collected protons
    for (bb_idx, _bb_n, bb_h) in backbone_peaks {
        let proton_shifts = bb_protons.get(&bb_idx).unwrap();

        if proton_shifts.is_empty() {
            continue;
        }

        // Deduplicate proton shifts (using 0.05 ppm bins to catch near-degenerate protons)
        let mut sorted_protons = proton_shifts.clone();
        sorted_protons.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut unique_protons: Vec<f64> = Vec::new();
        for h in &sorted_protons {
            if unique_protons.is_empty() || (h - unique_protons.last().unwrap()).abs() > 0.05 {
                unique_protons.push(*h);
            }
        }

        // Characterize the spin system
        // Methyl: protons < 1.5 ppm (Ala, Val, Leu, Ile, Thr methyl groups)
        let has_methyl = unique_protons.iter().any(|&h| h < 1.5);

        // Aromatic: protons 6.5-8.0 ppm, but exclude backbone amide (which is at bb_h)
        // Backbone amide is typically 7-9 ppm, aromatic is 6.5-7.5 ppm for most
        let has_aromatic = unique_protons.iter().any(|&h| {
            h >= 6.5 && h <= 8.0 && (h - bb_h).abs() > 0.3  // Not the backbone amide
        });

        profiles.push(SpinSystemProtonProfile {
            backbone_idx: bb_idx,
            proton_count: unique_protons.len(),
            proton_shifts: unique_protons,
            has_methyl,
            has_aromatic,
        });
    }

    tracing::debug!(
        "Built {} proton profiles from {} 15N-HSQC-TOCSY peaks for {} backbone peaks",
        profiles.len(),
        hsqc_tocsy_15n.len(),
        peaks.iter().filter(|p| p.peak_type == PeakType::Backbone).count()
    );

    profiles
}

/// Get expected proton pattern for an amino acid type from KDE database.
///
/// Returns (unique_proton_count, has_methyl, has_aromatic) for scoring.
/// Uses the KDE database to determine:
/// - Proton count: number of unique H atoms in the database
/// - Methyl: any proton with mode < 1.5 ppm (HB for ALA, HG for VAL, etc.)
/// - Aromatic: any proton with mode in 6.5-8.0 ppm range (excluding backbone H)
fn get_expected_proton_pattern_kde(aa_type: &str, kde: &KDEDatabase) -> (usize, bool, bool) {
    let atoms = kde.get_residue_atoms(aa_type);

    // Filter to just protons (atoms starting with H)
    let protons: Vec<&String> = atoms.iter()
        .filter(|a| a.starts_with('H'))
        .collect();

    // Count unique proton types (not counting degeneracy like HG11/HG12/HG13)
    // We count base names like H, HA, HB, HG, HD, HE, HZ
    let mut base_protons: HashSet<String> = HashSet::new();
    let mut has_methyl = false;
    let mut has_aromatic = false;

    for proton in &protons {
        // Get the mode (most likely shift) for this proton
        if let Some(grid) = kde.get(aa_type, proton) {
            let mode = grid.mode();

            // Methyl check: shifts < 1.5 ppm typically methyl (ALA HB, VAL HG, LEU HD, etc.)
            if mode < 1.5 {
                has_methyl = true;
            }

            // Aromatic check: shifts 6.5-8.0 ppm, but not backbone H (which is 7-10 ppm typically)
            // Backbone H is named "H" exactly, aromatic protons are HD1, HD2, HE1, HE2, HZ, etc.
            if mode >= 6.5 && mode <= 8.0 && *proton != "H" {
                has_aromatic = true;
            }
        }

        // Extract base name (strip trailing numbers for degeneracy)
        // HA, HA2, HA3 -> HA
        // HG11, HG12, HG13, HG21, HG22, HG23 -> HG1, HG2 (but we want just HG)
        let base = proton.chars()
            .take_while(|c| !c.is_ascii_digit())
            .collect::<String>();
        if !base.is_empty() {
            base_protons.insert(base);
        }
    }

    // Proton count is number of unique base proton types
    let proton_count = base_protons.len();

    (proton_count, has_methyl, has_aromatic)
}

/// Score a proton shift against all sidechain proton atoms for an amino acid.
/// Returns the BEST (maximum) KDE density across all possible atom assignments.
/// This is the key discriminating function - sidechain protons have characteristic shifts.
fn score_proton_shift_for_residue(h_shift: f64, aa_type: &str, kde: &KDEDatabase) -> f64 {
    // Sidechain proton atoms to try (excludes backbone H and N)
    let sidechain_atoms = [
        "HA", "HA2", "HA3",
        "HB", "HB1", "HB2", "HB3",
        "HG", "HG1", "HG2", "HG3", "HG11", "HG12", "HG13", "HG21", "HG22", "HG23",
        "HD", "HD1", "HD2", "HD3", "HD11", "HD12", "HD13", "HD21", "HD22", "HD23",
        "HE", "HE1", "HE2", "HE3", "HE21", "HE22",
        "HZ", "HZ2", "HZ3",
        "HH", "HH11", "HH12", "HH21", "HH22",
    ];

    let mut best_density = 0.0;

    for atom in &sidechain_atoms {
        if let Some(density) = kde.get(aa_type, atom).map(|g| g.evaluate(h_shift)) {
            if density > best_density {
                best_density = density;
            }
        }
    }

    best_density
}

/// Fallback: hardcoded expected proton pattern (used if KDE unavailable).
#[allow(dead_code)]
fn get_expected_proton_pattern_fallback(aa_type: &str) -> (usize, bool, bool) {
    match aa_type {
        "GLY" => (2, false, false),
        "ALA" => (3, true, false),   // H, HA, HB
        "THR" => (4, true, false),   // H, HA, HB, HG2
        "VAL" => (4, true, false),   // H, HA, HB, HG
        "LEU" => (5, true, false),   // H, HA, HB, HG, HD
        "ILE" => (5, true, false),   // H, HA, HB, HG, HD
        "MET" => (5, true, false),   // H, HA, HB, HG, HE
        "SER" => (3, false, false),  // H, HA, HB
        "CYS" => (3, false, false),  // H, HA, HB
        "ASN" => (4, false, false),  // H, HA, HB, HD2
        "ASP" => (3, false, false),  // H, HA, HB
        "GLN" => (5, false, false),  // H, HA, HB, HG, HE2
        "GLU" => (4, false, false),  // H, HA, HB, HG
        "LYS" => (6, false, false),  // H, HA, HB, HG, HD, HE, HZ
        "ARG" => (6, false, false),  // H, HA, HB, HG, HD, HE
        "PHE" => (5, false, true),   // H, HA, HB, HD, HE, HZ
        "TYR" => (5, false, true),   // H, HA, HB, HD, HE
        "TRP" => (7, false, true),   // H, HA, HB, HD1, HE1, HE3, HZ2, HZ3, HH2
        "HIS" => (5, false, true),   // H, HA, HB, HD2, HE1
        "PRO" => (4, false, false),  // HA, HB, HG, HD
        _ => (4, false, false),
    }
}

/// Compute carbon typing scores: how well does each carbon peak match each residue type?
fn compute_carbon_typing_scores(
    peaks: &[ObservedPeak],
    residue_types: &[String],
    _bmrb: &BMRBDatabase,
    _params: &UnifiedAssignmentParams,
) -> Vec<Vec<f64>> {
    let domain_size = residue_types.len() + 1;  // +1 for "unassigned"
    let kde = KDEDatabase::load_embedded();  // Use KDE for scoring

    peaks.iter().map(|peak| {
        let mut scores = vec![0.1; domain_size];  // Small prior for unassigned

        if peak.peak_type == PeakType::Carbon {
            if let Some(c_shift) = peak.heavy_shift {
                let h_shift = peak.h_shift;  // Attached proton shift

                // Score C-H pair jointly against each residue type
                for (r, res_type) in residue_types.iter().enumerate() {
                    let score = score_carbon_ch_pair(c_shift, h_shift, res_type, &kde);
                    scores[r + 1] = score;  // r+1 because 0 is "unassigned"
                }
            }
        } else {
            // Backbone peaks get uniform scores (typing comes from linked carbons)
            for r in 1..domain_size {
                scores[r] = 1.0 / (domain_size - 1) as f64;
            }
        }

        scores
    }).collect()
}

/// C-H attachment mapping: which protons are directly bonded to which carbons.
/// For each carbon atom, lists the possible attached proton names (varies by AA type).
const CH_ATTACHMENTS: &[(&str, &[&str])] = &[
    ("CA", &["HA", "HA2", "HA3"]),  // GLY has HA2/HA3 instead of single HA
    ("CB", &["HB", "HB2", "HB3"]),
    ("CG", &["HG", "HG2", "HG3"]),
    ("CG1", &["HG11", "HG12", "HG13"]),
    ("CG2", &["HG21", "HG22", "HG23"]),
    ("CD", &["HD2", "HD3"]),
    ("CD1", &["HD1", "HD11", "HD12", "HD13"]),
    ("CD2", &["HD21", "HD22", "HD23"]),
    ("CE", &["HE", "HE2", "HE3"]),
    ("CE1", &["HE1"]),
    ("CE2", &["HE2"]),
    ("CE3", &["HE3"]),
    ("CZ", &["HZ"]),
    ("CZ2", &["HZ2"]),
    ("CZ3", &["HZ3"]),
    ("CH2", &["HH2"]),
];

/// Score a C-H pair jointly using KDE densities for both shifts.
/// Uses P(C, H | AA, atom) = P(C | AA, carbon_atom) × P(H | AA, proton_atom)
/// Returns the best score across all possible C-H atom pair assignments.
fn score_carbon_ch_pair(c_shift: f64, h_shift: f64, res_type: &str, kde: &KDEDatabase) -> f64 {
    let mut best_score = 1e-10;

    // Try each C-H attachment to find the best match
    for (c_atom, h_atoms) in CH_ATTACHMENTS {
        // Get carbon density
        let c_density = kde.density(res_type, c_atom, c_shift);
        if c_density <= 0.0 {
            continue;
        }

        // Find best matching attached proton
        let mut best_h_density = 0.0;
        for h_atom in *h_atoms {
            let h_density = kde.density(res_type, h_atom, h_shift);
            if h_density > best_h_density {
                best_h_density = h_density;
            }
        }

        if best_h_density > 0.0 {
            // Joint probability = P(C) × P(H)
            // Take geometric mean to normalize scale (so units match single-shift scoring)
            let joint_score = (c_density * best_h_density).sqrt();

            if joint_score > best_score {
                best_score = joint_score;
            }
        }
    }

    // If no C-H pair matched, fall back to C-only scoring (for safety)
    if best_score < 1e-9 {
        // Fall back to carbon-only scoring
        for (c_atom, _) in CH_ATTACHMENTS {
            let c_density = kde.density(res_type, c_atom, c_shift);
            if c_density > best_score {
                best_score = c_density;
            }
        }
    }

    best_score
}

/// Score how well a carbon shift matches a residue type.
fn score_carbon_for_residue(
    c_shift: f64,
    res_type: &str,
    bmrb: &BMRBDatabase,
    params: &UnifiedAssignmentParams,
) -> f64 {
    score_carbon_for_residue_adaptive(c_shift, res_type, bmrb, params.c_tolerance_initial)
}

/// Score how well a carbon shift matches a residue type with adaptive tolerance.
fn score_carbon_for_residue_adaptive(
    c_shift: f64,
    res_type: &str,
    _bmrb: &BMRBDatabase,
    _c_tolerance: f64,
) -> f64 {
    // Use KDE scoring for better accuracy with multi-modal distributions
    let scorer = KDEScorer::new();
    score_carbon_for_residue_kde(c_shift, res_type, &scorer)
}

/// Score carbon shift using KDE scorer (preferred method).
fn score_carbon_for_residue_kde(
    c_shift: f64,
    res_type: &str,
    scorer: &dyn ShiftScorer,
) -> f64 {
    let mut max_score: f64 = 0.01;  // Minimum score

    // Check all carbon atoms for this residue type
    for atom in &["CA", "CB", "CG", "CG1", "CG2", "CD", "CD1", "CD2", "CE", "CE1", "CE2", "CZ"] {
        let score = scorer.score(res_type, atom, c_shift);
        if score > 0.0 {
            max_score = max_score.max(score);
        }
    }

    max_score
}

/// Convert one-letter amino acid code to three-letter code.
fn one_letter_to_three(c: &char) -> String {
    match c {
        'A' => "ALA", 'C' => "CYS", 'D' => "ASP", 'E' => "GLU", 'F' => "PHE",
        'G' => "GLY", 'H' => "HIS", 'I' => "ILE", 'K' => "LYS", 'L' => "LEU",
        'M' => "MET", 'N' => "ASN", 'P' => "PRO", 'Q' => "GLN", 'R' => "ARG",
        'S' => "SER", 'T' => "THR", 'V' => "VAL", 'W' => "TRP", 'Y' => "TYR",
        _ => "UNK",
    }.to_string()
}

// =============================================================================
// NEW: Observation-Based Assignment (Unified Model)
// =============================================================================

/// Result from observation-based assignment.
#[derive(Debug, Clone)]
pub struct ObservationAssignmentResult {
    pub observation_id: Uuid,
    pub assigned_residue: i32,  // 0 = unassigned, 1..N = residue position
    pub confidence: f64,
    pub experiment_type: PeakExperimentType,
}

/// Run assignment on unified Observations (new model).
///
/// This treats ALL experiment types equally - TOCSY, NOESY, HSQC are all
/// first-class observations that can be assigned to residues.
pub fn run_observation_assignment(
    observations: &[Observation],
    sequence: &str,
    params: &UnifiedAssignmentParams,
    tol_params: &NucleusToleranceParams,
) -> Vec<ObservationAssignmentResult> {
    if observations.is_empty() {
        return vec![];
    }

    let residue_types: Vec<String> = sequence.chars()
        .map(|c| one_letter_to_three(&c))
        .collect();
    let domain_size = sequence.len() + 1;  // +1 for unassigned (index 0)

    // Load KDE database for typing
    let kde = KDEDatabase::load_embedded();

    // Initialize beliefs uniformly
    let mut beliefs: Vec<Vec<f64>> = observations.iter()
        .map(|_| vec![1.0 / domain_size as f64; domain_size])
        .collect();

    // Run belief propagation
    let max_iterations = params.max_iterations;
    for iteration in 0..max_iterations {
        let progress = iteration as f64 / max_iterations as f64;
        let interp = params.interpolate(progress);

        // Compute typing scores (observation -> residue type)
        let typing_scores = compute_observation_typing_scores(
            observations, &residue_types, &kde, tol_params, iteration, max_iterations
        );

        // Compute correlation scores (observation <-> observation with matching shifts)
        let correlation_scores = compute_observation_correlations(
            observations, tol_params, iteration, max_iterations
        );

        // Compute sequential relationships from triple-resonance carbon matching
        let sequential_links = compute_sequential_links(
            observations, tol_params, iteration, max_iterations
        );

        // Message passing update with both same-residue and sequential factors
        let new_beliefs = update_observation_beliefs_with_sequential(
            &beliefs, &typing_scores, &correlation_scores, &sequential_links,
            domain_size, interp.tocsy_weight, interp.typing_weight, interp.sequential_weight
        );

        // Apply damping to prevent oscillation on loopy graph
        let damping = params.damping;
        for i in 0..beliefs.len() {
            for d in 0..beliefs[i].len() {
                beliefs[i][d] = damping * beliefs[i][d] + (1.0 - damping) * new_beliefs[i][d];
            }
        }

        // Check convergence (after damping)
        let max_delta = beliefs.iter().zip(new_beliefs.iter())
            .flat_map(|(old, new)| old.iter().zip(new.iter()).map(|(a, b)| (a - b).abs()))
            .fold(0.0f64, |acc, x| acc.max(x));

        if params.verbose && iteration % 10 == 0 {
            let h_tol = tol_params.tolerance_for(NucleusType::H1, iteration, max_iterations);
            let c_tol = tol_params.tolerance_for(NucleusType::C13, iteration, max_iterations);
            println!("Iteration {}: H_tol={:.4}, C_tol={:.2}, max_delta={:.6}",
                     iteration, h_tol, c_tol, max_delta);
        }

        if max_delta < params.convergence_threshold {
            if params.verbose {
                println!("Converged at iteration {}", iteration);
            }
            break;
        }
    }

    // Extract assignments
    observations.iter().zip(beliefs.iter()).map(|(obs, belief)| {
        let (best_idx, &best_prob) = belief.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap_or((0, &0.0));

        ObservationAssignmentResult {
            observation_id: obs.id,
            assigned_residue: best_idx as i32,
            confidence: best_prob,
            experiment_type: obs.experiment_type,
        }
    }).collect()
}

/// Compute typing scores for observations based on chemical shifts.
fn compute_observation_typing_scores(
    observations: &[Observation],
    residue_types: &[String],
    kde: &KDEDatabase,
    tol_params: &NucleusToleranceParams,
    iteration: usize,
    max_iterations: usize,
) -> Vec<Vec<f64>> {
    observations.iter().map(|obs| {
        let domain_size = residue_types.len() + 1;

        // Unassigned base score scales with dimensionality - 3D observations naturally have
        // lower joint probabilities (product of 3 densities), so we scale the null hypothesis
        // accordingly: 0.01 for 2D, 0.001 for 3D, etc.
        let n_dims = obs.dimensions.len();
        let unassigned_base = 0.1_f64.powi(n_dims as i32);

        let mut scores = vec![unassigned_base; domain_size];

        // Score against each residue type
        for (r, res_type) in residue_types.iter().enumerate() {
            let score = score_observation_for_residue_type(
                obs, res_type, kde, tol_params, iteration, max_iterations
            );
            scores[r + 1] = score.max(1e-10);
        }

        // Normalize
        let sum: f64 = scores.iter().sum();
        if sum > 0.0 {
            for s in &mut scores {
                *s /= sum;
            }
        }
        scores
    }).collect()
}

/// Score how well an observation matches a residue type.
///
/// Physics-based: uses atom_constraint instead of nucleus-based guessing.
/// NOTE: For now, we use ALL dimensions for typing, including inter-residue carbons.
/// A more sophisticated model would need to handle that inter-residue observations
/// provide evidence about BOTH the anchor residue AND the preceding residue.
fn score_observation_for_residue_type(
    obs: &Observation,
    res_type: &str,
    kde: &KDEDatabase,
    _tol_params: &NucleusToleranceParams,
    _iteration: usize,
    _max_iterations: usize,
) -> f64 {
    let mut log_score = 0.0;
    let debug_this = false;  // Disable verbose debug

    for dim in &obs.dimensions {
        // Physics-based: use atom_constraint instead of nucleus-based guessing
        let atom_candidates = atoms_from_constraint(&dim.atom_constraint, dim.nucleus);
        let mut best_density = 1e-10;
        let mut best_atom = "";

        for atom in &atom_candidates {
            let density = kde.density(res_type, atom, dim.shift);
            if density > best_density {
                best_density = density;
                best_atom = atom;
            }
        }

        if debug_this {
            println!("  SCORE: {} {:?} constraint={:?} shift={:.2} -> atoms={:?}, best={}:{:.2e}",
                     res_type, dim.nucleus, dim.atom_constraint, dim.shift, atom_candidates, best_atom, best_density);
        }

        log_score += best_density.max(1e-10).ln();
    }

    let score = log_score.exp();
    if debug_this {
        println!("  SCORE: {} log={:.4} -> exp={:.2e}", res_type, log_score, score);
    }

    score
}

/// Convert AtomConstraint to atom candidates for KDE scoring.
fn atoms_from_constraint(constraint: &crate::data::spin_system::AtomConstraint, nucleus: NucleusType) -> Vec<&'static str> {
    use crate::data::spin_system::AtomConstraint;

    match constraint {
        AtomConstraint::Exact(atom) => {
            // Map the atom name to a static str we can return
            match atom.as_str() {
                "H" | "HN" => vec!["H"],
                "N" => vec!["N"],
                "CA" => vec!["CA"],
                "CB" => vec!["CB"],
                "HA" => vec!["HA", "HA2", "HA3"],
                "C" => vec!["C"],
                _ => atoms_for_nucleus(nucleus),
            }
        }
        AtomConstraint::OneOf(atoms) => {
            // Convert the Vec<String> to static atom names
            let mut result = Vec::new();
            for atom in atoms {
                match atom.as_str() {
                    "CA" => result.push("CA"),
                    "CB" => result.push("CB"),
                    "H" | "HN" => result.push("H"),
                    "HA" => {
                        result.push("HA");
                        result.push("HA2");
                        result.push("HA3");
                    }
                    "N" => result.push("N"),
                    "C" => result.push("C"),
                    _ => {}
                }
            }
            if result.is_empty() {
                atoms_for_nucleus(nucleus)
            } else {
                result
            }
        }
        AtomConstraint::Any => atoms_for_nucleus(nucleus),
    }
}

/// Get candidate atoms for a nucleus type and optional atom hint.
fn atoms_for_nucleus_and_hint(nucleus: NucleusType, atom_hint: &Option<String>) -> Vec<&'static str> {
    if let Some(hint) = atom_hint {
        // Use the hint if provided
        match hint.as_str() {
            "HN" | "H" => vec!["H"],
            "HA" => vec!["HA", "HA2", "HA3"],
            "N" => vec!["N"],
            "CA" => vec!["CA"],
            "CB" => vec!["CB"],
            _ => atoms_for_nucleus(nucleus),
        }
    } else {
        atoms_for_nucleus(nucleus)
    }
}

/// Get all candidate atoms for a nucleus type.
fn atoms_for_nucleus(nucleus: NucleusType) -> Vec<&'static str> {
    match nucleus {
        NucleusType::H1 => vec![
            "H", "HA", "HA2", "HA3",
            "HB", "HB2", "HB3",
            "HG", "HG2", "HG3", "HG11", "HG12", "HG13", "HG21", "HG22", "HG23",
            "HD1", "HD2", "HD21", "HD22", "HD3",
            "HE", "HE1", "HE2", "HE21", "HE22", "HE3",
            "HZ", "HZ2", "HZ3", "HH", "HH11", "HH12", "HH21", "HH22"
        ],
        NucleusType::C13 => vec![
            "CA", "CB", "CG", "CG1", "CG2",
            "CD", "CD1", "CD2", "CE", "CE1", "CE2", "CE3",
            "CZ", "CZ2", "CZ3", "CH2", "C"
        ],
        NucleusType::N15 => vec!["N", "ND1", "ND2", "NE", "NE1", "NE2", "NZ", "NH1", "NH2"],
        _ => vec![],
    }
}

/// Compute correlation scores between observations based on chemical shift matching.
fn compute_observation_correlations(
    observations: &[Observation],
    tol_params: &NucleusToleranceParams,
    iteration: usize,
    max_iterations: usize,
) -> Vec<Vec<f64>> {
    let n = observations.len();
    let mut correlations = vec![vec![0.0; n]; n];
    let mut nonzero_count = 0;

    for i in 0..n {
        correlations[i][i] = 1.0;  // Self-correlation

        for j in (i + 1)..n {
            let score = compute_observation_pair_correlation(
                &observations[i], &observations[j],
                tol_params, iteration, max_iterations
            );
            correlations[i][j] = score;
            correlations[j][i] = score;
            if score > 0.01 {
                nonzero_count += 1;
            }
        }
    }

    // Debug: correlation discovery stats (verbose mode only, see main output)
    // Keeping for debugging if needed:
    // if iteration == 0 {
    //     println!("Correlation matrix: {} observations, {} nonzero correlations (of {} possible pairs)",
    //              n, nonzero_count, n * (n - 1) / 2);
    // }

    correlations
}

/// Compute correlation between two observations based on magnetization transfer PHYSICS.
///
/// This function uses physics-based fields (transfer_pathway, residue_offset) instead of
/// experiment_type to determine correlations. This allows the same logic to handle any
/// experiment that produces observations with the same physics properties.
///
/// Correlation semantics by transfer pathway:
/// - DirectBond: Nuclei are directly bonded → matching shifts means same residue
/// - ThroughBond: Nuclei are J-coupled → matching shifts means same spin system (residue)
/// - BackboneSequential: Check residue_offset to determine if observing same residue
/// - ThroughSpace: Distance-dependent, handled separately (no direct correlation here)
fn compute_observation_pair_correlation(
    obs_a: &Observation,
    obs_b: &Observation,
    tol_params: &NucleusToleranceParams,
    iteration: usize,
    max_iterations: usize,
) -> f64 {
    let h_tol = tol_params.tolerance_for(NucleusType::H1, iteration, max_iterations);
    let n_tol = tol_params.tolerance_for(NucleusType::N15, iteration, max_iterations);
    let c_tol = tol_params.tolerance_for(NucleusType::C13, iteration, max_iterations);

    // Helper to check if a shift matches within tolerance
    let shifts_match = |s1: f64, s2: f64, tol: f64| -> Option<f64> {
        let diff = (s1 - s2).abs();
        if diff < tol {
            Some((-0.5 * (diff / tol).powi(2)).exp())
        } else {
            None
        }
    };

    // Helper functions (defined inline to avoid lifetime issues)

    // Get backbone anchor (H, N) shifts from Intra dimensions
    fn get_backbone_anchor(obs: &Observation) -> Option<(f64, f64)> {
        let h = obs.dimensions.iter()
            .find(|d| d.nucleus == NucleusType::H1 && d.residue_offset == ResidueOffset::Intra)
            .map(|d| d.shift)?;
        let n = obs.dimensions.iter()
            .find(|d| d.nucleus == NucleusType::N15 && d.residue_offset == ResidueOffset::Intra)
            .map(|d| d.shift)?;
        Some((h, n))
    }

    // Get heavy atom shift (first N15 or C13)
    fn get_heavy_shift(obs: &Observation) -> Option<(NucleusType, f64)> {
        obs.dimensions.iter()
            .find(|d| d.nucleus == NucleusType::N15 || d.nucleus == NucleusType::C13)
            .map(|d| (d.nucleus, d.shift))
    }

    // Check if carbon dimension is intra-residue
    fn carbon_is_intra(obs: &Observation) -> bool {
        obs.dimensions.iter()
            .find(|d| d.nucleus == NucleusType::C13)
            .map(|d| d.residue_offset == ResidueOffset::Intra)
            .unwrap_or(true)
    }

    // Get proton shifts for an observation
    fn get_proton_shifts(obs: &Observation) -> Vec<f64> {
        obs.dimensions.iter()
            .filter(|d| d.nucleus == NucleusType::H1)
            .map(|d| d.shift)
            .collect()
    }

    // Get first proton shift
    fn get_first_proton(obs: &Observation) -> Option<f64> {
        obs.dimensions.iter()
            .find(|d| d.nucleus == NucleusType::H1)
            .map(|d| d.shift)
    }

    // Check if observation has N15
    fn has_n15(obs: &Observation) -> bool {
        obs.dimensions.iter().any(|d| d.nucleus == NucleusType::N15)
    }

    // Get N15 shift
    fn get_n15_shift(obs: &Observation) -> Option<f64> {
        obs.dimensions.iter()
            .find(|d| d.nucleus == NucleusType::N15)
            .map(|d| d.shift)
    }

    // === PHYSICS-BASED CORRELATION LOGIC ===

    // CASE 1: Both DirectBond (HSQC-type experiments)
    // Same anchor shift → same residue
    if obs_a.transfer_pathway == TransferPathway::DirectBond
       && obs_b.transfer_pathway == TransferPathway::DirectBond
    {
        // Get proton and heavy atom shifts
        if let (Some(ha_h), Some(hb_h)) = (get_first_proton(obs_a), get_first_proton(obs_b)) {
            if let Some(h_score) = shifts_match(ha_h, hb_h, h_tol) {
                // Check heavy atom (N15 or C13)
                let da_heavy = get_heavy_shift(obs_a);
                let db_heavy = get_heavy_shift(obs_b);

                if let (Some((nuc_a, shift_a)), Some((nuc_b, shift_b))) = (da_heavy, db_heavy) {
                    if nuc_a == nuc_b {
                        let tol = if nuc_a == NucleusType::N15 { n_tol } else { c_tol };
                        if let Some(heavy_score) = shifts_match(shift_a, shift_b, tol) {
                            return (h_score + heavy_score) / 2.0;
                        }
                    }
                }
            }
        }
        return 0.0;
    }

    // CASE 2: Both ThroughBond (TOCSY-type experiments)
    // Require BOTH protons to match (strict spin system link)
    if obs_a.transfer_pathway == TransferPathway::ThroughBond
       && obs_b.transfer_pathway == TransferPathway::ThroughBond
    {
        let ha_dims = get_proton_shifts(obs_a);
        let hb_dims = get_proton_shifts(obs_b);

        if ha_dims.len() >= 2 && hb_dims.len() >= 2 {
            // Check both orientations
            let match_direct = shifts_match(ha_dims[0], hb_dims[0], h_tol)
                .and_then(|s1| shifts_match(ha_dims[1], hb_dims[1], h_tol).map(|s2| (s1 + s2) / 2.0));
            let match_flipped = shifts_match(ha_dims[0], hb_dims[1], h_tol)
                .and_then(|s1| shifts_match(ha_dims[1], hb_dims[0], h_tol).map(|s2| (s1 + s2) / 2.0));

            if let Some(score) = match_direct.or(match_flipped) {
                return score;
            }
        }
        return 0.0;
    }

    // CASE 3: Both BackboneSequential (triple-resonance experiments)
    // Check if same backbone anchor AND same residue_offset on heavy atoms
    if obs_a.transfer_pathway == TransferPathway::BackboneSequential
       && obs_b.transfer_pathway == TransferPathway::BackboneSequential
    {
        if let (Some((ha, na)), Some((hb, nb))) = (get_backbone_anchor(obs_a), get_backbone_anchor(obs_b)) {
            let h_match = shifts_match(ha, hb, h_tol);
            let n_match = shifts_match(na, nb, n_tol);

            if h_match.is_some() && n_match.is_some() {
                // SAME backbone anchor - but do they observe the same residue?
                // Use residue_offset to determine this (physics-based!)
                let a_intra = carbon_is_intra(obs_a);
                let b_intra = carbon_is_intra(obs_b);

                if a_intra == b_intra {
                    // Both observe the same residue → correlate
                    let backbone_score = (h_match.unwrap() + n_match.unwrap()) / 2.0;
                    return backbone_score;
                } else {
                    // One is intra, one is inter → different residues
                    return 0.0;
                }
            }
        }
        return 0.0;
    }

    // CASE 4: DirectBond + BackboneSequential
    // HSQC correlates with triple-res if anchor matches AND triple-res carbon is INTRA
    if (obs_a.transfer_pathway == TransferPathway::DirectBond && obs_b.transfer_pathway == TransferPathway::BackboneSequential)
       || (obs_a.transfer_pathway == TransferPathway::BackboneSequential && obs_b.transfer_pathway == TransferPathway::DirectBond)
    {
        let (direct_obs, sequential_obs) = if obs_a.transfer_pathway == TransferPathway::DirectBond {
            (obs_a, obs_b)
        } else {
            (obs_b, obs_a)
        };

        // Check if direct observation has N15 (15N-HSQC)
        if !has_n15(direct_obs) { return 0.0; }

        // Get backbone anchor from both
        let direct_h = get_first_proton(direct_obs);
        let direct_n = get_n15_shift(direct_obs);

        if let (Some((seq_h, seq_n)), Some(dh), Some(dn)) =
               (get_backbone_anchor(sequential_obs), direct_h, direct_n)
        {
            if let (Some(h_score), Some(n_score)) = (shifts_match(dh, seq_h, h_tol), shifts_match(dn, seq_n, n_tol)) {
                // Check if sequential observation's carbon is INTRA (same residue as HSQC)
                if carbon_is_intra(sequential_obs) {
                    return (h_score + n_score) / 2.0;
                }
            }
        }
        return 0.0;
    }

    // CASE 5: DirectBond + ThroughBond
    // HSQC↔TOCSY correlation is weak (single proton match is too loose)
    // Disabled to prevent spurious correlations
    if (obs_a.transfer_pathway == TransferPathway::DirectBond && obs_b.transfer_pathway == TransferPathway::ThroughBond)
       || (obs_a.transfer_pathway == TransferPathway::ThroughBond && obs_b.transfer_pathway == TransferPathway::DirectBond)
    {
        return 0.0;
    }

    // CASE 6: ThroughSpace (NOESY) - complex distance-dependent correlations
    // Sequential correlations from NOESY are handled in compute_sequential_links
    if obs_a.transfer_pathway == TransferPathway::ThroughSpace
       || obs_b.transfer_pathway == TransferPathway::ThroughSpace
    {
        return 0.0;
    }

    // Default: no correlation for unhandled pathway combinations
    0.0
}

/// Update beliefs using typing and correlation factors.
fn update_observation_beliefs(
    beliefs: &[Vec<f64>],
    typing_scores: &[Vec<f64>],
    correlation_scores: &[Vec<f64>],
    domain_size: usize,
    correlation_weight: f64,
    typing_weight: f64,
) -> Vec<Vec<f64>> {
    let n = beliefs.len();
    let mut new_beliefs = vec![vec![0.0; domain_size]; n];

    for i in 0..n {
        // Start with typing prior
        for d in 0..domain_size {
            new_beliefs[i][d] = typing_scores[i][d].ln() * typing_weight;
        }

        // Add correlation messages from other observations
        for j in 0..n {
            if i == j { continue; }

            let corr = correlation_scores[i][j];
            if corr > 0.01 {  // Threshold to avoid noise
                // Observations with high correlation should agree on residue
                for d in 0..domain_size {
                    new_beliefs[i][d] += beliefs[j][d].max(1e-10).ln() * corr * correlation_weight;
                }
            }
        }

        // Convert from log to probability and normalize
        let max_val = new_beliefs[i].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let sum: f64 = new_beliefs[i].iter_mut()
            .map(|v| { *v = (*v - max_val).exp(); *v })
            .sum();

        if sum > 0.0 {
            for v in &mut new_beliefs[i] {
                *v /= sum;
            }
        }
    }

    new_beliefs
}

/// Sequential link between two observations indicating they are from adjacent residues.
/// If obs_from is at position P, then obs_to should be at position P+1.
#[derive(Debug, Clone)]
struct SequentialLink {
    from_idx: usize,  // Observation index that is at position i
    to_idx: usize,    // Observation index that is at position i+1
    strength: f64,    // Carbon match quality (0-1)
}

/// Compute sequential links from backbone-sequential transfer pathway observations.
///
/// Physics-based approach: uses TransferPathway and ResidueOffset instead of experiment types.
///
/// Observations with BackboneSequential transfer pathway show carbon correlations to backbone NH.
/// The carbon dimension has a ResidueOffset indicating whether it observes the anchor residue (Intra)
/// or the preceding residue (PrecedingResidue).
///
/// When carbon shifts match across different backbone anchors:
/// - Intra@backbone_A + PrecedingResidue@backbone_B → both see same residue → same-residue link
/// - PrecedingResidue@backbone_A + Intra@backbone_B → both see same residue → same-residue link
/// - Both Intra or both PrecedingResidue → different residues → no link
fn compute_sequential_links(
    observations: &[Observation],
    tol_params: &NucleusToleranceParams,
    iteration: usize,
    max_iterations: usize,
) -> Vec<SequentialLink> {
    use crate::data::spin_system::{TransferPathway, ResidueOffset};

    let c_tol = tol_params.tolerance_for(NucleusType::C13, iteration, max_iterations);
    let h_tol = tol_params.tolerance_for(NucleusType::H1, iteration, max_iterations);
    let n_tol = tol_params.tolerance_for(NucleusType::N15, iteration, max_iterations);

    let mut links = Vec::new();

    // Helper to get backbone anchor (H, N) shifts
    fn get_backbone(obs: &Observation) -> Option<(f64, f64)> {
        let h = obs.dimensions.iter()
            .find(|d| d.nucleus == NucleusType::H1)
            .map(|d| d.shift)?;
        let n = obs.dimensions.iter()
            .find(|d| d.nucleus == NucleusType::N15)
            .map(|d| d.shift)?;
        Some((h, n))
    }

    // Helper to get carbon shift and its residue offset (physics-based!)
    fn get_carbon_with_offset(obs: &Observation) -> Option<(f64, ResidueOffset)> {
        obs.dimensions.iter()
            .find(|d| d.nucleus == NucleusType::C13)
            .map(|d| (d.shift, d.residue_offset))
    }

    // Helper to check if two backbones are different (not the same NH)
    let different_backbone = |h1: f64, n1: f64, h2: f64, n2: f64| -> bool {
        (h1 - h2).abs() > h_tol || (n1 - n2).abs() > n_tol
    };

    // Find all backbone-sequential observations (HNCA, HNCACB, CBCACONH, etc.)
    // Physics-based: filter by transfer pathway, NOT experiment type!
    let sequential_obs: Vec<(usize, &Observation)> = observations.iter()
        .enumerate()
        .filter(|(_, obs)| obs.transfer_pathway == TransferPathway::BackboneSequential)
        .collect();

    // For each pair of backbone-sequential observations at DIFFERENT backbones
    for (i, (idx_a, obs_a)) in sequential_obs.iter().enumerate() {
        let Some((h_a, n_a)) = get_backbone(obs_a) else { continue };
        let Some((c_a, offset_a)) = get_carbon_with_offset(obs_a) else { continue };

        for (idx_b, obs_b) in sequential_obs.iter().skip(i + 1) {
            let Some((h_b, n_b)) = get_backbone(obs_b) else { continue };
            let Some((c_b, offset_b)) = get_carbon_with_offset(obs_b) else { continue };

            // Skip if same backbone anchor
            if !different_backbone(h_a, n_a, h_b, n_b) {
                continue;
            }

            // Check for carbon shift match
            let c_diff = (c_a - c_b).abs();
            if c_diff >= c_tol {
                continue;  // No carbon match
            }

            // Carbon match found at DIFFERENT backbones!
            // Use physics-based residue_offset instead of experiment-type dispatch:
            //   - Intra@backbone_X sees residue X
            //   - PrecedingResidue@backbone_Y sees residue Y-1
            // If carbons match: the residues they observe are the SAME
            //   - If A Intra@X and B PrecedingResidue@Y match: residue X = residue Y-1
            //     → A and B observe the SAME residue (NOT sequential!)
            //     → We learn backbone X+1 = backbone Y (backbone ordering)
            //   - If A PrecedingResidue@X and B Intra@Y match: residue X-1 = residue Y
            //     → A and B observe the SAME residue
            //     → We learn backbone X = backbone Y+1
            //
            // So cross-backbone intra/inter carbon matches are SAME-RESIDUE correlations,
            // not sequential relationships!

            let match_strength = (-0.5 * (c_diff / c_tol).powi(2)).exp();

            let is_intra_a = offset_a == ResidueOffset::Intra;
            let is_intra_b = offset_b == ResidueOffset::Intra;

            if is_intra_a && !is_intra_b {
                // A shows Intra (residue at backbone A), B shows PrecedingResidue (residue at backbone B - 1)
                // Match means: residue(A) = residue(B-1)
                // KEY INSIGHT: Both observations see the SAME RESIDUE!
                // This is a SAME-RESIDUE correlation, not a sequential position shift!
                links.push(SequentialLink {
                    from_idx: *idx_a,
                    to_idx: *idx_b,
                    strength: -match_strength,  // NEGATIVE = same-residue (both at same position)
                });
            } else if !is_intra_a && is_intra_b {
                // A shows PrecedingResidue (residue at backbone A - 1), B shows Intra (residue at backbone B)
                // Match means: residue(A-1) = residue(B)
                // Both observe the SAME RESIDUE
                links.push(SequentialLink {
                    from_idx: *idx_a,
                    to_idx: *idx_b,
                    strength: -match_strength,  // NEGATIVE = same-residue
                });
            }
            // Both Intra or both PrecedingResidue: they observe different residues, skip
        }
    }

    // Debug: sequential links discovery (verbose mode only, see main output)
    // if iteration == 0 && !links.is_empty() {
    //     println!("Sequential links: {} found from backbone-sequential carbon matching", links.len());
    // }

    links
}

/// Update beliefs using typing, correlation, and sequential factors.
fn update_observation_beliefs_with_sequential(
    beliefs: &[Vec<f64>],
    typing_scores: &[Vec<f64>],
    correlation_scores: &[Vec<f64>],
    sequential_links: &[SequentialLink],
    domain_size: usize,
    correlation_weight: f64,
    typing_weight: f64,
    sequential_weight: f64,
) -> Vec<Vec<f64>> {
    let n = beliefs.len();
    let mut new_beliefs = vec![vec![0.0; domain_size]; n];

    for i in 0..n {
        // Start with typing prior
        for d in 0..domain_size {
            new_beliefs[i][d] = typing_scores[i][d].max(1e-10).ln() * typing_weight;
        }

        // Add correlation messages (same-residue factors)
        for j in 0..n {
            if i == j { continue; }

            let corr = correlation_scores[i][j];
            if corr > 0.01 {
                for d in 0..domain_size {
                    new_beliefs[i][d] += beliefs[j][d].max(1e-10).ln() * corr * correlation_weight;
                }
            }
        }

        // Add sequential/same-residue messages from triple-resonance carbon matching
        // POSITIVE strength = sequential (from at d → to at d+1)
        // NEGATIVE strength = same-residue (both at same position d)
        for link in sequential_links {
            if link.strength < 0.0 {
                // SAME-RESIDUE: cross-backbone intra/inter match
                // Both observations see the same residue → same position
                let abs_strength = link.strength.abs();
                let other_idx = if link.from_idx == i { link.to_idx } else if link.to_idx == i { link.from_idx } else { continue };
                for d in 0..domain_size {
                    let other_belief = beliefs[other_idx][d].max(1e-10);
                    new_beliefs[i][d] += other_belief.ln() * abs_strength * correlation_weight;
                }
            } else if link.strength > 0.0 {
                // SEQUENTIAL: true backbone ordering (not currently generated, but kept for future)
                if link.to_idx == i {
                    for d in 1..domain_size {
                        if d + 1 < domain_size {
                            let from_belief = beliefs[link.from_idx][d].max(1e-10);
                            new_beliefs[i][d + 1] += from_belief.ln() * link.strength * sequential_weight;
                        }
                    }
                }
                if link.from_idx == i {
                    for d in 2..domain_size {
                        let to_belief = beliefs[link.to_idx][d].max(1e-10);
                        new_beliefs[i][d - 1] += to_belief.ln() * link.strength * sequential_weight;
                    }
                }
            }
        }

        // Convert from log to probability and normalize
        let max_val = new_beliefs[i].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let sum: f64 = new_beliefs[i].iter_mut()
            .map(|v| { *v = (*v - max_val).exp(); *v })
            .sum();

        if sum > 0.0 {
            for v in &mut new_beliefs[i] {
                *v /= sum;
            }
        }
    }

    new_beliefs
}

// =============================================================================
// END: Observation-Based Assignment
// =============================================================================

/// Run the unified assignment algorithm.
pub fn run_unified_assignment(
    hsqc_15n: &[UnlabeledPeak],
    hsqc_13c: &[UnlabeledPeak],
    tocsy: &[UnlabeledPeak],
    noesy: &[UnlabeledPeak],
    hsqc_tocsy_15n: &[UnlabeledPeak],
    hsqc_tocsy_13c: &[UnlabeledPeak],
    // 3D HSQC-TOCSY experiments
    hsqc_tocsy_15n_3d: &[UnlabeledPeak],
    hsqc_tocsy_13c_3d: &[UnlabeledPeak],
    // 3D triple-resonance experiments
    hnco: &[UnlabeledPeak],
    hnca: &[UnlabeledPeak],
    hncacb: &[UnlabeledPeak],
    cbcaconh: &[UnlabeledPeak],
    hbhaconh: &[UnlabeledPeak],
    sequence: &str,
    params: &UnifiedAssignmentParams,
) -> Vec<UnifiedAssignmentResult> {
    let mut graph = UnifiedFactorGraph::new(
        hsqc_15n, hsqc_13c, tocsy, noesy, hsqc_tocsy_15n, hsqc_tocsy_13c,
        hsqc_tocsy_15n_3d, hsqc_tocsy_13c_3d,
        hnco, hnca, hncacb, cbcaconh, hbhaconh,
        sequence, params
    );

    tracing::info!(
        "Built unified graph: {} peaks, {} TOCSY correlations, {} HSQC-TOCSY correlations, {} NOESY backbone-carbon",
        graph.peaks.len(),
        graph.tocsy_correlations.len(),
        graph.hsqc_tocsy_correlations.len(),
        graph.noesy_backbone_carbon.len()
    );

    graph.run_bp(params);
    graph.extract_assignments()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unified_simple() {
        // Simple test with 2 residues
        let sequence = "AC";

        // Create idealized peaks
        let hsqc_15n = vec![
            UnlabeledPeak::hsqc_15n(120.0, 8.0, 1.0),  // ALA backbone
            UnlabeledPeak::hsqc_15n(115.0, 8.5, 1.0),  // CYS backbone
        ];

        let hsqc_13c = vec![
            UnlabeledPeak::hsqc_13c(52.5, 4.3, 1.0),   // ALA CA (CA ~52.5 for ALA)
            UnlabeledPeak::hsqc_13c(19.0, 1.4, 1.0),   // ALA CB (CB ~19 for ALA)
            UnlabeledPeak::hsqc_13c(58.0, 4.5, 1.0),   // CYS CA
            UnlabeledPeak::hsqc_13c(28.0, 3.0, 1.0),   // CYS CB
        ];

        // TOCSY: link backbone to carbons
        let tocsy = vec![
            // ALA: backbone H=8.0 correlates with HA=4.3 and HB=1.4
            UnlabeledPeak::tocsy(8.0, 4.3, 1.0),
            UnlabeledPeak::tocsy(4.3, 8.0, 1.0),
            UnlabeledPeak::tocsy(8.0, 1.4, 1.0),
            UnlabeledPeak::tocsy(1.4, 8.0, 1.0),
            // CYS: backbone H=8.5 correlates with HA=4.5 and HB=3.0
            UnlabeledPeak::tocsy(8.5, 4.5, 1.0),
            UnlabeledPeak::tocsy(4.5, 8.5, 1.0),
            UnlabeledPeak::tocsy(8.5, 3.0, 1.0),
            UnlabeledPeak::tocsy(3.0, 8.5, 1.0),
        ];

        // NOESY: sequential dαN (H(i) to HA(i-1))
        let noesy = vec![
            UnlabeledPeak::noesy(8.5, 4.3, 0.5),  // CYS H to ALA HA
            UnlabeledPeak::noesy(4.3, 8.5, 0.5),
        ];

        // HSQC-TOCSY (empty for this test)
        let hsqc_tocsy_15n: Vec<UnlabeledPeak> = vec![];
        let hsqc_tocsy_13c: Vec<UnlabeledPeak> = vec![];
        let hsqc_tocsy_15n_3d: Vec<UnlabeledPeak> = vec![];
        let hsqc_tocsy_13c_3d: Vec<UnlabeledPeak> = vec![];

        // 3D triple-resonance (empty for this test)
        let hnco: Vec<UnlabeledPeak> = vec![];
        let hnca: Vec<UnlabeledPeak> = vec![];
        let hncacb: Vec<UnlabeledPeak> = vec![];
        let cbcaconh: Vec<UnlabeledPeak> = vec![];
        let hbhaconh: Vec<UnlabeledPeak> = vec![];

        let params = UnifiedAssignmentParams::default();
        let results = run_unified_assignment(
            &hsqc_15n, &hsqc_13c, &tocsy, &noesy,
            &hsqc_tocsy_15n, &hsqc_tocsy_13c,
            &hsqc_tocsy_15n_3d, &hsqc_tocsy_13c_3d,
            &hnco, &hnca, &hncacb, &cbcaconh, &hbhaconh,
            sequence, &params
        );

        // Check backbone assignments
        let backbone_results: Vec<_> = results.iter()
            .filter(|r| r.peak_type == PeakType::Backbone)
            .collect();

        assert_eq!(backbone_results.len(), 2, "Should assign both backbone peaks");

        // The ALA backbone (H=8.0) should be assigned to residue 1
        // The CYS backbone (H=8.5) should be assigned to residue 2
        println!("\nBackbone assignments:");
        for r in &backbone_results {
            println!("  Peak {:?} -> residue {} (conf={:.2})", r.peak_id, r.assigned_residue, r.confidence);
        }
    }
}
