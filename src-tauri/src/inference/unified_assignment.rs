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
use crate::data::residue_topology::{get_topology_by_three, ResidueTopology};
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
            h_tolerance_base: 0.02,   // 0.02 ppm for protons (tight for disambiguation)
            c_tolerance_base: 0.2,    // 0.2 ppm for carbons (sharp 13C lines)
            n_tolerance_base: 0.05,   // 0.05 ppm for nitrogen (N15 is very sharp)
            tolerance_schedule: ToleranceSchedule::Linear {
                start_mult: 4.0,  // Start at 4x base (loose for initial matching)
                end_mult: 1.0,    // End at 1x base (tight for disambiguation)
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
                        // Could be any aliphatic or aromatic carbon
                        atom_constraint: AtomConstraint::OneOf(vec![
                            "CA".into(), "CB".into(), "CG".into(), "CG1".into(), "CG2".into(),
                            "CD".into(), "CD1".into(), "CD2".into(),
                            "CE".into(), "CE1".into(), "CE2".into(), "CE3".into(),
                            "CZ".into(), "CZ2".into(), "CZ3".into(), "CH2".into(),
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
                // HNCACB encoding:
                // - SIGN determines CA vs CB: positive = CA, negative = CB (CB is inverted)
                // - MAGNITUDE determines intra vs inter: |intensity| > 0.5 = intra (i), <= 0.5 = inter (i-1)
                let is_ca = peak.intensity > 0.0;
                let is_intra = peak.intensity.abs() > 0.5;
                let carbon_offset = if is_intra { ResidueOffset::Intra } else { ResidueOffset::PrecedingResidue };
                let is_seq = !is_intra;

                // Use sign to determine CA vs CB (not chemical shift threshold!)
                // This correctly handles Ser/Thr CB (~60-70 ppm) which would be misclassified by shift alone
                let carbon_shift = peak.position_ppm[2];
                let (atom_hint, atom_constraint) = if is_ca {
                    (Some("CA".to_string()), AtomConstraint::Exact("CA".to_string()))
                } else {
                    (Some("CB".to_string()), AtomConstraint::Exact("CB".to_string()))
                };

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
                        shift: carbon_shift,
                        atom_hint,
                        residue_offset: carbon_offset,
                        atom_constraint,
                    },
                ];
                (dims, TransferPathway::BackboneSequential, is_seq)
            }

            Cbcaconh => {
                if peak.position_ppm.len() < 3 { return None; }
                // CBCACONH ALWAYS shows i-1 carbons (inter-residue only)
                // Use sign encoding consistent with HNCACB: positive = CA, negative = CB
                let is_ca = peak.intensity > 0.0;
                let carbon_shift = peak.position_ppm[2];
                let (atom_hint, atom_constraint) = if is_ca {
                    (Some("CA".to_string()), AtomConstraint::Exact("CA".to_string()))
                } else {
                    (Some("CB".to_string()), AtomConstraint::Exact("CB".to_string()))
                };

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
                        shift: carbon_shift,
                        atom_hint,
                        residue_offset: ResidueOffset::PrecedingResidue,
                        atom_constraint,
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

            Hncaco => {
                if peak.position_ppm.len() < 3 { return None; }
                // HN(CA)CO shows CO(i) strong, CO(i-1) weak
                // Similar to HNCA - intensity determines intra vs inter
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
                        atom_hint: Some("C".to_string()),
                        residue_offset: carbon_offset,
                        atom_constraint: AtomConstraint::Exact("C".to_string()),
                    },
                ];
                (dims, TransferPathway::BackboneSequential, is_seq)
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

    // =========================================================================
    // Physics-based query methods (use these instead of experiment_type checks)
    // =========================================================================

    /// Returns true if this observation has backbone H-N anchor for grouping.
    /// This replaces: `experiment_type == Hsqc15N || matches!(experiment_type, Hnca | Hncaco | ...)`
    pub fn has_backbone_hn(&self) -> bool {
        let has_n15_intra = self.dimensions.iter().any(|d|
            d.nucleus == NucleusType::N15 && d.residue_offset == ResidueOffset::Intra);
        let has_h1_intra = self.dimensions.iter().any(|d|
            d.nucleus == NucleusType::H1 && d.residue_offset == ResidueOffset::Intra);
        has_n15_intra && has_h1_intra
    }

    /// Get the backbone (H, N) shifts if present, with H first, N second.
    /// Returns None if this observation doesn't have a backbone anchor.
    pub fn backbone_shifts(&self) -> Option<(f64, f64)> {
        let h_shift = self.dimensions.iter()
            .find(|d| d.nucleus == NucleusType::H1 && d.residue_offset == ResidueOffset::Intra)?
            .shift;
        let n_shift = self.dimensions.iter()
            .find(|d| d.nucleus == NucleusType::N15 && d.residue_offset == ResidueOffset::Intra)?
            .shift;
        Some((h_shift, n_shift))
    }

    /// Get all carbon dimensions from this observation.
    pub fn carbon_dimensions(&self) -> impl Iterator<Item = &ObservedDimension> {
        self.dimensions.iter().filter(|d| d.nucleus == NucleusType::C13)
    }

    /// Get all intra-residue carbon dimensions.
    pub fn intra_carbon_dimensions(&self) -> impl Iterator<Item = &ObservedDimension> {
        self.dimensions.iter().filter(|d|
            d.nucleus == NucleusType::C13 && d.residue_offset == ResidueOffset::Intra)
    }

    /// Get all inter-residue (preceding) carbon dimensions.
    pub fn inter_carbon_dimensions(&self) -> impl Iterator<Item = &ObservedDimension> {
        self.dimensions.iter().filter(|d|
            d.nucleus == NucleusType::C13 && d.residue_offset == ResidueOffset::PrecedingResidue)
    }

    /// Check if all non-unknown dimensions are intra-residue.
    /// Returns true for HSQC, TOCSY, etc. where all nuclei are at the same residue.
    pub fn is_purely_intra(&self) -> bool {
        self.dimensions.iter()
            .filter(|d| d.residue_offset != ResidueOffset::Unknown)
            .all(|d| d.residue_offset == ResidueOffset::Intra)
    }

    /// Check if any dimension provides inter-residue evidence.
    /// This is equivalent to `is_sequential_evidence` but computed from dimensions.
    pub fn has_inter_residue_dimension(&self) -> bool {
        self.dimensions.iter().any(|d| d.residue_offset == ResidueOffset::PrecedingResidue)
    }

    // =========================================================================
    // Nucleus presence checks (for filtering without experiment_type)
    // =========================================================================

    /// Check if this observation has any N15 dimension.
    pub fn has_n15(&self) -> bool {
        self.dimensions.iter().any(|d| d.nucleus == NucleusType::N15)
    }

    /// Check if this observation has any C13 dimension.
    pub fn has_c13(&self) -> bool {
        self.dimensions.iter().any(|d| d.nucleus == NucleusType::C13)
    }

    /// Check if this observation has any H1 dimension.
    pub fn has_h1(&self) -> bool {
        self.dimensions.iter().any(|d| d.nucleus == NucleusType::H1)
    }

    /// Check if this observation has an intra-residue N15 dimension.
    pub fn has_intra_n15(&self) -> bool {
        self.dimensions.iter().any(|d|
            d.nucleus == NucleusType::N15 && d.residue_offset == ResidueOffset::Intra)
    }

    /// Check if this observation has an intra-residue C13 dimension.
    pub fn has_intra_c13(&self) -> bool {
        self.dimensions.iter().any(|d|
            d.nucleus == NucleusType::C13 && d.residue_offset == ResidueOffset::Intra)
    }

    /// Check if this observation has an intra-residue H1 dimension.
    pub fn has_intra_h1(&self) -> bool {
        self.dimensions.iter().any(|d|
            d.nucleus == NucleusType::H1 && d.residue_offset == ResidueOffset::Intra)
    }

    // =========================================================================
    // Transfer pathway checks (for filtering without experiment_type)
    // =========================================================================

    /// Check if this observation is from a direct bond experiment (HSQC).
    pub fn is_direct_bond(&self) -> bool {
        self.transfer_pathway == TransferPathway::DirectBond
    }

    /// Check if this observation is from a through-bond experiment (TOCSY, HSQC-TOCSY).
    pub fn is_throughbond(&self) -> bool {
        self.transfer_pathway == TransferPathway::ThroughBond
    }

    /// Check if this observation is from a through-space experiment (NOESY).
    pub fn is_throughspace(&self) -> bool {
        self.transfer_pathway == TransferPathway::ThroughSpace
    }

    /// Check if this observation is from a backbone sequential experiment (triple resonance).
    pub fn is_backbone_sequential(&self) -> bool {
        self.transfer_pathway == TransferPathway::BackboneSequential
    }

    // =========================================================================
    // Anchor type checks (for replacing PeakType experiment-specific logic)
    // =========================================================================

    /// Check if this observation is nitrogen-anchored (intra H + intra N15).
    /// Replaces experiment_type == Hsqc15N check.
    pub fn is_nitrogen_anchored(&self) -> bool {
        self.has_intra_n15() && self.has_intra_h1()
    }

    /// Check if this observation is carbon-anchored (intra H + intra C13).
    /// Replaces experiment_type == Hsqc13C check.
    pub fn is_carbon_anchored(&self) -> bool {
        self.has_intra_c13() && self.has_intra_h1()
    }

    /// Check if this observation is HSQC-TOCSY-like: through-bond with N15 anchor.
    /// Replaces experiment_type == HsqcTocsy15N check.
    pub fn is_tocsy_n15_anchored(&self) -> bool {
        self.is_throughbond() && self.has_n15()
    }

    /// Check if this observation is HSQC-TOCSY-like: through-bond with C13 anchor.
    /// Replaces experiment_type == HsqcTocsy13C check.
    pub fn is_tocsy_c13_anchored(&self) -> bool {
        self.is_throughbond() && self.has_c13()
    }

    /// Check if this observation is TOCSY-like: through-bond proton-proton correlation.
    /// Replaces experiment_type == Tocsy check.
    pub fn is_homonuclear_tocsy(&self) -> bool {
        self.is_throughbond() &&
        self.dimensions.iter().filter(|d| d.nucleus == NucleusType::H1).count() >= 2
    }

    /// Check if this observation is NOESY-like: through-space proton-proton correlation.
    /// Replaces experiment_type == Noesy check.
    pub fn is_homonuclear_noesy(&self) -> bool {
        self.is_throughspace() &&
        self.dimensions.iter().filter(|d| d.nucleus == NucleusType::H1).count() >= 2
    }
}

impl ObservedDimension {
    /// Check if this dimension is a carbonyl carbon (C').
    pub fn is_carbonyl(&self) -> bool {
        matches!(&self.atom_constraint, AtomConstraint::Exact(s) if s == "C")
    }

    /// Check if this dimension is a CA carbon.
    pub fn is_ca(&self) -> bool {
        matches!(&self.atom_constraint, AtomConstraint::Exact(s) if s == "CA")
    }

    /// Check if this dimension is a CB carbon.
    pub fn is_cb(&self) -> bool {
        matches!(&self.atom_constraint, AtomConstraint::Exact(s) if s == "CB")
    }

    /// Check if this dimension is intra-residue.
    pub fn is_intra(&self) -> bool {
        self.residue_offset == ResidueOffset::Intra
    }

    /// Check if this dimension is inter-residue (preceding).
    pub fn is_inter(&self) -> bool {
        self.residue_offset == ResidueOffset::PrecedingResidue
    }
}

// =============================================================================
// END: Unified Observation Model
// =============================================================================

// =============================================================================
// NEW: Group-Level Bayesian Assignment (Simplified Physics-Based Approach)
// =============================================================================
//
// Key insight: We have N_obs observations but only N_groups decisions to make.
// Each spin system (backbone group) maps to exactly one sequence position.
//
// Instead of running BP on observations (fighting synchronization), we:
// 1. Group observations by backbone (H, N)
// 2. Aggregate evidence into each group ONCE
// 3. Run BP on groups (66 variables, not 586)
// 4. Map group assignments back to observations

/// Aggregated evidence for one spin system (backbone group).
/// This is the unit of assignment - one group = one sequence position.
#[derive(Debug, Clone)]
pub struct SpinSystemEvidence {
    /// Index of this group
    pub group_idx: usize,
    /// Observation indices belonging to this group
    pub observation_indices: Vec<usize>,
    /// Backbone (H, N) coordinates
    pub h_shift: f64,
    pub n_shift: f64,
    /// Intra-residue carbon shifts: atom_type -> shift
    /// These are carbons observed at THIS residue (intensity > 0.5)
    pub intra_carbons: HashMap<String, f64>,
    /// Inter-residue carbon shifts: atom_type -> shift
    /// These are carbons observed at PRECEDING residue (intensity <= 0.5)
    pub inter_carbons: HashMap<String, f64>,
    /// Intra-residue proton shifts (from TOCSY/HSQC-13C): atom_type -> shift
    pub intra_protons: HashMap<String, f64>,
    /// Inter-residue proton shifts (from HBHACONH): HA(i-1), HB(i-1)
    pub inter_protons: HashMap<String, f64>,
}

impl SpinSystemEvidence {
    /// Create evidence for a group from observations
    pub fn from_observations(
        group_idx: usize,
        obs_indices: &[usize],
        observations: &[Observation],
    ) -> Self {
        let mut h_shift = 0.0;
        let mut n_shift = 0.0;
        let mut intra_carbons: HashMap<String, f64> = HashMap::new();
        let mut inter_carbons: HashMap<String, f64> = HashMap::new();
        let mut intra_protons: HashMap<String, f64> = HashMap::new();
        let mut inter_protons: HashMap<String, f64> = HashMap::new();

        // Find backbone (H, N) from any observation in the group
        for &idx in obs_indices {
            let obs = &observations[idx];
            for dim in &obs.dimensions {
                if dim.residue_offset == ResidueOffset::Intra {
                    match dim.nucleus {
                        NucleusType::H1 => {
                            // Backbone H
                            if dim.atom_hint.as_deref() == Some("H") ||
                               dim.atom_hint.as_deref() == Some("HN") ||
                               matches!(&dim.atom_constraint, AtomConstraint::Exact(s) if s == "H") {
                                h_shift = dim.shift;
                            }
                        }
                        NucleusType::N15 => {
                            n_shift = dim.shift;
                        }
                        _ => {}
                    }
                }
            }
            if h_shift > 0.0 && n_shift > 0.0 {
                break;
            }
        }

        // Collect all carbon and proton shifts from observations
        for &idx in obs_indices {
            let obs = &observations[idx];
            let is_intra = obs.intensity > 0.5;

            for dim in &obs.dimensions {
                match dim.nucleus {
                    NucleusType::C13 => {
                        // Determine atom type from constraint or hint
                        let atom_type = match &dim.atom_constraint {
                            AtomConstraint::Exact(s) => s.clone(),
                            AtomConstraint::OneOf(v) if v.len() == 1 => v[0].clone(),
                            _ => dim.atom_hint.clone().unwrap_or_else(|| "C".to_string()),
                        };

                        // Use residue_offset if available, otherwise use intensity
                        let is_dim_intra = match dim.residue_offset {
                            ResidueOffset::Intra => true,
                            ResidueOffset::PrecedingResidue => false,
                            ResidueOffset::Unknown => is_intra,
                            _ => is_intra,  // FollowingResidue, etc.
                        };

                        if is_dim_intra {
                            intra_carbons.insert(atom_type, dim.shift);
                        } else {
                            inter_carbons.insert(atom_type, dim.shift);
                        }
                    }
                    NucleusType::H1 => {
                        // Check if this is an aliphatic proton (HA/HB from HBHACONH)
                        let is_aliphatic = match &dim.atom_constraint {
                            AtomConstraint::Exact(s) => s.starts_with("HA") || s.starts_with("HB"),
                            AtomConstraint::OneOf(v) => v.iter().any(|s| s.starts_with("HA") || s.starts_with("HB")),
                            AtomConstraint::Any => false,
                        };

                        if is_aliphatic {
                            // Determine HA vs HB based on shift (HA ~4.3, HB ~1.5-3.0)
                            let atom_type = if dim.shift > 3.5 { "HA".to_string() } else { "HB".to_string() };

                            let is_dim_intra = match dim.residue_offset {
                                ResidueOffset::Intra => true,
                                ResidueOffset::PrecedingResidue => false,
                                _ => is_intra,
                            };

                            if is_dim_intra {
                                intra_protons.insert(atom_type, dim.shift);
                            } else {
                                inter_protons.insert(atom_type, dim.shift);
                            }
                        } else if dim.atom_hint.as_deref() != Some("H") &&
                                  dim.atom_hint.as_deref() != Some("HN") {
                            // Sidechain protons from TOCSY (not backbone H)
                            if let Some(hint) = &dim.atom_hint {
                                intra_protons.insert(hint.clone(), dim.shift);
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        Self {
            group_idx,
            observation_indices: obs_indices.to_vec(),
            h_shift,
            n_shift,
            intra_carbons,
            inter_carbons,
            intra_protons,
            inter_protons,
        }
    }

    /// Get intra-residue CA shift if present
    pub fn intra_ca(&self) -> Option<f64> {
        self.intra_carbons.get("CA").copied()
    }

    /// Get inter-residue (i-1) CA shift if present
    pub fn inter_ca(&self) -> Option<f64> {
        self.inter_carbons.get("CA").copied()
    }

    /// Get intra-residue CB shift if present
    pub fn intra_cb(&self) -> Option<f64> {
        self.intra_carbons.get("CB").copied()
    }

    /// Get inter-residue (i-1) CB shift if present
    pub fn inter_cb(&self) -> Option<f64> {
        self.inter_carbons.get("CB").copied()
    }
}

// ============================================================================
// SIMULATED ANNEALING FOR SEQUENTIAL CONSISTENCY
// ============================================================================

/// A sequential link between two backbone groups based on carbon matching.
/// If group_i has inter-CA that matches group_j's intra-CA, it means
/// group_j PRECEDES group_i in the sequence (j is at position i-1).
#[derive(Debug, Clone)]
pub struct SequentialLink {
    /// The group that comes FIRST in sequence (has matching intra carbons)
    pub from_group: usize,
    /// The group that comes SECOND in sequence (has matching inter carbons)
    pub to_group: usize,
    /// Match quality (0-1), higher = better match
    pub strength: f64,
    /// |inter_CA - intra_CA| in ppm
    pub ca_delta: f64,
    /// |inter_CB - intra_CB| in ppm, if both present
    pub cb_delta: Option<f64>,
}

/// Build sequential connectivity graph from carbon matching.
/// Returns links where from_group PRECEDES to_group in sequence.
pub fn build_sequential_links(
    groups: &[SpinSystemEvidence],
    ca_tolerance: f64,
    cb_tolerance: f64,
) -> Vec<SequentialLink> {
    let mut links = Vec::new();

    for (i, group_i) in groups.iter().enumerate() {
        // group_i has inter carbons = previous residue's carbons
        let inter_ca = group_i.inter_ca();
        let inter_cb = group_i.inter_cb();

        // Skip if no inter carbons
        if inter_ca.is_none() {
            continue;
        }

        for (j, group_j) in groups.iter().enumerate() {
            if i == j {
                continue;
            }

            // group_j has intra carbons = this residue's carbons
            let intra_ca = group_j.intra_ca();
            let intra_cb = group_j.intra_cb();

            // Check CA match: group_i's inter-CA should match group_j's intra-CA
            if let (Some(inter), Some(intra)) = (inter_ca, intra_ca) {
                let ca_delta = (inter - intra).abs();
                if ca_delta < ca_tolerance {
                    let mut strength = 1.0 - (ca_delta / ca_tolerance);

                    // Boost if CB also matches
                    let cb_delta = match (inter_cb, intra_cb) {
                        (Some(inter_cb_val), Some(intra_cb_val)) => {
                            let delta = (inter_cb_val - intra_cb_val).abs();
                            if delta < cb_tolerance {
                                strength += 0.5 * (1.0 - delta / cb_tolerance);
                            }
                            Some(delta)
                        }
                        _ => None,
                    };

                    if strength > 0.3 {
                        // Link means: group_j PRECEDES group_i in sequence
                        links.push(SequentialLink {
                            from_group: j, // previous (has intra that matches)
                            to_group: i,   // current (has inter)
                            strength,
                            ca_delta,
                            cb_delta,
                        });
                    }
                }
            }
        }
    }

    links
}

/// Score an assignment for sequential consistency and typing.
/// Higher score = better assignment.
pub fn score_sa_assignment(
    assignment: &[Option<usize>], // group_idx -> residue position (1-indexed)
    groups: &[SpinSystemEvidence],
    sequence: &str,
    links: &[SequentialLink],
) -> f64 {
    let mut score = 0.0;
    let seq_len = sequence.len();

    // === 1. Sequential Consistency (most important!) ===
    for link in links {
        if let (Some(pos_from), Some(pos_to)) =
            (assignment[link.from_group], assignment[link.to_group])
        {
            // from_group should be at pos_to - 1 (it precedes to_group)
            let expected_delta = 1i32;
            let actual_delta = pos_to as i32 - pos_from as i32;

            if actual_delta == expected_delta {
                // Perfect! Sequential link satisfied
                score += link.strength * 20.0;
            } else if actual_delta == 0 {
                // Same position - impossible!
                score -= 100.0;
            } else {
                // Wrong gap - penalize proportionally
                score -= (actual_delta - expected_delta).abs() as f64 * 5.0;
            }
        }
    }

    // === 2. Anchor Typing (Gly, Ala, Ser/Thr have distinctive CB) ===
    let seq_chars: Vec<char> = sequence.chars().collect();
    for (group_idx, &pos) in assignment.iter().enumerate() {
        if let Some(pos) = pos {
            if pos < 1 || pos > seq_len {
                score -= 1000.0; // Out of bounds
                continue;
            }

            let residue_type = seq_chars[pos - 1];
            let group = &groups[group_idx];

            let intra_ca = group.intra_ca();
            let intra_cb = group.intra_cb();

            match residue_type {
                'G' => {
                    // Glycine: CA ~45ppm, NO CB
                    if let Some(ca) = intra_ca {
                        if ca < 48.0 {
                            score += 25.0; // Good Gly CA
                            if intra_cb.is_none() {
                                score += 25.0; // No CB = definitely Gly
                            } else {
                                score -= 30.0; // Gly shouldn't have CB!
                            }
                        } else {
                            score -= 20.0; // CA too high for Gly
                        }
                    }
                }
                'A' => {
                    // Alanine: CB ~18ppm (uniquely low)
                    if let Some(cb) = intra_cb {
                        if cb < 22.0 {
                            score += 20.0; // Perfect Ala CB
                        } else if cb > 25.0 {
                            score -= 15.0; // Too high for Ala
                        }
                    }
                }
                'S' => {
                    // Serine: CB ~63ppm
                    if let Some(cb) = intra_cb {
                        if cb > 60.0 && cb < 68.0 {
                            score += 15.0;
                        } else if cb < 55.0 {
                            score -= 10.0;
                        }
                    }
                }
                'T' => {
                    // Threonine: CB ~69ppm (highest)
                    if let Some(cb) = intra_cb {
                        if cb > 65.0 {
                            score += 15.0;
                        } else if cb < 60.0 {
                            score -= 10.0;
                        }
                    }
                }
                'V' => {
                    // Valine: CB ~32ppm
                    if let Some(cb) = intra_cb {
                        if cb > 29.0 && cb < 36.0 {
                            score += 8.0;
                        }
                    }
                }
                'I' => {
                    // Isoleucine: CB ~38ppm
                    if let Some(cb) = intra_cb {
                        if cb > 35.0 && cb < 42.0 {
                            score += 8.0;
                        }
                    }
                }
                'L' => {
                    // Leucine: CB ~42ppm
                    if let Some(cb) = intra_cb {
                        if cb > 39.0 && cb < 45.0 {
                            score += 8.0;
                        }
                    }
                }
                'P' => {
                    // Proline: no backbone NH, shouldn't have a group!
                    // If a group is assigned to Pro, penalize
                    score -= 50.0;
                }
                _ => {
                    // General typing - no specific bonus/penalty
                }
            }
        }
    }

    // === 3. Uniqueness Constraint ===
    let mut position_counts = vec![0usize; seq_len];
    for &pos in assignment.iter().flatten() {
        if pos >= 1 && pos <= seq_len {
            position_counts[pos - 1] += 1;
        }
    }
    for count in position_counts {
        if count > 1 {
            score -= (count - 1) as f64 * 50.0; // Heavy penalty for collisions
        }
    }

    // === 4. Completeness Bonus ===
    let assigned_count = assignment.iter().filter(|a| a.is_some()).count();
    score += assigned_count as f64 * 2.0;

    score
}

/// SA Move types for exploring the assignment space
#[derive(Debug, Clone)]
pub enum SAMove {
    /// Swap assignments of two groups
    Swap { group_a: usize, group_b: usize },
    /// Move one group to a new position
    Relocate { group: usize, new_pos: usize },
    /// Shift an entire connected stretch by ±1
    ShiftStretch { anchor_group: usize, delta: i32 },
}

/// Find all groups transitively connected to anchor via sequential links
fn find_connected_stretch(
    anchor: usize,
    links: &[SequentialLink],
    n_groups: usize,
) -> Vec<usize> {
    let mut stretch = vec![anchor];
    let mut visited = vec![false; n_groups];
    visited[anchor] = true;

    let mut queue = vec![anchor];
    while let Some(current) = queue.pop() {
        for link in links {
            let neighbor = if link.from_group == current {
                link.to_group
            } else if link.to_group == current {
                link.from_group
            } else {
                continue;
            };

            if !visited[neighbor] && link.strength > 0.5 {
                visited[neighbor] = true;
                stretch.push(neighbor);
                queue.push(neighbor);
            }
        }
    }

    stretch
}

/// Propose a move for SA based on temperature (exploration vs exploitation)
fn propose_sa_move(
    assignment: &[Option<usize>],
    links: &[SequentialLink],
    sequence_len: usize,
    rng: &mut impl rand::Rng,
    temperature: f64,
) -> SAMove {
    use rand::seq::SliceRandom;

    // Higher temperature = more exploration, lower = more local refinement
    let exploration_prob = (temperature / 10.0).min(0.5);

    if rng.gen::<f64>() < exploration_prob {
        // Exploration: big moves
        match rng.gen_range(0..3) {
            0 => {
                // Random swap
                let a = rng.gen_range(0..assignment.len());
                let b = rng.gen_range(0..assignment.len());
                SAMove::Swap { group_a: a, group_b: b }
            }
            1 => {
                // Random relocate
                let group = rng.gen_range(0..assignment.len());
                let new_pos = rng.gen_range(1..=sequence_len);
                SAMove::Relocate { group, new_pos }
            }
            _ => {
                // Shift stretch
                let anchor = rng.gen_range(0..assignment.len());
                let delta = if rng.gen_bool(0.5) { 1 } else { -1 };
                SAMove::ShiftStretch {
                    anchor_group: anchor,
                    delta,
                }
            }
        }
    } else {
        // Exploitation: local moves to fix sequential violations
        let violated: Vec<_> = links
            .iter()
            .filter(|link| {
                if let (Some(pos_from), Some(pos_to)) =
                    (assignment[link.from_group], assignment[link.to_group])
                {
                    pos_to as i32 - pos_from as i32 != 1
                } else {
                    false
                }
            })
            .collect();

        if !violated.is_empty() {
            // Pick a random violated link
            let link = violated[rng.gen_range(0..violated.len())];
            // Try to fix this link by moving one of the groups
            if rng.gen_bool(0.5) {
                if let Some(pos_to) = assignment[link.to_group] {
                    SAMove::Relocate {
                        group: link.from_group,
                        new_pos: pos_to.saturating_sub(1).max(1usize),
                    }
                } else {
                    SAMove::Swap {
                        group_a: link.from_group,
                        group_b: link.to_group,
                    }
                }
            } else if let Some(pos_from) = assignment[link.from_group] {
                SAMove::Relocate {
                    group: link.to_group,
                    new_pos: (pos_from + 1).min(sequence_len),
                }
            } else {
                SAMove::Swap {
                    group_a: link.from_group,
                    group_b: link.to_group,
                }
            }
        } else {
            // No violations - small random perturbation
            let group = rng.gen_range(0..assignment.len());
            if let Some(pos) = assignment[group] {
                let delta: i32 = if rng.gen_bool(0.5) { -1 } else { 1 };
                let new_pos = ((pos as i32 + delta).max(1) as usize).min(sequence_len);
                SAMove::Relocate { group, new_pos }
            } else {
                let new_pos = rng.gen_range(1..=sequence_len);
                SAMove::Relocate { group, new_pos }
            }
        }
    }
}

/// Apply a move to an assignment
fn apply_sa_move(
    assignment: &mut [Option<usize>],
    mv: &SAMove,
    links: &[SequentialLink],
    seq_len: usize,
) {
    match mv {
        SAMove::Swap { group_a, group_b } => {
            assignment.swap(*group_a, *group_b);
        }
        SAMove::Relocate { group, new_pos } => {
            assignment[*group] = Some(*new_pos);
        }
        SAMove::ShiftStretch { anchor_group, delta } => {
            // Find all groups connected to anchor
            let stretch = find_connected_stretch(*anchor_group, links, assignment.len());
            for group_idx in stretch {
                if let Some(pos) = assignment[group_idx] {
                    let new_pos = (pos as i32 + delta).max(1).min(seq_len as i32) as usize;
                    assignment[group_idx] = Some(new_pos);
                }
            }
        }
    }
}

/// Refine an assignment using Simulated Annealing
pub fn refine_with_sa(
    beliefs: &[Vec<f64>], // group_idx -> position -> probability
    groups: &[SpinSystemEvidence],
    sequence: &str,
    links: &[SequentialLink],
    temperature: f64,
    sa_iterations: usize,
) -> Vec<Option<usize>> {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    // Initialize from BP beliefs (argmax)
    let mut assignment: Vec<Option<usize>> = beliefs
        .iter()
        .map(|probs| {
            probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .filter(|(_, &p)| p > 0.05) // Only assign if belief is reasonable
                .map(|(pos, _)| pos + 1) // Convert 0-indexed to 1-indexed
        })
        .collect();

    let mut current_score = score_sa_assignment(&assignment, groups, sequence, links);
    let mut best_assignment = assignment.clone();
    let mut best_score = current_score;

    for iter in 0..sa_iterations {
        // Adaptive temperature within SA loop
        let local_temp = temperature * (1.0 - iter as f64 / sa_iterations as f64);

        // Propose move
        let mv = propose_sa_move(&assignment, links, sequence.len(), &mut rng, local_temp);

        // Apply move to copy
        let mut new_assignment = assignment.clone();
        apply_sa_move(&mut new_assignment, &mv, links, sequence.len());

        // Score
        let new_score = score_sa_assignment(&new_assignment, groups, sequence, links);

        // Metropolis acceptance
        let delta = new_score - current_score;
        let accept = if delta > 0.0 {
            true
        } else {
            let prob = (delta / local_temp.max(0.1)).exp();
            rng.gen::<f64>() < prob
        };

        if accept {
            assignment = new_assignment;
            current_score = new_score;

            if current_score > best_score {
                best_assignment = assignment.clone();
                best_score = current_score;
            }
        }
    }

    best_assignment
}

/// A carbon observation with weighted contribution to a soft spin system.
#[derive(Debug, Clone)]
pub struct WeightedCarbon {
    /// Chemical shift in ppm
    pub shift: f64,
    /// Fit quality weight (0-1) based on (H, N) match to group center
    pub weight: f64,
    /// Observation index this came from
    pub obs_idx: usize,
}

/// Weighted proton observation (similar to WeightedCarbon)
#[derive(Debug, Clone)]
pub struct WeightedProton {
    /// Chemical shift in ppm
    pub shift: f64,
    /// Fit quality weight (0-1) based on (H, N) match to group center
    pub weight: f64,
    /// Observation index this came from
    pub obs_idx: usize,
}

/// Soft spin system evidence - allows observations to contribute to multiple groups
/// with weighted probability based on chemical shift fit.
#[derive(Debug, Clone)]
pub struct SoftSpinSystemEvidence {
    /// Index of this group
    pub group_idx: usize,
    /// Group center (H, N) - can drift during BP
    pub h_center: f64,
    pub n_center: f64,
    /// Intra-residue carbons: atom_type -> Vec of weighted contributions
    pub intra_carbons: HashMap<String, Vec<WeightedCarbon>>,
    /// Inter-residue carbons: atom_type -> Vec of weighted contributions
    pub inter_carbons: HashMap<String, Vec<WeightedCarbon>>,
    /// Intra-residue protons: atom_type -> Vec of weighted contributions (HA, HB from HBHACONH etc)
    pub intra_protons: HashMap<String, Vec<WeightedProton>>,
    /// Inter-residue protons: atom_type -> Vec of weighted contributions (HA(i-1), HB(i-1) from HBHACONH)
    pub inter_protons: HashMap<String, Vec<WeightedProton>>,
    /// Observation weights: obs_idx -> fit_quality
    pub observation_weights: HashMap<usize, f64>,
    /// Total weight from all observations (for normalization)
    pub total_weight: f64,
}

impl SoftSpinSystemEvidence {
    /// Create a new soft spin system with given center
    pub fn new(group_idx: usize, h_center: f64, n_center: f64) -> Self {
        Self {
            group_idx,
            h_center,
            n_center,
            intra_carbons: HashMap::new(),
            inter_carbons: HashMap::new(),
            intra_protons: HashMap::new(),
            inter_protons: HashMap::new(),
            observation_weights: HashMap::new(),
            total_weight: 0.0,
        }
    }

    /// Add an observation's contribution with given weight
    pub fn add_observation(&mut self, obs_idx: usize, obs: &Observation, weight: f64) {
        self.observation_weights.insert(obs_idx, weight);
        self.total_weight += weight;

        for dim in &obs.dimensions {
            // Add carbons with weighted contribution
            if dim.nucleus == NucleusType::C13 {
                let atom_type = match &dim.atom_constraint {
                    AtomConstraint::Exact(s) => s.clone(),
                    AtomConstraint::OneOf(v) if v.len() == 1 => v[0].clone(),
                    _ => dim.atom_hint.clone().unwrap_or_else(|| "C".to_string()),
                };

                let wc = WeightedCarbon {
                    shift: dim.shift,
                    weight,
                    obs_idx,
                };

                // Use residue_offset if available, otherwise use intensity
                let is_intra = match dim.residue_offset {
                    ResidueOffset::Intra => true,
                    ResidueOffset::PrecedingResidue => false,
                    _ => obs.intensity > 0.5,
                };

                if is_intra {
                    self.intra_carbons.entry(atom_type).or_default().push(wc);
                } else {
                    self.inter_carbons.entry(atom_type).or_default().push(wc);
                }
            }
            // Add aliphatic protons (HA, HB from HBHACONH etc) - NOT backbone H/HN
            else if dim.nucleus == NucleusType::H1 {
                // Check if this is an aliphatic proton (HA/HB), not backbone H
                let is_aliphatic = match &dim.atom_constraint {
                    AtomConstraint::Exact(s) => s.starts_with("HA") || s.starts_with("HB"),
                    AtomConstraint::OneOf(v) => v.iter().any(|s| s.starts_with("HA") || s.starts_with("HB")),
                    AtomConstraint::Any => false, // Don't know, skip
                };

                if is_aliphatic {
                    // Determine atom type - use "HA" or "HB" as generic type
                    let atom_type = match &dim.atom_constraint {
                        AtomConstraint::Exact(s) => {
                            if s.starts_with("HA") { "HA".to_string() }
                            else { "HB".to_string() }
                        },
                        AtomConstraint::OneOf(v) => {
                            // HBHACONH has mixed HA/HB - check hint or first match
                            if let Some(hint) = &dim.atom_hint {
                                if hint.starts_with("HA") { "HA".to_string() }
                                else { "HB".to_string() }
                            } else if v.iter().any(|s| s.starts_with("HA")) && !v.iter().any(|s| s.starts_with("HB")) {
                                "HA".to_string()
                            } else if v.iter().any(|s| s.starts_with("HB")) && !v.iter().any(|s| s.starts_with("HA")) {
                                "HB".to_string()
                            } else {
                                // Mixed HA/HB - use shift to discriminate (HA ~4.3, HB ~1.5-3.0)
                                if dim.shift > 3.5 { "HA".to_string() } else { "HB".to_string() }
                            }
                        },
                        AtomConstraint::Any => "H_ali".to_string(),
                    };

                    let wp = WeightedProton {
                        shift: dim.shift,
                        weight,
                        obs_idx,
                    };

                    // Use residue_offset for intra/inter
                    let is_intra = match dim.residue_offset {
                        ResidueOffset::Intra => true,
                        ResidueOffset::PrecedingResidue => false,
                        _ => obs.intensity > 0.5,
                    };

                    if is_intra {
                        self.intra_protons.entry(atom_type).or_default().push(wp);
                    } else {
                        self.inter_protons.entry(atom_type).or_default().push(wp);
                    }
                }
            }
        }
    }

    /// Get weighted average shift for an intra carbon atom type
    pub fn get_weighted_intra_carbon(&self, atom_type: &str) -> Option<f64> {
        let carbons = self.intra_carbons.get(atom_type)?;
        if carbons.is_empty() {
            return None;
        }
        let total_weight: f64 = carbons.iter().map(|c| c.weight).sum();
        if total_weight < 0.001 {
            return None;
        }
        let weighted_sum: f64 = carbons.iter().map(|c| c.shift * c.weight).sum();
        Some(weighted_sum / total_weight)
    }

    /// Get weighted average shift for an inter carbon atom type
    pub fn get_weighted_inter_carbon(&self, atom_type: &str) -> Option<f64> {
        let carbons = self.inter_carbons.get(atom_type)?;
        if carbons.is_empty() {
            return None;
        }
        let total_weight: f64 = carbons.iter().map(|c| c.weight).sum();
        if total_weight < 0.001 {
            return None;
        }
        let weighted_sum: f64 = carbons.iter().map(|c| c.shift * c.weight).sum();
        Some(weighted_sum / total_weight)
    }

    /// Get weighted average shift for an intra proton atom type (HA, HB)
    pub fn get_weighted_intra_proton(&self, atom_type: &str) -> Option<f64> {
        let protons = self.intra_protons.get(atom_type)?;
        if protons.is_empty() {
            return None;
        }
        let total_weight: f64 = protons.iter().map(|p| p.weight).sum();
        if total_weight < 0.001 {
            return None;
        }
        let weighted_sum: f64 = protons.iter().map(|p| p.shift * p.weight).sum();
        Some(weighted_sum / total_weight)
    }

    /// Get weighted average shift for an inter proton atom type (HA(i-1), HB(i-1))
    pub fn get_weighted_inter_proton(&self, atom_type: &str) -> Option<f64> {
        let protons = self.inter_protons.get(atom_type)?;
        if protons.is_empty() {
            return None;
        }
        let total_weight: f64 = protons.iter().map(|p| p.weight).sum();
        if total_weight < 0.001 {
            return None;
        }
        let weighted_sum: f64 = protons.iter().map(|p| p.shift * p.weight).sum();
        Some(weighted_sum / total_weight)
    }

    /// Move center toward weighted average of contributing observations
    pub fn update_center_from_weights(&mut self, observations: &[Observation]) {
        let mut sum_h = 0.0;
        let mut sum_n = 0.0;
        let mut total_weight = 0.0;

        for (&obs_idx, &weight) in &self.observation_weights {
            if let Some((h, n)) = get_backbone_hn_from_obs(&observations[obs_idx]) {
                sum_h += h * weight;
                sum_n += n * weight;
                total_weight += weight;
            }
        }

        if total_weight > 0.01 {
            // Blend toward new center (0.3 momentum to prevent oscillation)
            let new_h = sum_h / total_weight;
            let new_n = sum_n / total_weight;
            self.h_center = 0.7 * self.h_center + 0.3 * new_h;
            self.n_center = 0.7 * self.n_center + 0.3 * new_n;
        }
    }

    /// Clear all observation contributions (for recomputation)
    pub fn clear_contributions(&mut self) {
        self.intra_carbons.clear();
        self.inter_carbons.clear();
        self.intra_protons.clear();
        self.inter_protons.clear();
        self.observation_weights.clear();
        self.total_weight = 0.0;
    }

    /// Get all observation indices contributing to this group
    pub fn get_observation_indices(&self) -> Vec<usize> {
        self.observation_weights.keys().copied().collect()
    }

    /// Convert to SpinSystemEvidence format for compatibility with typing functions.
    /// Uses weighted averages for carbon and proton shifts.
    pub fn to_spin_system_evidence(&self) -> SpinSystemEvidence {
        let mut intra_carbons: HashMap<String, f64> = HashMap::new();
        let mut inter_carbons: HashMap<String, f64> = HashMap::new();
        let mut intra_protons: HashMap<String, f64> = HashMap::new();
        let mut inter_protons: HashMap<String, f64> = HashMap::new();

        // Compute weighted average for each carbon type
        for (atom_type, _carbons) in &self.intra_carbons {
            if let Some(avg) = self.get_weighted_intra_carbon(atom_type) {
                intra_carbons.insert(atom_type.clone(), avg);
            }
        }
        for (atom_type, _carbons) in &self.inter_carbons {
            if let Some(avg) = self.get_weighted_inter_carbon(atom_type) {
                inter_carbons.insert(atom_type.clone(), avg);
            }
        }

        // Compute weighted average for each proton type
        for (atom_type, _protons) in &self.intra_protons {
            if let Some(avg) = self.get_weighted_intra_proton(atom_type) {
                intra_protons.insert(atom_type.clone(), avg);
            }
        }
        for (atom_type, _protons) in &self.inter_protons {
            if let Some(avg) = self.get_weighted_inter_proton(atom_type) {
                inter_protons.insert(atom_type.clone(), avg);
            }
        }

        SpinSystemEvidence {
            group_idx: self.group_idx,
            observation_indices: self.get_observation_indices(),
            h_shift: self.h_center,
            n_shift: self.n_center,
            intra_carbons,
            inter_carbons,
            intra_protons,
            inter_protons,
        }
    }
}

/// Helper to get backbone (H, N) from an observation
fn get_backbone_hn_from_obs(obs: &Observation) -> Option<(f64, f64)> {
    let h = obs.dimensions.iter()
        .find(|d| d.nucleus == NucleusType::H1 && d.residue_offset == ResidueOffset::Intra)
        .map(|d| d.shift)?;
    let n = obs.dimensions.iter()
        .find(|d| d.nucleus == NucleusType::N15 && d.residue_offset == ResidueOffset::Intra)
        .map(|d| d.shift)?;
    Some((h, n))
}

/// Compute how well an observation fits a group center using Gaussian likelihood
fn compute_group_fit(
    obs_h: f64, obs_n: f64,
    group_h: f64, group_n: f64,
    sigma_h: f64, sigma_n: f64,
) -> f64 {
    let d_h = (obs_h - group_h) / sigma_h;
    let d_n = (obs_n - group_n) / sigma_n;
    (-0.5 * (d_h * d_h + d_n * d_n)).exp()
}

/// Compute Mahalanobis distance between two (H, N) points.
/// This accounts for the different measurement precision of H vs N:
/// - H has ~10x better precision than N (tighter line width)
/// - Using Mahalanobis distance prevents rectangular tolerance artifacts
///
/// CRYSTALLINE-compatible: This maps to the precision matrix (inverse covariance)
/// in GaussianComponent<2> for backbone (H, N) density.
fn compute_mahalanobis_distance(
    h_a: f64, n_a: f64,
    h_b: f64, n_b: f64,
    sigma_h: f64, sigma_n: f64,
) -> f64 {
    // precision_h = 1/σ_H², precision_n = 1/σ_N² (diagonal precision matrix)
    let d_h = (h_a - h_b) / sigma_h;
    let d_n = (n_a - n_b) / sigma_n;
    (d_h * d_h + d_n * d_n).sqrt()
}

/// Find unique backbone (H, N) centers from observations using Mahalanobis distance.
/// Compute centroid of cluster members (helper for deterministic clustering)
fn compute_cluster_centroid(pairs: &[(f64, f64)], member_indices: &[usize]) -> (f64, f64) {
    let n = member_indices.len() as f64;
    let sum_h: f64 = member_indices.iter().map(|&i| pairs[i].0).sum();
    let sum_n: f64 = member_indices.iter().map(|&i| pairs[i].1).sum();
    (sum_h / n, sum_n / n)
}

/// Find unique backbone (H, N) centers using Mahalanobis distance clustering.
/// Uses covariance-aware clustering to identify distinct spin systems.
///
/// DETERMINISTIC: Results are independent of input observation order.
/// This is achieved by:
/// 1. Sorting (H, N) pairs before clustering
/// 2. Assigning to nearest centroid (not first match)
/// 3. Computing final centers as cluster averages
fn find_backbone_centers_mahalanobis(
    observations: &[Observation],
    backbone_indices: &[usize],
    sigma_h: f64,         // H measurement uncertainty (e.g., 0.015 ppm)
    sigma_n: f64,         // N measurement uncertainty (e.g., 0.15 ppm)
    d_mahal_threshold: f64,  // Merge if d_mahal < threshold (e.g., 2.0 for 2-sigma)
) -> Vec<(f64, f64)> {
    // 1. Extract all (H, N) pairs from backbone observations
    let mut hn_pairs: Vec<(f64, f64)> = backbone_indices
        .iter()
        .filter_map(|&idx| get_backbone_hn_from_obs(&observations[idx]))
        .collect();

    if hn_pairs.is_empty() {
        return Vec::new();
    }

    // 2. Sort pairs for deterministic iteration (by H first, then N)
    hn_pairs.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    });

    // 3. Assign each pair to a cluster (greedy, but deterministic due to sorting)
    let mut cluster_members: Vec<Vec<usize>> = Vec::new();

    for (i, &(h, n)) in hn_pairs.iter().enumerate() {
        // Find existing cluster within threshold (using centroid distance)
        let mut best_cluster: Option<usize> = None;
        let mut best_dist = d_mahal_threshold;

        for (cluster_idx, members) in cluster_members.iter().enumerate() {
            // Compute distance to cluster centroid
            let (c_h, c_n) = compute_cluster_centroid(&hn_pairs, members);
            let dist = compute_mahalanobis_distance(h, n, c_h, c_n, sigma_h, sigma_n);
            if dist < best_dist {
                best_dist = dist;
                best_cluster = Some(cluster_idx);
            }
        }

        match best_cluster {
            Some(cluster_idx) => {
                cluster_members[cluster_idx].push(i);
            }
            None => {
                // Create new cluster
                cluster_members.push(vec![i]);
            }
        }
    }

    // 4. Compute final centers as cluster centroids (order-independent)
    cluster_members
        .iter()
        .map(|members| compute_cluster_centroid(&hn_pairs, members))
        .collect()
}

/// Legacy function for backward compatibility - uses hard rectangular tolerance
/// DETERMINISTIC: Results are independent of input observation order.
fn find_backbone_centers(
    observations: &[Observation],
    backbone_indices: &[usize],
    h_tolerance: f64,
    n_tolerance: f64,
) -> Vec<(f64, f64)> {
    // 1. Extract all (H, N) pairs from backbone observations
    let mut hn_pairs: Vec<(f64, f64)> = backbone_indices
        .iter()
        .filter_map(|&idx| get_backbone_hn_from_obs(&observations[idx]))
        .collect();

    if hn_pairs.is_empty() {
        return Vec::new();
    }

    // 2. Sort pairs for deterministic iteration (by H first, then N)
    hn_pairs.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    });

    // 3. Assign each pair to a cluster (greedy, but deterministic due to sorting)
    let mut cluster_members: Vec<Vec<usize>> = Vec::new();

    for (i, &(h, n)) in hn_pairs.iter().enumerate() {
        // Find existing cluster within tolerance (using centroid distance)
        let mut best_cluster: Option<usize> = None;

        for (cluster_idx, members) in cluster_members.iter().enumerate() {
            let (c_h, c_n) = compute_cluster_centroid(&hn_pairs, members);
            if (h - c_h).abs() < h_tolerance && (n - c_n).abs() < n_tolerance {
                best_cluster = Some(cluster_idx);
                break; // Take first match for rectangular tolerance
            }
        }

        match best_cluster {
            Some(cluster_idx) => {
                cluster_members[cluster_idx].push(i);
            }
            None => {
                cluster_members.push(vec![i]);
            }
        }
    }

    // 4. Compute final centers as cluster centroids
    cluster_members
        .iter()
        .map(|members| compute_cluster_centroid(&hn_pairs, members))
        .collect()
}

/// Create soft backbone groups with weighted observation contributions.
/// Observations can contribute to multiple groups based on fit quality.
///
/// CRYSTALLINE-compatible: This implements GMM-style soft clustering where:
/// - Centers are detected using Mahalanobis distance (covariance-aware)
/// - Observations contribute to groups with Gaussian likelihood weights (responsibilities)
/// - Multiple groups can claim partial ownership of ambiguous observations
fn create_soft_backbone_groups(
    observations: &[Observation],
    backbone_indices: &[usize],
    sigma_h: f64,  // Gaussian width for H (for soft weighting)
    sigma_n: f64,  // Gaussian width for N (for soft weighting)
    fit_threshold: f64,  // Minimum fit quality to contribute (e.g., 0.01)
) -> Vec<SoftSpinSystemEvidence> {
    // 1. Find unique backbone centers using MAHALANOBIS distance
    // This properly accounts for the different precision of H vs N:
    // - H line width ~0.015 ppm (10x more discriminating)
    // - N line width ~0.15 ppm
    // A 2-sigma threshold means d_mahal < 2.0 to merge
    let d_mahal_threshold = 2.0;  // 2-sigma ellipse
    let centers = find_backbone_centers_mahalanobis(
        observations, backbone_indices, sigma_h, sigma_n, d_mahal_threshold
    );

    // 2. Create soft groups for each center
    let mut groups: Vec<SoftSpinSystemEvidence> = centers.iter()
        .enumerate()
        .map(|(idx, (h, n))| SoftSpinSystemEvidence::new(idx, *h, *n))
        .collect();

    // 3. Assign observations to groups with weighted contribution (GMM responsibilities)
    for &obs_idx in backbone_indices {
        let Some((obs_h, obs_n)) = get_backbone_hn_from_obs(&observations[obs_idx]) else { continue };

        for group in &mut groups {
            let fit = compute_group_fit(obs_h, obs_n, group.h_center, group.n_center, sigma_h, sigma_n);
            if fit >= fit_threshold {
                group.add_observation(obs_idx, &observations[obs_idx], fit);
            }
        }
    }

    groups
}

/// Recompute observation weights for all groups with new tolerances.
/// Called during BP as tolerances anneal.
fn recompute_group_weights(
    groups: &mut [SoftSpinSystemEvidence],
    observations: &[Observation],
    backbone_indices: &[usize],
    sigma_h: f64,
    sigma_n: f64,
    fit_threshold: f64,
) {
    // Clear existing contributions
    for group in groups.iter_mut() {
        group.clear_contributions();
    }

    // Recompute with new tolerances
    for &obs_idx in backbone_indices {
        let Some((obs_h, obs_n)) = get_backbone_hn_from_obs(&observations[obs_idx]) else { continue };

        for group in groups.iter_mut() {
            let fit = compute_group_fit(obs_h, obs_n, group.h_center, group.n_center, sigma_h, sigma_n);
            if fit >= fit_threshold {
                group.add_observation(obs_idx, &observations[obs_idx], fit);
            }
        }
    }
}

/// Detect groups with multimodal INTRA CA distributions and split them.
/// Only splits when multiple INTRA CAs disagree (not intra vs inter, which is normal).
fn detect_and_split_multimodal_groups(
    groups: &mut Vec<SoftSpinSystemEvidence>,
    observations: &[Observation],
    min_ca_separation: f64,  // e.g., 2.0 ppm for same-residue ambiguity
    verbose: bool,
) {
    let mut groups_to_add: Vec<SoftSpinSystemEvidence> = Vec::new();
    let mut groups_to_remove: Vec<usize> = Vec::new();

    for (group_idx, group) in groups.iter().enumerate() {
        // Get INTRA CA shifts only (inter CAs are from different residues, so variance is expected)
        let Some(ca_carbons) = group.intra_carbons.get("CA") else { continue };
        if ca_carbons.len() < 2 {
            continue;
        }

        // Separate TRUE intra observations (intensity > 0.5) from inter observations
        let intra_shifts: Vec<(f64, usize, f64)> = ca_carbons.iter()
            .filter(|wc| observations[wc.obs_idx].intensity > 0.5)
            .map(|wc| (wc.shift, wc.obs_idx, wc.weight))
            .collect();

        // Also collect INTER CA shifts (intensity <= 0.5) - needed for matching inter-CB
        let inter_shifts: Vec<(f64, usize, f64)> = ca_carbons.iter()
            .filter(|wc| observations[wc.obs_idx].intensity <= 0.5)
            .map(|wc| (wc.shift, wc.obs_idx, wc.weight))
            .collect();

        if intra_shifts.len() < 2 {
            continue;  // Not enough intra CAs to detect bimodality
        }

        // Check if INTRA CA distribution is bimodal
        let (min_shift, max_shift) = intra_shifts.iter()
            .fold((f64::MAX, f64::MIN), |(min, max), &(s, _, _)| (min.min(s), max.max(s)));

        if max_shift - min_shift >= min_ca_separation {
            // Bimodal INTRA CAs! This suggests two different residues wrongly merged
            let midpoint = (min_shift + max_shift) / 2.0;

            if verbose {
                println!("  Bimodal CA detected in group {}: intra_shifts={}, inter_shifts={}",
                    group_idx, intra_shifts.len(), inter_shifts.len());
                for (shift, obs_idx, _w) in &intra_shifts {
                    let obs = &observations[*obs_idx];
                    println!("    intra CA={:.2} (obs {}, intensity={:.2})", shift, obs_idx, obs.intensity);
                }
                for (shift, obs_idx, _w) in &inter_shifts {
                    let obs = &observations[*obs_idx];
                    println!("    inter CA={:.2} (obs {}, intensity={:.2})", shift, obs_idx, obs.intensity);
                }
            }

            // Collect observations for each cluster
            // First, determine which CA cluster each CA observation belongs to
            let mut ca_obs_to_cluster: HashMap<usize, bool> = HashMap::new(); // true = high, false = low

            for &(shift, obs_idx, _weight) in &intra_shifts {
                ca_obs_to_cluster.insert(obs_idx, shift >= midpoint);
            }

            // Now, assign ALL observations to clusters based on their CA association
            // For non-CA observations (CB, C), find the closest CA observation with same (H, N)
            let mut low_obs: Vec<(usize, f64)> = Vec::new();
            let mut high_obs: Vec<(usize, f64)> = Vec::new();

            for (&obs_idx, &weight) in &group.observation_weights {
                let obs = &observations[obs_idx];

                // Check if this observation has CA
                let obs_ca: Option<f64> = obs.dimensions.iter()
                    .find(|d| d.nucleus == NucleusType::C13 &&
                        (d.atom_constraint == AtomConstraint::Exact("CA".to_string()) ||
                         d.atom_hint.as_deref() == Some("CA")))
                    .map(|d| d.shift);

                if let Some(ca_shift) = obs_ca {
                    // This observation has CA - assign based on CA shift
                    if ca_shift >= midpoint {
                        high_obs.push((obs_idx, weight));
                    } else {
                        low_obs.push((obs_idx, weight));
                    }
                } else {
                    // This observation doesn't have CA (e.g., CB or C only)
                    // Use KDE scoring to determine which cluster this carbon belongs to

                    // Get the carbon shift and type from this observation
                    let carbon_info: Option<(f64, String)> = obs.dimensions.iter()
                        .find(|d| d.nucleus == NucleusType::C13)
                        .map(|d| {
                            let atom_type = match &d.atom_constraint {
                                AtomConstraint::Exact(s) => s.clone(),
                                AtomConstraint::OneOf(v) if v.len() == 1 => v[0].clone(),
                                _ => d.atom_hint.clone().unwrap_or_else(|| "C".to_string()),
                            };
                            (d.shift, atom_type)
                        });

                    let mut assigned_high = low_obs.len() <= high_obs.len(); // Default fallback

                    if let Some((c_shift, atom_type)) = carbon_info {
                        // For CB/C: Match to a CA cluster using KDE joint scoring
                        // The intra_shifts define the clusters (low CA vs high CA)
                        // We use KDE to find which residue type best explains this CB+CA combination
                        let obs_is_intra = obs.intensity > 0.5;

                        // ALWAYS use intra_shifts for KDE matching since those define the clusters
                        // (The bimodal split is based on intra-CA, so we match against those)
                        let ca_shifts_to_use = &intra_shifts;

                        if verbose && (atom_type == "CB" || atom_type == "C") {
                            println!("    Processing {}={:.2} (obs {}, intra={}): ca_set_size={}",
                                atom_type, c_shift, obs_idx, obs_is_intra, ca_shifts_to_use.len());
                        }

                        // Find CA observations to match against
                        let matching_cas: Vec<(f64, bool)> = ca_shifts_to_use.iter()
                            .map(|(ca_shift, _, _)| (*ca_shift, *ca_shift >= midpoint))
                            .collect();

                        if !matching_cas.is_empty() {
                            // Use KDE to find which CA's residue type best explains this CB
                            let kde = KDEScorer::new();
                            let residue_types = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU",
                                "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO",
                                "SER", "THR", "TRP", "TYR", "VAL"];

                            let mut best_joint_score = 0.0_f64;
                            let mut best_is_high = false;
                            let mut best_res = "";

                            for (ca_shift, is_high) in &matching_cas {
                                for res in &residue_types {
                                    let cb_score = kde.score(res, &atom_type, c_shift);
                                    let ca_score = kde.score(res, "CA", *ca_shift);
                                    let joint = (cb_score * ca_score).sqrt();

                                    if joint > best_joint_score {
                                        best_joint_score = joint;
                                        best_is_high = *is_high;
                                        best_res = res;
                                    }
                                }
                            }

                            if verbose && atom_type == "CB" {
                                println!("    CB={:.2} (intra={}): best_res={} joint={:.4} -> {}",
                                    c_shift, obs_is_intra, best_res, best_joint_score,
                                    if best_is_high { "HIGH" } else { "LOW" });
                            }

                            assigned_high = best_is_high;
                        }
                    }

                    if assigned_high {
                        high_obs.push((obs_idx, weight));
                    } else {
                        low_obs.push((obs_idx, weight));
                    }
                }
            }

            // Only split if both clusters have observations
            if !low_obs.is_empty() && !high_obs.is_empty() {
                if verbose {
                    println!("  Splitting group {} (INTRA CA bimodal): {:.1}-{:.1} ppm ({} low, {} high)",
                        group_idx, min_shift, max_shift, low_obs.len(), high_obs.len());
                }

                // Compute new centers for each cluster
                let compute_cluster_center = |obs_list: &[(usize, f64)]| -> (f64, f64) {
                    let mut sum_h = 0.0;
                    let mut sum_n = 0.0;
                    let mut total_w = 0.0;
                    for &(obs_idx, w) in obs_list {
                        if let Some((h, n)) = get_backbone_hn_from_obs(&observations[obs_idx]) {
                            sum_h += h * w;
                            sum_n += n * w;
                            total_w += w;
                        }
                    }
                    if total_w > 0.001 {
                        (sum_h / total_w, sum_n / total_w)
                    } else {
                        (group.h_center, group.n_center)
                    }
                };

                let (h_low, n_low) = compute_cluster_center(&low_obs);
                let (h_high, n_high) = compute_cluster_center(&high_obs);

                // Create two new groups
                let new_idx_low = groups.len() + groups_to_add.len();
                let new_idx_high = new_idx_low + 1;

                let mut group_low = SoftSpinSystemEvidence::new(new_idx_low, h_low, n_low);
                let mut group_high = SoftSpinSystemEvidence::new(new_idx_high, h_high, n_high);

                for (obs_idx, weight) in low_obs {
                    group_low.add_observation(obs_idx, &observations[obs_idx], weight);
                }
                for (obs_idx, weight) in high_obs {
                    group_high.add_observation(obs_idx, &observations[obs_idx], weight);
                }

                groups_to_add.push(group_low);
                groups_to_add.push(group_high);
                groups_to_remove.push(group_idx);
            }
        }
    }

    // Remove split groups (in reverse order to preserve indices)
    groups_to_remove.sort();
    for &idx in groups_to_remove.iter().rev() {
        groups.remove(idx);
    }

    // Re-index remaining groups
    for (i, group) in groups.iter_mut().enumerate() {
        group.group_idx = i;
    }

    // Add new groups
    let base_idx = groups.len();
    for (i, mut new_group) in groups_to_add.into_iter().enumerate() {
        new_group.group_idx = base_idx + i;
        groups.push(new_group);
    }
}

/// Compute the probability that a group contains overlapping spin systems (K >= 2)
/// using Bayesian inference on observation count and carbon variance.
///
/// Mathematical model:
/// - P(n | K=k) = Poisson(n; k * lambda)  where lambda = expected_obs_per_system
/// - P(K=2 | n) = BF * prior_k2 / (BF * prior_k2 + prior_k1)
/// - BF = P(n | K=2) / P(n | K=1) = 2^n * e^(-lambda)
///
/// CRYSTALLINE-compatible: This is a simplified version of Dirichlet Process
/// mixture model multiplicity detection.
fn compute_overlap_probability(
    observed_count: usize,
    expected_per_system: f64,
    intra_ca_values: &[f64],
    prior_k2: f64,  // Prior probability of overlap (e.g., 0.05)
) -> f64 {
    let n = observed_count as f64;
    let lambda = expected_per_system;

    // 1. Count-based Bayes factor: BF = 2^n * e^(-lambda)
    // This captures: "too many observations for a single spin system"
    let bf_count = if lambda > 0.0 {
        2.0_f64.powf(n) * (-lambda).exp()
    } else {
        1.0
    };

    // 2. Variance-based Bayes factor
    // Under single spin system: all intra-CA should be identical (measurement error only)
    // Under overlap: different residues have different CA values
    let bf_variance = if intra_ca_values.len() >= 2 {
        let mean: f64 = intra_ca_values.iter().sum::<f64>() / intra_ca_values.len() as f64;
        let variance: f64 = intra_ca_values.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / (intra_ca_values.len() - 1) as f64;

        // Expected variance for single spin system: ~0.04 ppm² (0.2 ppm measurement error)
        // Expected variance for overlap: ~16 ppm² (4 ppm inter-residue difference)
        let expected_single_var = 0.04;
        let expected_overlap_var = 16.0;

        // Likelihood ratio using Gaussian model for variance
        let ll_overlap = (-variance / (2.0 * expected_overlap_var)).exp();
        let ll_single = (-variance / (2.0 * expected_single_var)).exp();

        if ll_single > 1e-10 {
            ll_overlap / ll_single
        } else {
            100.0  // Strong evidence for overlap if single hypothesis has near-zero likelihood
        }
    } else {
        1.0  // No variance evidence
    };

    // 3. Gap-based Bayes factor (bimodality)
    let bf_gap = if intra_ca_values.len() >= 2 {
        let mut sorted = intra_ca_values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let max_gap = sorted.windows(2)
            .map(|w| w[1] - w[0])
            .fold(0.0_f64, |a, b| a.max(b));

        if max_gap > 3.0 { 20.0 }      // Very strong evidence (>3 ppm gap)
        else if max_gap > 2.0 { 10.0 } // Strong evidence (>2 ppm gap)
        else if max_gap > 1.0 { 2.0 }  // Moderate evidence
        else { 0.5 }                    // Evidence against overlap
    } else {
        1.0
    };

    // Combined Bayes factor
    let bf_combined = bf_count * bf_variance * bf_gap;
    let prior_k1 = 1.0 - prior_k2;

    // Posterior probability: P(K=2 | evidence)
    (bf_combined * prior_k2) / (bf_combined * prior_k2 + prior_k1)
}

/// Detect groups with high probability of containing overlapping spin systems.
/// Uses observation count as primary evidence, combined with carbon variance.
///
/// This implements Option B from the plan: detect after initial grouping,
/// flag for splitting, and let BP naturally resolve via soft grouping factors.
fn detect_probable_overlaps(
    groups: &[SoftSpinSystemEvidence],
    observations: &[Observation],
    expected_obs_per_system: f64,  // e.g., 6.0 for HNCACB + HNCACO
    overlap_threshold: f64,        // e.g., 0.3 (30% probability)
    verbose: bool,
) -> Vec<(usize, f64)> {  // (group_idx, overlap_probability)
    let mut overlaps = Vec::new();

    for (group_idx, group) in groups.iter().enumerate() {
        let observed_count = group.observation_weights.len();

        // Get intra-CA values for variance calculation
        let intra_ca_values: Vec<f64> = group.intra_carbons.get("CA")
            .map(|cas| cas.iter()
                .filter(|wc| observations[wc.obs_idx].intensity > 0.5)  // True intra only
                .map(|wc| wc.shift)
                .collect())
            .unwrap_or_default();

        let p_overlap = compute_overlap_probability(
            observed_count,
            expected_obs_per_system,
            &intra_ca_values,
            0.05,  // Prior: 5% of groups have overlap
        );

        if p_overlap > overlap_threshold {
            if verbose {
                println!("  Group {} (H={:.3}, N={:.2}): P(overlap)={:.2} (n={}, expected={:.0}, CA_count={})",
                    group_idx, group.h_center, group.n_center,
                    p_overlap, observed_count, expected_obs_per_system, intra_ca_values.len());
            }
            overlaps.push((group_idx, p_overlap));
        }
    }

    overlaps
}

/// Sequential link between two spin systems at the group level.
#[derive(Debug, Clone)]
pub struct GroupSequentialLink {
    /// Index of preceding group (observes inter carbons)
    pub from_group: usize,
    /// Index of following group (observes intra carbons matching from's inter)
    pub to_group: usize,
    /// Base match strength (computed at discovery time)
    pub strength: f64,
    /// Which carbons matched
    pub matched_atoms: Vec<String>,
    /// Average absolute shift difference (ppm) - used for adaptive filtering
    pub avg_shift_diff: f64,
    /// Maximum shift difference among matched atoms (ppm)
    pub max_shift_diff: f64,
}

/// Run group-level belief propagation.
/// This is the simplified physics-based approach:
/// - 66 variables (groups) instead of 586 (observations)
/// - Typing factor from carbons + (H, N)
/// - Sequential factor from inter/intra carbon matching
/// - Uniqueness enforced at extraction time
pub fn run_group_level_bp(
    groups: &[SpinSystemEvidence],
    sequential_links: &[GroupSequentialLink],
    residue_types: &[String],
    sequence: &str,  // For path uniqueness scoring
    kde: &KDEDatabase,
    params: &UnifiedAssignmentParams,
) -> Vec<(usize, i32, f64)> {  // (group_idx, position, confidence)
    let n_groups = groups.len();
    let domain_size = residue_types.len() + 1;  // 0 = unassigned, 1..N = positions

    if n_groups == 0 {
        return vec![];
    }

    // Compute typing scores for each group
    let typing_scores = compute_group_typing_scores(groups, residue_types, kde);

    // Debug: show best typing scores for all groups - focus on GLY detection
    if params.verbose {
        println!("\n--- Typing scores (before BP) - GLY check ---");

        // Find which positions are GLY in the sequence
        let gly_positions: Vec<usize> = residue_types.iter().enumerate()
            .filter(|(_, t)| *t == "GLY")
            .map(|(i, _)| i)
            .collect();
        println!("GLY positions in sequence: {:?}", gly_positions.iter().map(|p| p + 1).collect::<Vec<_>>());

        // Check each group's GLY score
        let mut gly_groups: Vec<(usize, f64)> = Vec::new();
        for (g, scores) in typing_scores.iter().enumerate() {
            // Sum scores for GLY positions
            let gly_score: f64 = gly_positions.iter().map(|&p| scores[p]).sum();
            let total: f64 = scores.iter().sum();
            let gly_fraction = if total > 0.0 { gly_score / total } else { 0.0 };

            if gly_fraction > 0.3 {  // >30% GLY
                gly_groups.push((g, gly_fraction));
            }
        }
        gly_groups.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        println!("Groups with >30% GLY typing ({} total):", gly_groups.len());
        for (g, frac) in gly_groups.iter().take(10) {
            let group = &groups[*g];
            let ca = group.intra_carbons.get("CA").unwrap_or(&0.0);
            let cb = group.intra_carbons.get("CB");
            println!("  Group {}: {:.1}% GLY, CA={:.1}, CB={:?}", g, frac * 100.0, ca, cb);
        }
        println!("---");
    }

    // Initialize beliefs from typing scores
    let mut beliefs: Vec<Vec<f64>> = typing_scores.iter()
        .map(|scores| {
            let mut b = vec![0.0; domain_size];
            // Small prior for position 0 (unassigned)
            b[0] = 0.01;
            for (pos, &score) in scores.iter().enumerate() {
                b[pos + 1] = score.max(1e-10);
            }
            // Normalize
            let sum: f64 = b.iter().sum();
            if sum > 0.0 {
                for v in b.iter_mut() {
                    *v /= sum;
                }
            }
            b
        })
        .collect();

    // Build link lookup: for each group, which links involve it?
    let mut links_to: Vec<Vec<usize>> = vec![vec![]; n_groups];  // links where group is 'to'
    let mut links_from: Vec<Vec<usize>> = vec![vec![]; n_groups]; // links where group is 'from'
    for (link_idx, link) in sequential_links.iter().enumerate() {
        links_to[link.to_group].push(link_idx);
        links_from[link.from_group].push(link_idx);
    }

    let max_iterations = params.max_iterations;
    let damping = 0.3;  // Damping factor for stability

    // Adaptive tolerance parameters: start broad, converge to tight
    let tol_initial = params.triple_res_c_tolerance_initial.max(1.0);  // Broad (1+ ppm)
    let tol_final = params.triple_res_c_tolerance_final.max(0.1);      // Tight (0.1-0.2 ppm)

    if params.verbose {
        println!("\n=== GROUP-LEVEL BP ===");
        println!("Groups: {}, Positions: {}", n_groups, domain_size - 1);
        println!("Sequential links: {}", sequential_links.len());
        println!("Adaptive tolerance: {:.2} -> {:.2} ppm", tol_initial, tol_final);
    }

    for iteration in 0..max_iterations {
        let mut new_beliefs = vec![vec![0.0; domain_size]; n_groups];
        let mut max_delta = 0.0f64;

        // Compute current tolerance using exponential decay (fast initial narrowing)
        let progress = iteration as f64 / max_iterations as f64;
        let t = 1.0 - (-3.0 * progress).exp();  // Exponential approach to 1
        let current_tol = tol_initial * (1.0 - t) + tol_final * t;

        // Count active links at current tolerance (for debugging)
        let mut active_links = 0usize;

        for g in 0..n_groups {
            // Strategy: compute typing and sequential messages in PROBABILITY space,
            // then combine them multiplicatively.

            // Factor 1: Typing (unary) - directly use typing scores (already normalized)
            let mut typing_msg = vec![1e-10; domain_size];
            typing_msg[0] = 0.01;  // Small prior for unassigned
            for pos in 1..domain_size {
                typing_msg[pos] = typing_scores[g][pos - 1].max(1e-10);
            }

            // Factor 2: Sequential (aggregate messages from linked groups)
            // Message says: "based on my neighbors, position pos is likely"
            let mut seq_msg_incoming = vec![1.0; domain_size];  // Uniform prior
            let mut seq_msg_outgoing = vec![1.0; domain_size];
            let mut has_incoming_links = false;
            let mut has_outgoing_links = false;

            // Incoming links: if 'from' is at pos i, I should be at pos i+1
            // Link strength is log(LR) from Bayesian scoring - use exp(strength) as multiplier
            for &link_idx in &links_to[g] {
                let link = &sequential_links[link_idx];
                // Links were already filtered by LR threshold during construction
                // Use the strength directly (it's log(LR))
                active_links += 1;
                has_incoming_links = true;

                let from_g = link.from_group;
                // strength = log(LR), so exp(strength) = LR
                // Scale down for numerical stability: use strength directly as multiplier
                let lr_factor = link.strength;  // log(LR), typically 2.94-8.83

                for pos in 2..domain_size {
                    let from_prob = beliefs[from_g][pos - 1];
                    // Boost belief if predecessor is likely at pos-1
                    seq_msg_incoming[pos] *= 1.0 + lr_factor * from_prob;
                }
            }

            // Outgoing links: if 'to' is at pos i+1, I should be at pos i
            for &link_idx in &links_from[g] {
                let link = &sequential_links[link_idx];
                has_outgoing_links = true;

                let to_g = link.to_group;
                let lr_factor = link.strength;

                for pos in 1..(domain_size - 1) {
                    let to_prob = beliefs[to_g][pos + 1];
                    seq_msg_outgoing[pos] *= 1.0 + lr_factor * to_prob;
                }
            }

            // Normalize sequential messages
            if has_incoming_links {
                let sum: f64 = seq_msg_incoming.iter().sum();
                if sum > 0.0 {
                    for v in seq_msg_incoming.iter_mut() { *v /= sum; }
                }
            }
            if has_outgoing_links {
                let sum: f64 = seq_msg_outgoing.iter().sum();
                if sum > 0.0 {
                    for v in seq_msg_outgoing.iter_mut() { *v /= sum; }
                }
            }

            // Combine: typing * sequential_in * sequential_out
            // Balance typing and sequential evidence
            let typing_weight = 2.0;  // Typing provides residue type information
            let seq_weight = 2.0;    // Sequential links connect groups
            let _has_links = has_incoming_links || has_outgoing_links;

            for pos in 0..domain_size {
                let t = typing_msg[pos].powf(typing_weight);
                let s_in = if has_incoming_links { seq_msg_incoming[pos] } else { 1.0 };
                let s_out = if has_outgoing_links { seq_msg_outgoing[pos] } else { 1.0 };
                new_beliefs[g][pos] = t * s_in.powf(seq_weight) * s_out.powf(seq_weight);
            }

            // Normalize beliefs (we're in probability space, not log space)
            let sum: f64 = new_beliefs[g].iter().sum();
            if sum > 0.0 {
                for v in new_beliefs[g].iter_mut() {
                    *v /= sum;
                }
            }

            // Apply damping
            for pos in 0..domain_size {
                let old = beliefs[g][pos];
                let new = new_beliefs[g][pos];
                new_beliefs[g][pos] = damping * old + (1.0 - damping) * new;
                max_delta = max_delta.max((old - new_beliefs[g][pos]).abs());
            }
        }

        beliefs = new_beliefs;

        // Check convergence
        if iteration > 0 && max_delta < 0.001 {
            if params.verbose {
                println!("Converged at iteration {} (max_delta={:.4})", iteration, max_delta);
            }
            break;
        }

        if params.verbose && (iteration < 5 || iteration % 20 == 0) {
            println!("Iteration {}: max_delta={:.4}, active_links={}", iteration, max_delta, active_links / 2);
        }
    }

    // Debug: show beliefs after BP for first few groups
    if params.verbose {
        println!("\n--- Beliefs (after BP) ---");
        for (g, b) in beliefs.iter().enumerate().take(5) {
            let (best_pos, &best_belief) = b.iter().enumerate()
                .skip(1)  // Skip position 0
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap();
            let res_type = if best_pos > 0 && best_pos <= residue_types.len() {
                &residue_types[best_pos - 1]
            } else { "?" };
            println!("  Group {}: best pos {} ({}) with belief {:.4}", g, best_pos, res_type, best_belief);
        }
        println!("---");
    }

    // POST-BP REFINEMENT: Apply path uniqueness scoring
    // This boosts positions where the implied sequential path is unique in the sequence
    if params.verbose {
        println!("\n--- Applying path uniqueness refinement ---");
    }

    let mut uniqueness_adjusted = 0usize;
    let mut debug_paths_shown = 0usize;
    for g in 0..n_groups {
        // Only refine groups that have sequential neighbors
        let has_links = !links_from[g].is_empty() || !links_to[g].is_empty();
        if !has_links {
            continue;
        }

        // Compute uniqueness for each position
        let mut best_original_pos = 0;
        let mut best_original_belief = 0.0f64;
        for pos in 1..domain_size {
            if beliefs[g][pos] > best_original_belief {
                best_original_belief = beliefs[g][pos];
                best_original_pos = pos;
            }
        }

        // Compute path uniqueness for top candidate positions
        let mut pos_scores: Vec<(usize, f64)> = beliefs[g].iter().enumerate()
            .skip(1)
            .filter(|(_, &b)| b > 0.01)  // Only consider positions with some belief
            .map(|(pos, &b)| {
                let u = compute_path_uniqueness_for_position(
                    g, pos, &beliefs, &typing_scores,
                    &links_from, &links_to, sequential_links,
                    residue_types, sequence, 3  // window of 3 residues each direction
                );
                (pos, b * u)  // Multiply belief by uniqueness factor
            })
            .collect();

        // Debug: show uniqueness factors for first few groups
        if params.verbose && debug_paths_shown < 5 && pos_scores.len() > 1 {
            debug_paths_shown += 1;
            println!("  Group {} (best={}, belief={:.3}): {} candidate positions",
                g, best_original_pos, best_original_belief, pos_scores.len());
            for (pos, score) in pos_scores.iter().take(3) {
                let orig_belief = beliefs[g][*pos];
                let uniqueness = if orig_belief > 0.0 { score / orig_belief } else { 1.0 };
                let res_type = if *pos <= residue_types.len() { &residue_types[*pos - 1] } else { "?" };
                println!("    pos {} ({}): belief={:.3}, u={:.3}, final={:.3}",
                    pos, res_type, orig_belief, uniqueness, score);
            }
        }

        pos_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Update beliefs with uniqueness-adjusted scores
        let sum: f64 = pos_scores.iter().map(|(_, s)| s).sum::<f64>().max(1e-10);
        for pos in 1..domain_size {
            if let Some((_, adjusted)) = pos_scores.iter().find(|(p, _)| *p == pos) {
                beliefs[g][pos] = *adjusted / sum;
            } else {
                beliefs[g][pos] = 0.001 / sum;  // Very low probability
            }
        }

        // Check if best position changed
        let mut new_best_pos = 0;
        let mut new_best_belief = 0.0f64;
        for pos in 1..domain_size {
            if beliefs[g][pos] > new_best_belief {
                new_best_belief = beliefs[g][pos];
                new_best_pos = pos;
            }
        }

        if new_best_pos != best_original_pos {
            uniqueness_adjusted += 1;
            if params.verbose && uniqueness_adjusted <= 10 {
                let old_type = if best_original_pos <= residue_types.len() {
                    &residue_types[best_original_pos - 1]
                } else { "?" };
                let new_type = if new_best_pos <= residue_types.len() {
                    &residue_types[new_best_pos - 1]
                } else { "?" };
                println!("  Group {}: {} (pos {}) -> {} (pos {}) via uniqueness",
                    g, old_type, best_original_pos, new_type, new_best_pos);
            }
        }
    }

    if params.verbose {
        println!("Path uniqueness adjusted {} groups", uniqueness_adjusted);
        println!("---");
    }

    // Extract assignments using greedy approach (highest confidence first)
    extract_group_assignments(&beliefs, groups, residue_types, params.verbose)
}

// =============================================================================
// Sequential Path Uniqueness Scoring
// =============================================================================

/// Convert 3-letter amino acid code to 1-letter code.
fn three_to_one(three: &str) -> char {
    match three {
        "GLY" => 'G', "ALA" => 'A', "VAL" => 'V', "LEU" => 'L',
        "ILE" => 'I', "PRO" => 'P', "PHE" => 'F', "TYR" => 'Y',
        "TRP" => 'W', "SER" => 'S', "THR" => 'T', "CYS" => 'C',
        "MET" => 'M', "ASN" => 'N', "GLN" => 'Q', "ASP" => 'D',
        "GLU" => 'E', "LYS" => 'K', "ARG" => 'R', "HIS" => 'H',
        _ => '?',
    }
}

/// Count how many times a type pattern appears in the sequence.
/// Returns (n_matches, starting positions) where positions are 1-indexed.
fn count_pattern_in_sequence(pattern: &[String], sequence: &str) -> (usize, Vec<usize>) {
    if pattern.is_empty() || pattern.len() > sequence.len() {
        return (0, Vec::new());
    }

    let seq_chars: Vec<char> = sequence.chars().collect();
    let pattern_chars: Vec<char> = pattern.iter().map(|s| three_to_one(s)).collect();
    let mut matches = Vec::new();

    for start in 0..=(seq_chars.len() - pattern.len()) {
        let mut all_match = true;
        for (i, &pat_char) in pattern_chars.iter().enumerate() {
            if pat_char == '?' || seq_chars[start + i] != pat_char {
                all_match = false;
                break;
            }
        }
        if all_match {
            matches.push(start + 1);  // 1-indexed positions
        }
    }
    (matches.len(), matches)
}

/// Score a sequential path by its uniqueness in the sequence.
/// Returns a score: 1.0 for unique, lower for ambiguous, 0.0 for non-existent.
fn score_path_uniqueness(path_types: &[String], sequence: &str) -> f64 {
    if path_types.is_empty() {
        return 0.0;
    }

    let (n_matches, _positions) = count_pattern_in_sequence(path_types, sequence);

    if n_matches == 0 {
        return 0.0;  // Pattern doesn't exist in sequence (typing error)
    } else if n_matches == 1 {
        return 1.0;  // Unique! Strong evidence
    } else {
        // Multiple matches - ambiguous
        // Longer paths with few matches are still valuable
        let length_bonus = (path_types.len() as f64).sqrt() / 3.0;
        length_bonus / (n_matches as f64)
    }
}

/// For a group at a candidate position, compute how well the sequential neighborhood
/// typing distributions match the sequence context.
///
/// This is PROBABILISTIC: instead of taking the best type, we compute the probability
/// that each neighbor's typing is consistent with the sequence position it would occupy.
///
/// Key insight: GLY (no CB), ALA (characteristic CB), and PRO boundaries are strong anchors.
fn compute_path_uniqueness_for_position(
    group_idx: usize,
    candidate_pos: usize,
    beliefs: &[Vec<f64>],
    typing_scores: &[Vec<f64>],
    links_from: &[Vec<usize>],  // group -> outgoing link indices
    links_to: &[Vec<usize>],    // group -> incoming link indices
    sequential_links: &[GroupSequentialLink],
    residue_types: &[String],
    sequence: &str,
    window: usize,
) -> f64 {
    let seq_chars: Vec<char> = sequence.chars().collect();
    if candidate_pos == 0 || candidate_pos > seq_chars.len() {
        return 1.0;
    }

    // Collect neighbor groups and their implied positions
    // Each neighbor contributes: P(neighbor typing matches sequence at implied position)
    let mut match_probability = 1.0f64;
    let mut n_neighbors = 0usize;
    let mut visited: HashSet<usize> = HashSet::new();
    visited.insert(group_idx);

    // Check this group's typing against sequence
    let seq_type_here = &seq_chars[candidate_pos - 1];
    let my_match_prob = typing_probability_for_sequence_char(
        &typing_scores[group_idx], residue_types, *seq_type_here
    );
    match_probability *= my_match_prob.max(0.01);  // Floor to avoid zero

    // Walk backward through predecessors
    let mut current_group = group_idx;
    let mut current_seq_pos = candidate_pos as i32;
    for _ in 0..window {
        current_seq_pos -= 1;
        if current_seq_pos < 1 {
            break;
        }

        // Check for PRO at this position - it's a boundary (no NH)
        let seq_char_here = seq_chars[current_seq_pos as usize - 1];
        if seq_char_here == 'P' {
            // PRO boundary - if we have no predecessor link here, that's GOOD (consistent)
            // If we DO have a link, that's inconsistent (PRO shouldn't have backbone NH observation)
            let has_pred_link = links_to[current_group].iter().any(|&idx| {
                !visited.contains(&sequential_links[idx].from_group)
            });
            if !has_pred_link {
                // Correct! Chain ends at PRO as expected
                match_probability *= 2.0;  // Bonus for correct PRO boundary
            }
            break;  // Stop walking - PRO is a boundary
        }

        // Find best predecessor link
        let mut best_pred: Option<(usize, f64)> = None;
        for &link_idx in &links_to[current_group] {
            let link = &sequential_links[link_idx];
            if visited.contains(&link.from_group) {
                continue;
            }
            let score = link.strength;
            if best_pred.is_none() || score > best_pred.unwrap().1 {
                best_pred = Some((link.from_group, score));
            }
        }

        if let Some((pred_group, _)) = best_pred {
            visited.insert(pred_group);
            n_neighbors += 1;

            // How well does predecessor's typing match sequence at this position?
            let pred_match = typing_probability_for_sequence_char(
                &typing_scores[pred_group], residue_types, seq_char_here
            );

            // GLY and ALA are distinctive - weight them more
            let weight = if seq_char_here == 'G' || seq_char_here == 'A' {
                2.0  // Strong anchor
            } else {
                1.0
            };

            match_probability *= pred_match.max(0.01).powf(weight);
            current_group = pred_group;
        } else {
            break;  // No more predecessors
        }
    }

    // Walk forward through successors
    current_group = group_idx;
    current_seq_pos = candidate_pos as i32;
    for _ in 0..window {
        current_seq_pos += 1;
        if current_seq_pos as usize > seq_chars.len() {
            break;
        }

        let seq_char_here = seq_chars[current_seq_pos as usize - 1];

        // Find best successor link
        let mut best_succ: Option<(usize, f64)> = None;
        for &link_idx in &links_from[current_group] {
            let link = &sequential_links[link_idx];
            if visited.contains(&link.to_group) {
                continue;
            }
            let score = link.strength;
            if best_succ.is_none() || score > best_succ.unwrap().1 {
                best_succ = Some((link.to_group, score));
            }
        }

        // Check for PRO - successor at PRO position shouldn't exist (no backbone NH)
        if seq_char_here == 'P' {
            if best_succ.is_none() {
                // Correct! No successor at PRO position
                match_probability *= 2.0;  // Bonus
            } else {
                // Inconsistent - we have a successor but sequence says PRO
                match_probability *= 0.1;  // Penalty
            }
            break;
        }

        if let Some((succ_group, _)) = best_succ {
            visited.insert(succ_group);
            n_neighbors += 1;

            let succ_match = typing_probability_for_sequence_char(
                &typing_scores[succ_group], residue_types, seq_char_here
            );

            let weight = if seq_char_here == 'G' || seq_char_here == 'A' {
                2.0
            } else {
                1.0
            };

            match_probability *= succ_match.max(0.01).powf(weight);
            current_group = succ_group;
        } else {
            break;
        }
    }

    // Normalize by number of neighbors to avoid penalizing longer paths
    if n_neighbors > 0 {
        match_probability = match_probability.powf(1.0 / (n_neighbors as f64 + 1.0));
    }

    // Return as a factor (>1 means good match, <1 means poor match)
    // Scale so that 0.5 probability -> factor 1.0, 1.0 probability -> factor 2.0
    1.0 + match_probability
}

/// Compute how well a group's typing matches a specific sequence character.
/// Returns a DISCRIMINATIVE score:
/// - For distinctive residues (GLY, ALA): strong signal if match, strong penalty if mismatch
/// - For other residues: weak signal (they're harder to distinguish)
fn typing_probability_for_sequence_char(
    typing_scores: &[f64],
    residue_types: &[String],
    seq_char: char,
) -> f64 {
    let target_type = match seq_char {
        'G' => "GLY", 'A' => "ALA", 'V' => "VAL", 'L' => "LEU",
        'I' => "ILE", 'P' => "PRO", 'F' => "PHE", 'Y' => "TYR",
        'W' => "TRP", 'S' => "SER", 'T' => "THR", 'C' => "CYS",
        'M' => "MET", 'N' => "ASN", 'Q' => "GLN", 'D' => "ASP",
        'E' => "GLU", 'K' => "LYS", 'R' => "ARG", 'H' => "HIS",
        _ => return 0.5,  // Unknown - neutral
    };

    let total_score: f64 = typing_scores.iter().sum();
    if total_score <= 0.0 {
        return 0.5;
    }

    let matching_score: f64 = residue_types.iter()
        .enumerate()
        .filter(|(_, t)| t.as_str() == target_type)
        .map(|(i, _)| typing_scores[i])
        .sum();

    let prob = matching_score / total_score;

    // For distinctive residues, amplify the signal
    // GLY: no CB, very low CA (~45 ppm) - highly distinctive
    // ALA: characteristic CB (~18 ppm) - highly distinctive
    if seq_char == 'G' || seq_char == 'A' {
        // If prob > 0.3, this is likely correct -> strong boost
        // If prob < 0.1, this is likely wrong -> strong penalty
        if prob > 0.3 {
            return prob * 3.0;  // Strong match
        } else if prob < 0.1 {
            return prob * 0.1;  // Strong mismatch - sequence says G/A but typing doesn't
        }
    }

    // For other residues, return moderate probability
    prob
}

/// Check if a group is confidently typed as GLY (no CB, low CA)
fn is_confident_gly(typing_scores: &[f64], residue_types: &[String]) -> bool {
    let total: f64 = typing_scores.iter().sum();
    if total <= 0.0 {
        return false;
    }

    let gly_score: f64 = residue_types.iter()
        .enumerate()
        .filter(|(_, t)| t.as_str() == "GLY")
        .map(|(i, _)| typing_scores[i])
        .sum();

    gly_score / total > 0.4  // >40% GLY probability
}

/// Check if a group is confidently typed as ALA
fn is_confident_ala(typing_scores: &[f64], residue_types: &[String]) -> bool {
    let total: f64 = typing_scores.iter().sum();
    if total <= 0.0 {
        return false;
    }

    let ala_score: f64 = residue_types.iter()
        .enumerate()
        .filter(|(_, t)| t.as_str() == "ALA")
        .map(|(i, _)| typing_scores[i])
        .sum();

    ala_score / total > 0.4  // >40% ALA probability
}

/// Compute typing scores for each group based on evidence.
///
/// UNIFIED JOINT SCORING: Collects ALL observed shifts (H, N, CA, CB, HA, HB, etc.)
/// and scores them together in ONE product of KDE densities. Each atom contributes
/// exactly ONCE regardless of how many observations provided it.
///
/// P(residue_type | shifts) ∝ ∏ KDE(atom_shift | residue_type, atom)
///
fn compute_group_typing_scores(
    groups: &[SpinSystemEvidence],
    residue_types: &[String],
    kde: &KDEDatabase,
) -> Vec<Vec<f64>> {
    let n_positions = residue_types.len();

    groups.iter().map(|group| {
        let mut scores = vec![1.0; n_positions];

        // Collect ALL intra-residue shifts into one map: atom_name -> shift
        // Each atom contributes exactly ONCE (HashMap ensures deduplication)
        let mut all_shifts: HashMap<String, f64> = HashMap::new();

        // Backbone H and N
        if group.h_shift > 0.0 {
            all_shifts.insert("H".to_string(), group.h_shift);
        }
        if group.n_shift > 0.0 {
            all_shifts.insert("N".to_string(), group.n_shift);
        }

        // Carbons: CA, CB, C', CG, etc.
        for (atom, &shift) in &group.intra_carbons {
            all_shifts.insert(atom.clone(), shift);
        }

        // Sidechain protons: HA, HB, HG, etc.
        for (atom, &shift) in &group.intra_protons {
            all_shifts.insert(atom.clone(), shift);
        }

        // Hard constraints based on what we observed
        let has_backbone_hn = all_shifts.contains_key("H") && all_shifts.contains_key("N");
        let has_cb = all_shifts.contains_key("CB");

        for (pos, res_type) in residue_types.iter().enumerate() {
            // HARD CONSTRAINT: Groups with backbone H/N CANNOT be prolines
            // Prolines have no amide proton (it's an imino acid)
            if has_backbone_hn && res_type == "PRO" {
                scores[pos] = 1e-20;
                continue;
            }

            // HARD CONSTRAINT: N-terminus (position 0) has no backbone amide
            // Position 0 has -NH3+ (amino terminus), not -NH- (amide)
            if has_backbone_hn && pos == 0 {
                scores[pos] = 1e-20;
                continue;
            }

            // HARD CONSTRAINT: GLY has no CB - if we see CB, it's not glycine
            if has_cb && res_type == "GLY" {
                scores[pos] = 1e-20;
                continue;
            }

            // UNIFIED SCORING: Product of KDE densities for ALL observed atoms
            // Each atom contributes exactly once to the joint probability
            let mut log_score = 0.0;

            for (atom, &shift) in &all_shifts {
                let density = get_atom_kde_density(kde, res_type, atom, shift);

                if density > 0.0 {
                    log_score += density.ln();
                } else {
                    // Zero density = this residue type doesn't have this atom
                    // Strong negative evidence (but not as harsh as hard constraint)
                    log_score += -30.0;  // ln(1e-13) ≈ -30
                }
            }

            // Inter-residue protons (HBHACONH): score against PRECEDING residue
            // These constrain the (i-1) position, not the current position
            if !group.inter_protons.is_empty() && pos > 0 {
                let prev_res_type = &residue_types[pos - 1];
                for (atom, &shift) in &group.inter_protons {
                    let density = get_atom_kde_density(kde, prev_res_type, atom, shift);
                    if density > 0.0 {
                        log_score += density.ln();
                    } else {
                        log_score += -20.0;  // ln(1e-9) ≈ -20 (softer for inter)
                    }
                }
            }

            scores[pos] = log_score.exp();
        }

        // Normalize to probabilities
        let sum: f64 = scores.iter().sum();
        if sum > 0.0 {
            for s in scores.iter_mut() {
                *s /= sum;
            }
        }

        scores
    }).collect()
}

/// Get KDE density for an atom, handling stereospecific naming variants.
/// Tries exact name first, then common variants (HA -> HA2/HA3 for GLY, etc.)
fn get_atom_kde_density(kde: &KDEDatabase, res_type: &str, atom: &str, shift: f64) -> f64 {
    // Try exact atom name first
    let direct = kde.density(res_type, atom, shift);
    if direct > 0.0 {
        return direct;
    }

    // Try stereospecific variants
    match atom {
        "HA" => {
            // GLY has HA2/HA3 instead of HA
            kde.density(res_type, "HA2", shift)
                .max(kde.density(res_type, "HA3", shift))
        }
        "HB" => {
            // Many residues have HB2/HB3 instead of single HB
            kde.density(res_type, "HB", shift)
                .max(kde.density(res_type, "HB2", shift))
                .max(kde.density(res_type, "HB3", shift))
        }
        "HG" => {
            kde.density(res_type, "HG", shift)
                .max(kde.density(res_type, "HG2", shift))
                .max(kde.density(res_type, "HG3", shift))
        }
        "HD" => {
            kde.density(res_type, "HD1", shift)
                .max(kde.density(res_type, "HD2", shift))
                .max(kde.density(res_type, "HD3", shift))
        }
        _ => 0.0
    }
}


/// Extract unique assignments from beliefs using greedy approach.
fn extract_group_assignments(
    beliefs: &[Vec<f64>],
    groups: &[SpinSystemEvidence],
    residue_types: &[String],
    verbose: bool,
) -> Vec<(usize, i32, f64)> {
    let n_groups = groups.len();
    let domain_size = beliefs[0].len();

    // Collect (group_idx, best_position, confidence) for all groups
    let mut candidates: Vec<(usize, usize, f64)> = groups.iter().enumerate()
        .map(|(g, _)| {
            let (best_pos, &best_conf) = beliefs[g].iter().enumerate()
                .skip(1)  // Skip position 0 (unassigned)
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap_or((0, &0.0));
            (g, best_pos, best_conf)
        })
        .collect();

    // Sort by confidence (highest first)
    candidates.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

    // Greedy assignment
    let mut assigned_positions: HashSet<usize> = HashSet::new();
    let mut results: Vec<(usize, i32, f64)> = Vec::new();

    for (group_idx, best_pos, confidence) in candidates {
        if best_pos == 0 || assigned_positions.contains(&best_pos) {
            // Find best available position
            let available = beliefs[group_idx].iter().enumerate()
                .skip(1)
                .filter(|(pos, _)| !assigned_positions.contains(pos))
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap());

            if let Some((pos, &conf)) = available {
                assigned_positions.insert(pos);
                results.push((group_idx, pos as i32, conf));
            }
        } else {
            assigned_positions.insert(best_pos);
            results.push((group_idx, best_pos as i32, confidence));
        }
    }

    if verbose {
        println!("\n=== GROUP ASSIGNMENTS ===");
        println!("Assigned {} / {} groups", results.len(), n_groups);
        for (g, pos, conf) in results.iter().take(10) {
            let res_type = if *pos > 0 && (*pos as usize) <= residue_types.len() {
                &residue_types[*pos as usize - 1]
            } else {
                "?"
            };
            let group = &groups[*g];
            println!("  Group {} -> pos {} ({}): conf={:.3}, H={:.3}, N={:.2}",
                g, pos, res_type, conf, group.h_shift, group.n_shift);
        }
        // Debug: Show assignments for groups with H near 8.58 (31 Q / 72 R case)
        println!("\n--- 31 Q / 72 R case (H~8.58) ---");
        for (g, pos, conf) in results.iter() {
            let group = &groups[*g];
            if (group.h_shift - 8.58).abs() < 0.05 {
                let res_type = if *pos > 0 && (*pos as usize) <= residue_types.len() {
                    &residue_types[*pos as usize - 1]
                } else {
                    "?"
                };
                println!("  Group {} -> pos {} ({}): conf={:.3}, H={:.3}, N={:.2}, CA={:?}",
                    g, pos, res_type, conf, group.h_shift, group.n_shift, group.intra_carbons.get("CA"));
            }
        }
        println!("=========================\n");
    }

    results
}

/// Compute Bayesian likelihood ratio for a sequential link.
///
/// For each atom observed in BOTH groups:
/// - Match (diff < tolerance): LR contribution = 19 (strong evidence FOR)
/// - Mismatch (diff > tolerance): LR contribution = 0.05 (strong evidence AGAINST)
/// - Missing in one or both: LR contribution = 1 (neutral)
///
/// Final LR = product of all contributions. A single mismatch among observed atoms
/// destroys the link probability (LR drops by ~20x per mismatch).
fn compute_sequential_link_likelihood_ratio(
    from_inter: &std::collections::HashMap<String, f64>,
    to_intra: &std::collections::HashMap<String, f64>,
    c_tolerance: f64,
    h_tolerance: f64,
) -> (f64, Vec<String>, Vec<String>, f64, f64) {
    // Returns: (likelihood_ratio, matched_atoms, conflicting_atoms, avg_diff, max_diff)

    let mut log_lr = 0.0;
    let mut matched_atoms = Vec::new();
    let mut conflicting_atoms = Vec::new();
    let mut total_diff = 0.0;
    let mut max_diff = 0.0f64;

    // Log likelihood ratio for match/mismatch
    // P(match | true_link) ≈ 0.95, P(match | false_link) ≈ 0.05
    // LR_match = 0.95/0.05 = 19, log(19) ≈ 2.94
    // LR_mismatch = 0.05/0.95 ≈ 0.053, log(0.053) ≈ -2.94
    let log_lr_match = (0.95_f64 / 0.05).ln();      // +2.94
    let log_lr_mismatch = (0.05_f64 / 0.95).ln();  // -2.94

    // Check all backbone atoms that could provide sequential evidence
    // Carbons: CA, CB, C (carbonyl)
    // Protons: HA, HB (from HBHACONH)
    for atom in &["CA", "CB", "C", "HA", "HB", "HA2", "HA3", "HB2", "HB3"] {
        let atom_str = atom.to_string();

        // Get tolerance based on nucleus type
        let tol = if atom.starts_with('H') { h_tolerance } else { c_tolerance };

        match (from_inter.get(&atom_str), to_intra.get(&atom_str)) {
            (Some(&inter), Some(&intra)) => {
                // Both groups have this atom - compare them
                let diff = (inter - intra).abs();

                if diff < tol {
                    // Match: strong evidence FOR the link
                    log_lr += log_lr_match;
                    matched_atoms.push(atom_str);
                    total_diff += diff;
                    max_diff = max_diff.max(diff);
                } else {
                    // Mismatch: strong evidence AGAINST the link
                    log_lr += log_lr_mismatch;
                    conflicting_atoms.push(atom_str);
                }
            }
            _ => {
                // One or both missing: neutral (no information)
                // LR contribution = 1, log(1) = 0
            }
        }
    }

    let likelihood_ratio = log_lr.exp();
    let avg_diff = if !matched_atoms.is_empty() {
        total_diff / matched_atoms.len() as f64
    } else {
        0.0
    };

    (likelihood_ratio, matched_atoms, conflicting_atoms, avg_diff, max_diff)
}

/// Build sequential links between groups based on inter/intra carbon matching.
/// Uses Bayesian likelihood ratio scoring - each observed atom provides independent evidence.
/// A mismatch in ANY observed atom drastically reduces link probability.
pub fn build_group_sequential_links_with_diffs(
    groups: &[SpinSystemEvidence],
    c_tolerance: f64,
) -> Vec<GroupSequentialLink> {
    let mut links = Vec::new();

    // Proton tolerance (tighter than carbon)
    let h_tolerance = 0.05;

    // Minimum LR to accept a link
    // LR = 19 means 1 match, 0 conflicts (ln(19) ≈ 2.94)
    // LR = 361 means 2 matches, 0 conflicts (ln(361) ≈ 5.89)
    // LR = 6859 means 3 matches, 0 conflicts (ln(6859) ≈ 8.83)
    // LR = 19 with 1 conflict becomes LR ≈ 1 (useless)
    //
    // Accept single-atom matches (LR > 15) but prefer multi-atom (LR > 300)
    // The key is rejecting links with CONFLICTS, not requiring multiple matches
    let min_lr = 15.0;

    for (from_idx, from_group) in groups.iter().enumerate() {
        // from_group has inter carbons (observed at i-1)
        if from_group.inter_carbons.is_empty() {
            continue;
        }

        for (to_idx, to_group) in groups.iter().enumerate() {
            if from_idx == to_idx {
                continue;
            }

            // Compute Bayesian likelihood ratio
            let (lr, matched_atoms, conflicting_atoms, avg_diff, max_diff) =
                compute_sequential_link_likelihood_ratio(
                    &from_group.inter_carbons,
                    &to_group.intra_carbons,
                    c_tolerance,
                    h_tolerance,
                );

            // Only accept links with sufficient evidence
            if lr >= min_lr && !matched_atoms.is_empty() {
                // Use log(LR) as strength for BP (proper Bayesian)
                let strength = lr.ln();

                links.push(GroupSequentialLink {
                    from_group: from_idx,
                    to_group: to_idx,
                    strength,
                    matched_atoms,
                    avg_shift_diff: avg_diff,
                    max_shift_diff: max_diff,
                });
            }
            // Links with conflicts are implicitly rejected (LR < min_lr)
        }
    }

    links
}

/// Build sequential links (legacy version without diffs).
pub fn build_group_sequential_links(
    groups: &[SpinSystemEvidence],
    c_tolerance: f64,
) -> Vec<GroupSequentialLink> {
    build_group_sequential_links_with_diffs(groups, c_tolerance)
}

// =============================================================================
// END: Group-Level Bayesian Assignment
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
            sequential_weight: 3.0,     // Moderate sequential links
            sequence_type_weight: 10.0,  // Strong: typed X → must be at X position (anchor)
            sequence_type_confidence_threshold: 0.35,  // Apply when confidence > 35% (catch more anchors)
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
            // Sequence-type constraint: distinctive residues (Gly, Ala, Thr, Ser) need anchoring
            // from the START, not just during refinement. Start at 50% strength.
            sequence_type_weight: self.sequence_type_weight * (0.5 + 0.5 * t),  // Start at 50%, ramp to 100%
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

/// Carbon atom type for triple-resonance observations.
/// Physics-based classification derived from atom_constraint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CarbonAtomType {
    /// Alpha carbon (CA)
    CA,
    /// Beta carbon (CB)
    CB,
    /// Carbonyl carbon (C')
    CO,
}

/// Carbon observation from a 3D triple-resonance experiment, linked to a backbone NH.
/// Used for both typing (CA/CB shifts are diagnostic) and sequential assignment.
#[derive(Debug, Clone)]
struct TripleResCarbonObs {
    /// Index of the backbone peak this observation is linked to (via H/N matching)
    backbone_idx: usize,
    /// Carbon chemical shift (CA, CB, or CO)
    carbon_shift: f64,
    /// Carbon atom type: CA, CB, or CO (physics-based, derived from atom_constraint)
    atom_type: CarbonAtomType,
    /// True = intra-residue (i), False = inter-residue (i-1)
    is_intra: bool,
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
    /// CA ppm difference (absolute value) if both observed
    ca_ppm_diff: Option<f64>,
    /// CB ppm difference if both observed (absolute value)
    cb_ppm_diff: Option<f64>,
    /// CO ppm difference if both observed (absolute value)
    co_ppm_diff: Option<f64>,
    /// Bayesian log likelihood ratio (pre-computed)
    log_likelihood_ratio: f64,
}

impl TripleResSequentialLink {
    /// Compute quality using Bayesian likelihood ratio.
    /// Returns None if log_LR is below threshold (link not valid).
    fn quality(&self, c_tolerance: f64) -> Option<f64> {
        // Require minimum log-LR of 2.7 (LR ≈ 15, roughly one confident match)
        let min_log_lr = 2.7;
        if self.log_likelihood_ratio < min_log_lr {
            return None;
        }
        // Quality scales with log-LR, normalized to [0, 1] range
        // log-LR of 2.7 → 0.0, log-LR of 8.8 (3 matches) → 1.0
        let quality = ((self.log_likelihood_ratio - min_log_lr) / 6.0).min(1.0);
        Some(quality)
    }

    /// Create a new link with Bayesian scoring.
    /// Each observed atom contributes evidence: match → +2.94, mismatch → -2.94, missing → 0
    fn new_with_bayesian_scoring(
        from_backbone_idx: usize,
        to_backbone_idx: usize,
        ca_ppm_diff: Option<f64>,
        cb_ppm_diff: Option<f64>,
        co_ppm_diff: Option<f64>,
        c_tolerance: f64,
    ) -> Self {
        let log_lr_match = (0.95_f64 / 0.05).ln();      // +2.94
        let log_lr_mismatch = (0.05_f64 / 0.95).ln();  // -2.94

        let mut log_lr = 0.0;

        // CA evidence
        if let Some(diff) = ca_ppm_diff {
            if diff < c_tolerance {
                log_lr += log_lr_match;  // Match
            } else {
                log_lr += log_lr_mismatch;  // Conflict
            }
        }
        // Missing CA: log_lr += 0 (neutral)

        // CB evidence
        if let Some(diff) = cb_ppm_diff {
            if diff < c_tolerance {
                log_lr += log_lr_match;
            } else {
                log_lr += log_lr_mismatch;
            }
        }

        // CO evidence
        if let Some(diff) = co_ppm_diff {
            if diff < c_tolerance {
                log_lr += log_lr_match;
            } else {
                log_lr += log_lr_mismatch;
            }
        }

        Self {
            from_backbone_idx,
            to_backbone_idx,
            ca_ppm_diff,
            cb_ppm_diff,
            co_ppm_diff,
            log_likelihood_ratio: log_lr,
        }
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
        hncaco: &[UnlabeledPeak],
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
                        atom_type: CarbonAtomType::CO,
                        is_intra: false,  // HNCO CO is always from i-1
                    });
                }
            }
        }

        // Process HN(CA)CO: (H, N, CO) - CO(i) strong, CO(i-1) weak
        // Similar to HNCA - intensity determines intra vs inter
        for peak in hncaco {
            if peak.position_ppm.len() >= 3 {
                let (h, n, co) = (peak.position_ppm[0], peak.position_ppm[1], peak.position_ppm[2]);
                if let Some(bb_idx) = find_backbone(h, n) {
                    let is_intra = peak.intensity > 0.5;  // Strong = intra-residue CO(i)
                    triple_res_carbons.push(TripleResCarbonObs {
                        backbone_idx: bb_idx,
                        carbon_shift: co,
                        atom_type: CarbonAtomType::CO,
                        is_intra,
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
                        atom_type: CarbonAtomType::CA,
                        is_intra,
                    });
                }
            }
        }

        // Process HNCACB: (H, N, CA/CB)
        // SIGN determines CA vs CB: positive = CA, negative = CB (CB is inverted)
        // MAGNITUDE determines intra vs inter: |intensity| > 0.5 = intra (i), <= 0.5 = inter (i-1)
        for peak in hncacb {
            if peak.position_ppm.len() >= 3 {
                let (h, n, c) = (peak.position_ppm[0], peak.position_ppm[1], peak.position_ppm[2]);
                if let Some(bb_idx) = find_backbone(h, n) {
                    let atom_type = if peak.intensity > 0.0 { CarbonAtomType::CA } else { CarbonAtomType::CB };
                    let is_intra = peak.intensity.abs() > 0.5;  // Magnitude determines intra/inter
                    triple_res_carbons.push(TripleResCarbonObs {
                        backbone_idx: bb_idx,
                        carbon_shift: c,
                        atom_type,
                        is_intra,
                    });
                }
            }
        }

        // Process CBCACONH: (H, N, CA/CB) - i-1 only
        // Use sign encoding consistent with HNCACB: positive = CA, negative = CB
        for peak in cbcaconh {
            if peak.position_ppm.len() >= 3 {
                let (h, n, c) = (peak.position_ppm[0], peak.position_ppm[1], peak.position_ppm[2]);
                if let Some(bb_idx) = find_backbone(h, n) {
                    let atom_type = if peak.intensity > 0.0 { CarbonAtomType::CA } else { CarbonAtomType::CB };
                    triple_res_carbons.push(TripleResCarbonObs {
                        backbone_idx: bb_idx,
                        carbon_shift: c,
                        atom_type,
                        is_intra: false,  // CBCACONH always shows i-1
                    });
                }
            }
        }

        // Process HBHACONH: (H, N, HA/HB) - i-1 protons (for future use)
        // Currently not used for BP but stored for potential future enhancement
        let _hbhaconh_count = hbhaconh.len();  // Acknowledge but don't store

        // === Build sequential links from CA/CB/CO matching with Bayesian LR scoring ===
        // For each backbone peak, collect its intra-residue atoms and inter-residue atoms
        // Then match: if backbone A has intra-atoms that match backbone B's inter-atoms, A precedes B
        // Bayesian scoring: each matching atom is +2.94 log-LR, each conflicting atom is -2.94 log-LR
        let mut triple_res_sequential: Vec<TripleResSequentialLink> = Vec::new();

        // Group carbon observations by backbone index and atom type
        let mut intra_ca_by_bb: HashMap<usize, Vec<f64>> = HashMap::new();
        let mut inter_ca_by_bb: HashMap<usize, Vec<f64>> = HashMap::new();
        let mut intra_cb_by_bb: HashMap<usize, Vec<f64>> = HashMap::new();
        let mut inter_cb_by_bb: HashMap<usize, Vec<f64>> = HashMap::new();
        let mut intra_co_by_bb: HashMap<usize, Vec<f64>> = HashMap::new();
        let mut inter_co_by_bb: HashMap<usize, Vec<f64>> = HashMap::new();

        for obs in &triple_res_carbons {
            let (intra_map, inter_map) = match obs.atom_type {
                CarbonAtomType::CA => (&mut intra_ca_by_bb, &mut inter_ca_by_bb),
                CarbonAtomType::CB => (&mut intra_cb_by_bb, &mut inter_cb_by_bb),
                CarbonAtomType::CO => (&mut intra_co_by_bb, &mut inter_co_by_bb),
            };
            if obs.is_intra {
                intra_map.entry(obs.backbone_idx).or_default().push(obs.carbon_shift);
            } else {
                inter_map.entry(obs.backbone_idx).or_default().push(obs.carbon_shift);
            }
        }

        // Helper to find best (smallest) match between two atom lists, returning None if both have values but no match
        let find_best_match = |intra_list: Option<&Vec<f64>>, inter_list: Option<&Vec<f64>>, tol: f64| -> Option<f64> {
            match (intra_list, inter_list) {
                (Some(intra), Some(inter)) if !intra.is_empty() && !inter.is_empty() => {
                    let mut best = f64::MAX;
                    for &a in intra {
                        for &b in inter {
                            best = best.min((a - b).abs());
                        }
                    }
                    Some(best)  // Returns the difference, even if > tol (for conflict detection)
                }
                _ => None,  // Missing data = neutral
            }
        };

        // Build all backbone pairs, score with Bayesian LR
        let backbone_count = peaks.iter().filter(|p| p.peak_type == PeakType::Backbone).count();
        let mut best_links: HashMap<(usize, usize), TripleResSequentialLink> = HashMap::new();

        for bb_a in 0..backbone_count {
            for bb_b in 0..backbone_count {
                if bb_a == bb_b { continue; }

                // Get differences for each atom type (if both observed)
                let ca_diff = find_best_match(
                    intra_ca_by_bb.get(&bb_a),
                    inter_ca_by_bb.get(&bb_b),
                    carbon_match_tolerance
                );
                let cb_diff = find_best_match(
                    intra_cb_by_bb.get(&bb_a),
                    inter_cb_by_bb.get(&bb_b),
                    carbon_match_tolerance
                );
                let co_diff = find_best_match(
                    intra_co_by_bb.get(&bb_a),
                    inter_co_by_bb.get(&bb_b),
                    carbon_match_tolerance
                );

                // Skip if no atoms to compare
                if ca_diff.is_none() && cb_diff.is_none() && co_diff.is_none() {
                    continue;
                }

                // Create link with Bayesian LR scoring
                let link = TripleResSequentialLink::new_with_bayesian_scoring(
                    bb_a, bb_b,
                    ca_diff, cb_diff, co_diff,
                    carbon_match_tolerance,
                );

                // Only keep links with positive log-LR (at least one match, no conflicts)
                if link.log_likelihood_ratio > 0.0 {
                    let key = (bb_a, bb_b);
                    if let Some(existing) = best_links.get(&key) {
                        // Keep link with higher log-LR
                        if link.log_likelihood_ratio > existing.log_likelihood_ratio {
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
            println!("\n--- 3D TRIPLE-RESONANCE DATA (Bayesian LR scoring) ---");
            println!("  Carbon observations linked to backbone: {}", triple_res_carbons.len());
            println!("  Initial tolerances: H/N={:.3} ppm, CA/CB/CO={:.2} ppm",
                hn_tolerance, carbon_match_tolerance);

            // Count observations by atom type (physics-based)
            let ca_count = triple_res_carbons.iter().filter(|o| o.atom_type == CarbonAtomType::CA).count();
            let cb_count = triple_res_carbons.iter().filter(|o| o.atom_type == CarbonAtomType::CB).count();
            let co_count = triple_res_carbons.iter().filter(|o| o.atom_type == CarbonAtomType::CO).count();
            println!("  Atom types: CA={}, CB={}, CO={}", ca_count, cb_count, co_count);

            // Show intra and inter atoms for each backbone
            println!("\n  Carbon shifts by backbone (for sequential matching):");
            for bb_idx in 0..peaks.iter().filter(|p| p.peak_type == PeakType::Backbone).count() {
                let intra_cas: Vec<f64> = intra_ca_by_bb.get(&bb_idx).map_or(vec![], |v| v.clone());
                let inter_cas: Vec<f64> = inter_ca_by_bb.get(&bb_idx).map_or(vec![], |v| v.clone());
                let intra_cbs: Vec<f64> = intra_cb_by_bb.get(&bb_idx).map_or(vec![], |v| v.clone());
                let inter_cbs: Vec<f64> = inter_cb_by_bb.get(&bb_idx).map_or(vec![], |v| v.clone());
                let intra_cos: Vec<f64> = intra_co_by_bb.get(&bb_idx).map_or(vec![], |v| v.clone());
                let inter_cos: Vec<f64> = inter_co_by_bb.get(&bb_idx).map_or(vec![], |v| v.clone());

                if !intra_cas.is_empty() || !inter_cas.is_empty() || !intra_cbs.is_empty() || !inter_cbs.is_empty() || !intra_cos.is_empty() || !inter_cos.is_empty() {
                    println!("    BB {}:", bb_idx);
                    if !intra_cas.is_empty() || !inter_cas.is_empty() {
                        println!("      CA: intra={:?}, inter={:?}",
                            intra_cas.iter().map(|x| format!("{:.1}", x)).collect::<Vec<_>>(),
                            inter_cas.iter().map(|x| format!("{:.1}", x)).collect::<Vec<_>>());
                    }
                    if !intra_cbs.is_empty() || !inter_cbs.is_empty() {
                        println!("      CB: intra={:?}, inter={:?}",
                            intra_cbs.iter().map(|x| format!("{:.1}", x)).collect::<Vec<_>>(),
                            inter_cbs.iter().map(|x| format!("{:.1}", x)).collect::<Vec<_>>());
                    }
                    if !intra_cos.is_empty() || !inter_cos.is_empty() {
                        println!("      CO: intra={:?}, inter={:?}",
                            intra_cos.iter().map(|x| format!("{:.1}", x)).collect::<Vec<_>>(),
                            inter_cos.iter().map(|x| format!("{:.1}", x)).collect::<Vec<_>>());
                    }
                }
            }

            println!("\n  Sequential links (Bayesian LR): {} links", triple_res_sequential.len());
            // Sort by log-LR for better display
            let mut sorted_links: Vec<_> = triple_res_sequential.iter().collect();
            sorted_links.sort_by(|a, b| b.log_likelihood_ratio.partial_cmp(&a.log_likelihood_ratio).unwrap_or(std::cmp::Ordering::Equal));
            for link in sorted_links.iter().take(20) {  // Show top 20
                let ca_str = link.ca_ppm_diff.map_or("-".to_string(), |d| format!("{:.3}", d));
                let cb_str = link.cb_ppm_diff.map_or("-".to_string(), |d| format!("{:.3}", d));
                let co_str = link.co_ppm_diff.map_or("-".to_string(), |d| format!("{:.3}", d));
                println!("    BB {} → BB {} (log-LR={:.2}, CA={}, CB={}, CO={})",
                    link.from_backbone_idx, link.to_backbone_idx, link.log_likelihood_ratio, ca_str, cb_str, co_str);
            }
            if triple_res_sequential.len() > 20 {
                println!("    ... and {} more links", triple_res_sequential.len() - 20);
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

                // Physics-based: use atom_type enum
                let atom_name = match obs.atom_type {
                    CarbonAtomType::CA => "CA",
                    CarbonAtomType::CB => "CB",
                    CarbonAtomType::CO => "C",  // Carbonyl is "C" in BMRB
                };

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

    // Build type → positions map for sequence-type constraint (Factor 5)
    // e.g., if sequence is "ACGDEF": "ALA" -> [1], "CYS" -> [2], "GLY" -> [3], etc.
    // If a type appears multiple times, it has multiple valid positions
    let mut type_to_positions: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, res_type) in residue_types.iter().enumerate() {
        type_to_positions.entry(res_type.clone()).or_default().push(i + 1);
    }

    if params.verbose {
        println!("\n--- SEQUENCE-TYPE MAP (Factor 5) ---");
        for (aa_type, positions) in &type_to_positions {
            println!("  {} appears at positions: {:?}", aa_type, positions);
        }
    }

    // Load KDE database for typing
    let kde = KDEDatabase::load_embedded();

    // Initialize beliefs uniformly
    let mut beliefs: Vec<Vec<f64>> = observations.iter()
        .map(|_| vec![1.0 / domain_size as f64; domain_size])
        .collect();

    // Run belief propagation
    // Identify backbone observations - include ALL observations with (H, N) backbone
    // For group-level BP, we need BOTH intra AND inter observations to aggregate evidence
    // The intra/inter distinction is handled when collecting carbon evidence
    let backbone_indices: Vec<usize> = observations.iter().enumerate()
        .filter(|(_, obs)| {
            // Physics-based: any observation with backbone H-N anchor
            // This includes HSQC-15N, HNCA, HNCACB, HNCO, HNCACO, CBCACONH, HBHACONH, etc.
            obs.has_backbone_hn()
        })
        .map(|(idx, _)| idx)
        .collect();

    // ==========================================================================
    // CRYSTALLINE-COMPATIBLE SOFT GROUPING (Mahalanobis + GMM)
    // ==========================================================================
    //
    // Replace hard rectangular tolerance with covariance-aware Mahalanobis distance.
    // This properly accounts for different H vs N measurement precision:
    //   σ_H = 0.015 ppm (H line width - 10x more discriminating)
    //   σ_N = 0.15 ppm  (N line width)
    //
    // Example: For residues 69 L (H=8.340, N=124.460) and 73 L (H=8.320, N=124.220):
    //   Hard tolerance: ΔH=0.02<0.03, ΔN=0.24<0.30 → MERGE (wrong!)
    //   Mahalanobis: d = sqrt((0.02/0.015)² + (0.24/0.15)²) = 2.08 > 2.0 → SEPARATE (correct!)

    let sigma_h = 0.015;  // H measurement uncertainty (ppm)
    let sigma_n = 0.15;   // N measurement uncertainty (ppm)
    let fit_threshold = 0.1;  // Minimum Gaussian likelihood to contribute to a group

    // Step 1: Create soft backbone groups using Mahalanobis-based center detection
    let mut soft_groups = create_soft_backbone_groups(
        observations, &backbone_indices, sigma_h, sigma_n, fit_threshold
    );

    // Step 2: Iterative center refinement (EM-like, CRYSTALLINE density evolution)
    // This allows groups to "crystallize" toward their true centers
    let em_iterations = 5;
    for _iteration in 0..em_iterations {
        // E-step: Recompute responsibilities with current centers
        recompute_group_weights(&mut soft_groups, observations, &backbone_indices, sigma_h, sigma_n, fit_threshold);

        // M-step: Update centers toward weighted mean
        for group in &mut soft_groups {
            group.update_center_from_weights(observations);
        }
    }

    // Step 3: Detect and split multimodal groups (overlapping H/N with different CA)
    // This handles exact H/N overlap cases where two residues have identical backbone
    detect_and_split_multimodal_groups(&mut soft_groups, observations, 2.0, params.verbose);

    // Step 4: Probabilistic overlap detection using observation count
    // Compute expected observations per spin system based on carbon atom types present
    // Physics-based: count unique (atom_type, residue_offset) combinations
    let expected_obs_per_system: f64 = {
        let mut count: f64 = 0.0;

        // Count CA observations (intra and inter)
        let has_intra_ca = observations.iter().any(|o|
            o.carbon_dimensions().any(|d| d.is_ca() && d.is_intra()));
        let has_inter_ca = observations.iter().any(|o|
            o.carbon_dimensions().any(|d| d.is_ca() && d.is_inter()));

        // Count CB observations (intra and inter)
        let has_intra_cb = observations.iter().any(|o|
            o.carbon_dimensions().any(|d| d.is_cb() && d.is_intra()));
        let has_inter_cb = observations.iter().any(|o|
            o.carbon_dimensions().any(|d| d.is_cb() && d.is_inter()));

        // Count CO (carbonyl) observations (intra and inter)
        let has_intra_co = observations.iter().any(|o|
            o.carbon_dimensions().any(|d| d.is_carbonyl() && d.is_intra()));
        let has_inter_co = observations.iter().any(|o|
            o.carbon_dimensions().any(|d| d.is_carbonyl() && d.is_inter()));

        // Each unique (atom_type, residue_offset) contributes 1 expected observation
        if has_intra_ca { count += 1.0; }
        if has_inter_ca { count += 1.0; }
        if has_intra_cb { count += 1.0; }
        if has_inter_cb { count += 1.0; }
        if has_intra_co { count += 1.0; }
        if has_inter_co { count += 1.0; }

        count.max(4.0)  // At least 4 expected
    };

    if params.verbose {
        println!("\n=== PROBABILISTIC OVERLAP DETECTION ===");
        println!("Expected observations per spin system: {:.0}", expected_obs_per_system);
    }

    let probable_overlaps = detect_probable_overlaps(
        &soft_groups,
        observations,
        expected_obs_per_system,
        0.3,  // Report groups with >30% overlap probability
        params.verbose,
    );

    if params.verbose {
        if probable_overlaps.is_empty() {
            println!("  No high-probability overlaps detected");
        } else {
            println!("  {} groups flagged for potential overlap", probable_overlaps.len());
        }
        println!("========================================\n");
    }

    // Convert soft groups to observation index lists for downstream compatibility
    // For each soft group, use the observations with weight > 0.5 (primary membership)
    let backbone_groups: Vec<Vec<usize>> = soft_groups.iter()
        .map(|group| {
            group.observation_weights.iter()
                .filter(|(_, &weight)| weight > 0.5)
                .map(|(&idx, _)| idx)
                .collect()
        })
        .filter(|group: &Vec<usize>| !group.is_empty())
        .collect();

    if params.verbose {
        println!("\n=== BACKBONE GROUPING ===");
        println!("Total backbone observations: {}", backbone_indices.len());
        println!("Backbone groups (unique spin systems): {}", backbone_groups.len());
        if backbone_groups.len() < 20 {
            for (i, group) in backbone_groups.iter().enumerate() {
                if let Some((h, n)) = get_backbone_hn_from_obs(&observations[group[0]]) {
                    println!("  Group {}: H={:.3}, N={:.2} ({} obs)", i, h, n, group.len());
                }
            }
        }
        println!("=========================\n");
    }

    // DEBUG: Find CG2-range carbon observations and track them
    let debug_cg2_indices: Vec<usize> = observations.iter().enumerate()
        .filter(|(_, obs)| {
            obs.dimensions.iter().any(|d| {
                d.nucleus == NucleusType::C13 && d.shift > 15.0 && d.shift < 25.0
            })
        })
        .map(|(idx, _)| idx)
        .collect();

    if params.verbose && !debug_cg2_indices.is_empty() {
        println!("\n=== CG-RANGE CARBON OBSERVATIONS ===");
        for &idx in &debug_cg2_indices {
            let obs = &observations[idx];
            let shifts: Vec<String> = obs.dimensions.iter()
                .map(|d| format!("{:?}={:.2}", d.nucleus, d.shift))
                .collect();
            println!("  obs[{}]: {} type={:?}", idx, shifts.join(", "), obs.experiment_type);
        }
        println!("=====================================\n");
    }

    // ==========================================================================
    // GROUP-LEVEL BP: The simplified physics-based approach
    // ==========================================================================
    //
    // Run BP on backbone GROUPS (66 variables) instead of observations (586).
    // This is the correct formulation: one decision per spin system.

    // 1. Build SpinSystemEvidence for each backbone group
    let spin_system_evidence: Vec<SpinSystemEvidence> = backbone_groups.iter()
        .enumerate()
        .map(|(group_idx, obs_indices)| {
            SpinSystemEvidence::from_observations(group_idx, obs_indices, observations)
        })
        .collect();

    if params.verbose {
        println!("\n=== SPIN SYSTEM EVIDENCE ===");
        for (i, ev) in spin_system_evidence.iter().enumerate().take(5) {
            println!("  Group {}: H={:.3}, N={:.2}", i, ev.h_shift, ev.n_shift);
            println!("    Intra carbons: {:?}", ev.intra_carbons);
            println!("    Inter carbons: {:?}", ev.inter_carbons);
        }
        println!("============================\n");

        // Debug: Find groups with carbons that match 31 Q (CA=60.0) or 72 R (CA=55.6)
        println!("=== SEARCHING FOR 31 Q (CA~60) or 72 R (CA~55.6) ===");
        for (i, ev) in spin_system_evidence.iter().enumerate() {
            // Check if any intra or inter CA is near 60.0 (GLN) or 55.6 (ARG)
            let mut matches = false;
            let mut match_info = String::new();

            if let Some(&ca_shift) = ev.intra_carbons.get("CA") {
                if (ca_shift - 60.0).abs() < 1.0 {
                    matches = true;
                    match_info.push_str(&format!("intra_CA={:.2} (Q?), ", ca_shift));
                }
                if (ca_shift - 55.6).abs() < 1.0 {
                    matches = true;
                    match_info.push_str(&format!("intra_CA={:.2} (R?), ", ca_shift));
                }
            }
            if let Some(&ca_shift) = ev.inter_carbons.get("CA") {
                if (ca_shift - 60.0).abs() < 1.0 {
                    matches = true;
                    match_info.push_str(&format!("inter_CA={:.2} (Q?), ", ca_shift));
                }
                if (ca_shift - 55.6).abs() < 1.0 {
                    matches = true;
                    match_info.push_str(&format!("inter_CA={:.2} (R?), ", ca_shift));
                }
            }

            if matches {
                println!("  Group {} (H={:.3}, N={:.2}): {} - {} obs",
                    i, ev.h_shift, ev.n_shift, match_info, ev.observation_indices.len());
                println!("    intra: {:?}", ev.intra_carbons);
                println!("    inter: {:?}", ev.inter_carbons);
            }
        }
        println!("============================\n");
    }

    // 2. Build sequential links between groups
    // Use BROAD tolerance initially - links will be filtered/weighted during BP based on match quality
    // This follows the adaptive tolerance principle: start loose, converge to tight
    let c_tolerance = params.triple_res_c_tolerance_initial.max(1.0);  // Broad initial tolerance
    let group_sequential_links = build_group_sequential_links_with_diffs(&spin_system_evidence, c_tolerance);

    if params.verbose {
        println!("Sequential links between groups: {}", group_sequential_links.len());
        for link in group_sequential_links.iter().take(10) {
            println!("  Group {} -> {} (strength={:.2}, atoms={:?})",
                link.from_group, link.to_group, link.strength, link.matched_atoms);
        }

        // Debug: Show sequential links and typing for groups with H~8.58 (31 Q / 72 R case)
        println!("\n=== SEQUENTIAL LINKS FOR 31 Q / 72 R CASE (H~8.58) ===");
        for (i, ev) in spin_system_evidence.iter().enumerate() {
            if (ev.h_shift - 8.58).abs() < 0.05 {
                println!("Group {} (H={:.3}, N={:.2}):", i, ev.h_shift, ev.n_shift);
                println!("  intra_CA={:?}, intra_CB={:?}", ev.intra_carbons.get("CA"), ev.intra_carbons.get("CB"));
                println!("  inter_CA={:?}, inter_CB={:?}", ev.inter_carbons.get("CA"), ev.inter_carbons.get("CB"));
                println!("  intra_protons={:?}", ev.intra_protons);
                println!("  inter_protons={:?}", ev.inter_protons);

                // Find links FROM this group (where this group's inter matches another's intra)
                let links_from: Vec<_> = group_sequential_links.iter()
                    .filter(|l| l.from_group == i)
                    .collect();
                if !links_from.is_empty() {
                    println!("  Links FROM (this group's inter → other's intra):");
                    for l in links_from {
                        let to_ev = &spin_system_evidence[l.to_group];
                        println!("    -> Group {} (H={:.3}, N={:.2}, intra_CA={:?}): str={:.2}, atoms={:?}",
                            l.to_group, to_ev.h_shift, to_ev.n_shift, to_ev.intra_carbons.get("CA"),
                            l.strength, l.matched_atoms);
                    }
                }

                // Find links TO this group (where another group's inter matches this group's intra)
                let links_to: Vec<_> = group_sequential_links.iter()
                    .filter(|l| l.to_group == i)
                    .collect();
                if !links_to.is_empty() {
                    println!("  Links TO (other's inter → this group's intra):");
                    for l in links_to {
                        let from_ev = &spin_system_evidence[l.from_group];
                        println!("    <- Group {} (H={:.3}, N={:.2}, inter_CA={:?}): str={:.2}, atoms={:?}",
                            l.from_group, from_ev.h_shift, from_ev.n_shift, from_ev.inter_carbons.get("CA"),
                            l.strength, l.matched_atoms);
                    }
                }

                // Show typing scores for positions 31 (GLN) and 72 (ARG)
                println!("  KDE typing scores:");
                if let Some(&ca) = ev.intra_carbons.get("CA") {
                    let gln_ca_score = kde.density("GLN", "CA", ca);
                    let arg_ca_score = kde.density("ARG", "CA", ca);
                    println!("    CA={:.2}: GLN={:.4}, ARG={:.4}", ca, gln_ca_score, arg_ca_score);
                }
                if let Some(&cb) = ev.intra_carbons.get("CB") {
                    let gln_cb_score = kde.density("GLN", "CB", cb);
                    let arg_cb_score = kde.density("ARG", "CB", cb);
                    println!("    CB={:.2}: GLN={:.4}, ARG={:.4}", cb, gln_cb_score, arg_cb_score);
                }
            }
        }
        println!("============================\n");
    }

    // 3. Run group-level BP
    let group_assignments = run_group_level_bp(
        &spin_system_evidence,
        &group_sequential_links,
        &residue_types,
        sequence,
        &kde,
        params,
    );

    // 4. Map group assignments back to observations
    // Create a lookup from observation index to assigned position
    let mut obs_to_position: HashMap<usize, i32> = HashMap::new();

    for (group_idx, position, _confidence) in &group_assignments {
        let evidence = &spin_system_evidence[*group_idx];
        for &obs_idx in &evidence.observation_indices {
            let obs = &observations[obs_idx];
            // Determine target position based on intra vs inter
            let target_pos = if obs.intensity > 0.5 {
                *position  // Intra: observes group's residue
            } else {
                *position - 1  // Inter: observes PRECEDING residue
            };
            if target_pos > 0 {
                obs_to_position.insert(obs_idx, target_pos);
            }
        }
    }

    // Also add observations not in backbone groups (e.g., TOCSY-only protons)
    // These need to be mapped via correlation with backbone groups
    for (obs_idx, obs) in observations.iter().enumerate() {
        if obs_to_position.contains_key(&obs_idx) {
            continue;
        }
        // For now, leave unmapped observations with their current beliefs
        // TODO: Use TOCSY correlations to map sidechain protons
    }

    // 5. Apply group assignments to beliefs
    // This replaces the observation-level BP with deterministic assignment from groups
    for (obs_idx, &position) in &obs_to_position {
        // Set belief to strongly favor the assigned position
        for d in 0..domain_size {
            beliefs[*obs_idx][d] = if d as i32 == position { 0.95 } else { 0.05 / (domain_size - 1) as f64 };
        }
    }

    if params.verbose {
        println!("\n=== GROUP-LEVEL BP COMPLETE ===");
        println!("Mapped {} / {} observations from group assignments", obs_to_position.len(), observations.len());
        println!("================================\n");
    }

    // The old observation-level BP follows (now largely bypassed by group assignments)
    // Keep it for observations not mapped by groups

    // ==========================================================================
    // PERFORMANCE OPTIMIZATION: Precompute O(n²) operations once at max tolerance
    // Then filter/recompute only every 5 iterations instead of every iteration
    // ==========================================================================
    let cached_correlation_pairs = precompute_correlation_pairs(observations, tol_params);
    let cached_noesy_pairs = precompute_noesy_pairs(observations, tol_params);

    if params.verbose {
        println!("Cached {} correlation pairs and {} NOESY pairs at max tolerance",
            cached_correlation_pairs.len(), cached_noesy_pairs.len());
    }

    // Mutable state for cached O(n²) results (recomputed every 5 iterations)
    let mut current_correlation_scores: Vec<Vec<f64>> = vec![];
    let mut current_sequential_links: Vec<ObsSequentialLink> = vec![];
    let mut current_noesy_links: Vec<(usize, usize, f64)> = vec![];

    // OPTIMIZATION: Compute typing scores ONCE before the loop
    // The scoring function doesn't actually use iteration/tolerance parameters
    let typing_scores = compute_observation_typing_scores(
        observations, &residue_types, &kde, tol_params, 0, 100
    );

    let max_iterations = params.max_iterations;
    for iteration in 0..max_iterations {
        let progress = iteration as f64 / max_iterations as f64;
        let interp = params.interpolate(progress);

        // DEBUG: Print type confidence for each backbone group (iteration 0 only)
        if params.verbose && iteration == 0 {
            // DEBUG: Check typing scores for first few observations
            println!("\n=== INITIAL TYPING SCORES (first 3 obs) ===");
            for (idx, scores) in typing_scores.iter().enumerate().take(3) {
                let obs = &observations[idx];
                let (best_pos, &best_score) = scores.iter().enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .unwrap_or((0, &0.0));
                let pos0_score = scores[0];
                println!("  obs[{}] {:?}: pos0={:.4}, best_pos={} score={:.4}",
                    idx, obs.experiment_type, pos0_score, best_pos, best_score);
            }
            println!("==========================================\n");

            println!("\n=== TYPE CONFIDENCE PER BACKBONE GROUP ===");
            let threshold = params.sequence_type_confidence_threshold;
            let mut anchors: Vec<(usize, String, f64)> = Vec::new();
            for (group_idx, group) in backbone_groups.iter().enumerate() {
                // Aggregate typing scores for this group
                let mut agg_scores = vec![0.0; domain_size];
                for &obs_idx in group {
                    for (d, &score) in typing_scores[obs_idx].iter().enumerate() {
                        agg_scores[d] += score;
                    }
                }
                // Normalize
                let sum: f64 = agg_scores.iter().sum();
                if sum > 0.0 {
                    for s in &mut agg_scores { *s /= sum; }
                }
                // Compute type confidence for this group
                if let Some((best_type, confidence)) = compute_type_confidence(&agg_scores, &residue_types) {
                    if confidence > threshold {
                        anchors.push((group_idx, best_type.clone(), confidence));
                    }
                }
            }
            println!("Anchor candidates (confidence > {:.0}%):", threshold * 100.0);
            for (g, t, c) in &anchors {
                println!("  Group {}: {} ({:.1}%)", g, t, c * 100.0);
            }
            println!("Total anchors: {} / {} backbone groups", anchors.len(), backbone_groups.len());
            println!("==========================================\n");
        }

        // ==========================================================================
        // OPTIMIZATION: Only recompute O(n²) operations every 5 iterations
        // Tolerances anneal gradually, so reusing cached results is safe
        // ==========================================================================
        if iteration % 5 == 0 {
            // Recompute correlations from cached pairs (O(cached) instead of O(n²))
            current_correlation_scores = filter_correlations_from_cache(
                &cached_correlation_pairs, observations, tol_params, iteration, max_iterations
            );

            // Recompute sequential links (O(groups²) - smaller than correlation)
            current_sequential_links = compute_sequential_links(
                observations, tol_params, iteration, max_iterations
            );

            // Recompute NOESY links from cached pairs
            current_noesy_links = filter_noesy_from_cache(
                &cached_noesy_pairs, tol_params, iteration, max_iterations
            );
        }

        // Use current cached values (alias for readability)
        let correlation_scores = &current_correlation_scores;
        let sequential_links = &current_sequential_links;
        let noesy_links = &current_noesy_links;

        // DEBUG: Count significant correlations (iteration 0 only)
        if params.verbose && iteration == 0 {
            let mut n_correlations = 0;
            let mut total_strength = 0.0;
            for i in 0..observations.len() {
                for j in 0..observations.len() {
                    if i != j && correlation_scores[i][j] > 0.01 {
                        n_correlations += 1;
                        total_strength += correlation_scores[i][j];
                    }
                }
            }
            println!("Correlations: {} pairs with strength > 0.01, avg={:.3}",
                n_correlations, if n_correlations > 0 { total_strength / n_correlations as f64 } else { 0.0 });
        }

        // Debug: Print sequential links info (first iteration only)
        if params.verbose && iteration == 0 {
            let pos_links: Vec<_> = sequential_links.iter().filter(|l| l.strength > 0.0).collect();
            let neg_links: Vec<_> = sequential_links.iter().filter(|l| l.strength < 0.0).collect();
            println!("\n=== SEQUENTIAL LINKS ===");
            println!("Positive (sequential ordering): {}", pos_links.len());
            println!("Negative (same-residue): {}", neg_links.len());

            // Count unique backbone pairs involved in links
            let mut backbone_pairs: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();
            for link in &pos_links {
                // Get backbone group for each observation
                let from_group = backbone_groups.iter().position(|g| g.contains(&link.from_idx));
                let to_group = backbone_groups.iter().position(|g| g.contains(&link.to_idx));
                if let (Some(fg), Some(tg)) = (from_group, to_group) {
                    backbone_pairs.insert((fg.min(tg), fg.max(tg)));
                }
            }
            println!("Unique backbone group pairs: {}", backbone_pairs.len());
            println!("========================\n");
        }

        // Message passing update with both same-residue and sequential factors
        // Now includes Factor 3 (NOESY sequential) and Factor 5 (sequence-type constraint)
        let mut new_beliefs = update_observation_beliefs_with_sequential(
            &beliefs, &typing_scores, &correlation_scores, &sequential_links,
            &noesy_links, &type_to_positions, &residue_types,
            domain_size, interp.tocsy_weight, interp.typing_weight, interp.sequential_weight,
            interp.sequence_type_weight, interp.sequence_type_threshold
        );

        // DEBUG: Check beliefs after message passing (iteration 0)
        if params.verbose && iteration == 0 {
            println!("\n=== BELIEFS AFTER MESSAGE PASSING ===");
            for idx in 0..3.min(observations.len()) {
                let obs = &observations[idx];
                let belief = &new_beliefs[idx];
                let (best_pos, &best_prob) = belief.iter().enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .unwrap_or((0, &0.0));
                let pos0 = belief[0];
                println!("  obs[{}] {:?}: pos0={:.4}, best_pos={} prob={:.4}",
                    idx, obs.experiment_type, pos0, best_pos, best_prob);
            }
        }

        // BACKBONE GROUPING FACTOR: Observations in the same backbone group should have same belief
        // This synchronizes beliefs among HNCA/HNCACB/etc. from the same (H, N)
        apply_backbone_grouping_factor(&mut new_beliefs, &backbone_groups, domain_size);

        // DEBUG: Check beliefs after backbone grouping (iteration 0)
        if params.verbose && iteration == 0 {
            println!("\n=== BELIEFS AFTER BACKBONE GROUPING ===");
            for idx in 0..3.min(observations.len()) {
                let obs = &observations[idx];
                let belief = &new_beliefs[idx];
                let (best_pos, &best_prob) = belief.iter().enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .unwrap_or((0, &0.0));
                let pos0 = belief[0];
                // Find which backbone group this observation is in
                let group_id = backbone_groups.iter().position(|g| g.contains(&idx)).unwrap_or(999);
                let group_size = backbone_groups.get(group_id).map(|g| g.len()).unwrap_or(0);
                println!("  obs[{}] {:?}: pos0={:.4}, best_pos={} prob={:.4} (group {} size {})",
                    idx, obs.experiment_type, pos0, best_pos, best_prob, group_id, group_size);
            }
        }

        // SOFT CONSTRAINT: Backbone group competition - each residue prefers one backbone group
        // Use softer penalty (0.3x instead of 0.01x) during BP to allow recovery from early mistakes
        // Full uniqueness is enforced at extraction time
        apply_soft_backbone_group_uniqueness(&mut new_beliefs, &backbone_groups, domain_size);

        // DEBUG: Check beliefs after soft uniqueness (iteration 0)
        if params.verbose && iteration == 0 {
            println!("\n=== BELIEFS AFTER SOFT UNIQUENESS ===");
            for idx in 0..3.min(observations.len()) {
                let obs = &observations[idx];
                let belief = &new_beliefs[idx];
                let (best_pos, &best_prob) = belief.iter().enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .unwrap_or((0, &0.0));
                let pos0 = belief[0];
                println!("  obs[{}] {:?}: pos0={:.4}, best_pos={} prob={:.4}",
                    idx, obs.experiment_type, pos0, best_pos, best_prob);
            }
        }

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

    // DEBUG: Check backbone group composition and final beliefs
    if params.verbose {
        println!("\n=== BACKBONE GROUP COMPOSITION ===");
        let mut exp_in_groups: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        for group in &backbone_groups {
            for &idx in group {
                let exp = format!("{:?}", observations[idx].experiment_type);
                *exp_in_groups.entry(exp).or_default() += 1;
            }
        }
        println!("Observations in backbone groups by experiment type:");
        for (exp, count) in &exp_in_groups {
            println!("  {}: {}", exp, count);
        }

        // Check beliefs for first backbone group
        if let Some(group) = backbone_groups.first() {
            println!("\nFirst backbone group beliefs (max 3 members):");
            for &idx in group.iter().take(3) {
                let obs = &observations[idx];
                let belief = &beliefs[idx];
                let (best_pos, &best_prob) = belief.iter().enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .unwrap_or((0, &0.0));
                println!("  {:?} intensity={:.2}: best_pos={} prob={:.3}",
                    obs.experiment_type, obs.intensity, best_pos, best_prob);
            }
        }
        println!("==================================\n");
    }

    // =========================================================================
    // CHAIN-WALKING ALGORITHM
    // =========================================================================
    // After BP, use sequential links to propagate assignments from anchor points.
    // This is more deterministic than soft BP for triple-resonance data.
    //
    // Algorithm:
    // 1. Build graph of backbone groups connected by sequential links
    // 2. Identify anchors: backbone groups with high-confidence residue type
    // 3. For each anchor, find candidate positions (where that type occurs in sequence)
    // 4. Walk chain from anchors, assigning neighbors by sequential constraints
    // =========================================================================

    // Recompute typing scores for chain-walking (using final tolerances)
    let typing_scores = compute_observation_typing_scores(
        observations, &residue_types, &kde, tol_params, params.max_iterations, params.max_iterations
    );

    // Step 1: Build backbone group connectivity from sequential links
    // Each backbone group is a node; sequential links create directed edges
    // AGGREGATE link strengths: if CA AND CB both match, that's stronger evidence!

    // Build sequential link index from observation index to backbone group
    // IMPORTANT: Include BOTH intra and inter observations!
    // Inter observations aren't in backbone_groups directly, but we can find their
    // associated backbone group by matching (H, N) coordinates.
    let mut obs_to_group: HashMap<usize, usize> = HashMap::new();

    // First: direct mapping for observations in backbone groups
    for (group_idx, group) in backbone_groups.iter().enumerate() {
        for &obs_idx in group {
            obs_to_group.insert(obs_idx, group_idx);
        }
    }

    // Second: map inter observations (not in backbone_groups) to their backbone group
    // by matching (H, N) coordinates
    let h_tolerance = 0.03;
    let n_tolerance = 0.3;

    for (obs_idx, obs) in observations.iter().enumerate() {
        if obs_to_group.contains_key(&obs_idx) {
            continue;  // Already mapped
        }

        // Try to find this observation's backbone (H, N)
        let h = obs.dimensions.iter()
            .find(|d| d.nucleus == NucleusType::H1)
            .map(|d| d.shift);
        let n = obs.dimensions.iter()
            .find(|d| d.nucleus == NucleusType::N15)
            .map(|d| d.shift);

        if let (Some(h_shift), Some(n_shift)) = (h, n) {
            // Find the backbone group with matching (H, N)
            for (group_idx, group) in backbone_groups.iter().enumerate() {
                if let Some((ref_h, ref_n)) = get_backbone_hn_from_obs(&observations[group[0]]) {
                    if (h_shift - ref_h).abs() < h_tolerance && (n_shift - ref_n).abs() < n_tolerance {
                        obs_to_group.insert(obs_idx, group_idx);
                        break;
                    }
                }
            }
        }
    }

    if params.verbose {
        let mapped_obs = obs_to_group.len();
        println!("Observations mapped to backbone groups: {} / {}", mapped_obs, observations.len());
    }

    // Aggregate sequential links at group level
    let sequential_links = compute_sequential_links(
        observations, tol_params, params.max_iterations, params.max_iterations
    );

    // First pass: aggregate link strengths per group pair
    // Key: (from_group, to_group) -> (sum_strength, count, max_strength)
    let mut pair_strengths: HashMap<(usize, usize), (f64, usize, f64)> = HashMap::new();

    for link in &sequential_links {
        if link.strength <= 0.0 { continue; }  // Only use sequential (positive) links

        let Some(&from_group) = obs_to_group.get(&link.from_idx) else { continue };
        let Some(&to_group) = obs_to_group.get(&link.to_idx) else { continue };

        if from_group == to_group { continue; }  // Skip same-group links

        let entry = pair_strengths.entry((from_group, to_group)).or_insert((0.0, 0, 0.0));
        entry.0 += link.strength;  // Sum
        entry.1 += 1;              // Count
        entry.2 = entry.2.max(link.strength);  // Max
    }

    // Second pass: build group links with aggregated strength
    // Use: sum * sqrt(count) to reward multiple confirming links
    let mut group_links: HashMap<usize, Vec<(usize, f64)>> = HashMap::new();
    let mut reverse_links: HashMap<usize, Vec<(usize, f64)>> = HashMap::new();

    for ((from_group, to_group), (sum, count, max)) in &pair_strengths {
        // Aggregated strength: reward multiple matching carbons
        // If CA and CB both match perfectly (strength=1.0 each), agg_strength = 2.0 * sqrt(2) ≈ 2.83
        // If only one matches perfectly, agg_strength = 1.0
        let agg_strength = sum * ((*count as f64).sqrt());

        group_links.entry(*from_group).or_default().push((*to_group, agg_strength));
        reverse_links.entry(*to_group).or_default().push((*from_group, agg_strength));
    }

    if params.verbose {
        // Count high-confidence links (multiple carbons matching)
        let multi_carbon_links = pair_strengths.values().filter(|(_, count, _)| *count >= 2).count();
        let perfect_links = pair_strengths.values().filter(|(_, _, max)| *max > 0.99).count();

        // Aggregated strength distribution
        let agg_strengths: Vec<f64> = pair_strengths.values()
            .map(|(sum, count, _)| sum * ((*count as f64).sqrt()))
            .collect();
        let above_4 = agg_strengths.iter().filter(|&&s| s >= 4.0).count();
        let above_3 = agg_strengths.iter().filter(|&&s| s >= 3.0).count();
        let above_2 = agg_strengths.iter().filter(|&&s| s >= 2.0).count();

        println!("\n=== GROUP-LEVEL SEQUENTIAL LINKS ===");
        println!("Total group pairs: {}", pair_strengths.len());
        println!("Multi-carbon links (count >= 2): {}", multi_carbon_links);
        println!("Perfect matches (strength > 0.99): {}", perfect_links);
        println!("Aggregated strength distribution:");
        println!("  >= 4.0: {}", above_4);
        println!("  >= 3.0: {}", above_3);
        println!("  >= 2.0: {}", above_2);
        println!("====================================\n");
    }

    // Step 2: Identify anchors with high-confidence typing
    let anchor_threshold = 0.40;  // 40% confidence threshold for anchors
    let mut anchors: Vec<(usize, String, f64, Vec<usize>)> = Vec::new();  // (group_idx, type, confidence, candidate_positions)

    for (group_idx, group) in backbone_groups.iter().enumerate() {
        // Aggregate typing scores for this group
        let mut agg_scores = vec![0.0f64; domain_size];
        for &obs_idx in group {
            for (d, &score) in typing_scores[obs_idx].iter().enumerate() {
                agg_scores[d] += score;
            }
        }
        // Normalize
        let sum: f64 = agg_scores.iter().sum();
        if sum > 0.0 {
            for s in &mut agg_scores { *s /= sum; }
        }

        // Find best type and confidence
        if let Some((best_type, confidence)) = compute_type_confidence(&agg_scores, &residue_types) {
            if confidence >= anchor_threshold {
                // Find which sequence positions have this residue type
                let candidate_positions: Vec<usize> = residue_types.iter()
                    .enumerate()
                    .filter(|(_, t)| *t == &best_type)
                    .map(|(pos, _)| pos + 1)  // 1-indexed
                    .collect();

                if !candidate_positions.is_empty() {
                    anchors.push((group_idx, best_type, confidence, candidate_positions));
                }
            }
        }
    }

    // Sort anchors by uniqueness (fewer candidates = more unique), then by confidence
    anchors.sort_by(|a, b| {
        // First criterion: fewer candidate positions = more unique (higher priority)
        let uniqueness_cmp = a.3.len().cmp(&b.3.len());
        if uniqueness_cmp != std::cmp::Ordering::Equal {
            return uniqueness_cmp;
        }
        // Second criterion: higher confidence
        b.2.partial_cmp(&a.2).unwrap()
    });

    // Step 2b: Find chain distances between anchor groups
    // This helps disambiguate which anchor is at which position
    let min_link_strength = 4.0;  // Only use high-confidence links for chain discovery

    // BFS to find shortest path (in terms of sequential steps) between groups
    fn find_chain_distance(
        from_group: usize,
        to_group: usize,
        forward_links: &HashMap<usize, Vec<(usize, f64)>>,
        min_strength: f64,
        max_distance: usize,
    ) -> Option<i32> {
        use std::collections::VecDeque;
        let mut visited = HashSet::new();
        let mut queue: VecDeque<(usize, i32)> = VecDeque::new();  // (group, distance)

        queue.push_back((from_group, 0));
        visited.insert(from_group);

        while let Some((current, dist)) = queue.pop_front() {
            if current == to_group {
                return Some(dist);
            }
            if dist as usize >= max_distance {
                continue;
            }

            if let Some(neighbors) = forward_links.get(&current) {
                for &(next, strength) in neighbors {
                    if strength >= min_strength && !visited.contains(&next) {
                        visited.insert(next);
                        queue.push_back((next, dist + 1));
                    }
                }
            }
        }
        None
    }

    // Find chain distances between GLY anchors (they're the most constrained)
    let gly_anchors: Vec<_> = anchors.iter()
        .filter(|(_, t, _, _)| t == "GLY")
        .collect();

    // Find groups with only one strong outgoing link (chain start candidates)
    // and groups with only one strong incoming link (chain end candidates)
    let mut out_degree: HashMap<usize, usize> = HashMap::new();
    let mut in_degree: HashMap<usize, usize> = HashMap::new();

    for (&from_group, neighbors) in &group_links {
        let strong_out = neighbors.iter().filter(|(_, s)| *s >= min_link_strength).count();
        *out_degree.entry(from_group).or_default() += strong_out;
    }
    for (&to_group, neighbors) in &reverse_links {
        let strong_in = neighbors.iter().filter(|(_, s)| *s >= min_link_strength).count();
        *in_degree.entry(to_group).or_default() += strong_in;
    }

    // Chain starts: out_degree > 0, in_degree == 0 (or not in in_degree)
    let chain_starts: Vec<usize> = (0..backbone_groups.len())
        .filter(|&g| out_degree.get(&g).copied().unwrap_or(0) > 0 &&
                     in_degree.get(&g).copied().unwrap_or(0) == 0)
        .collect();

    // Find longest chain from each start
    fn find_longest_chain(
        start: usize,
        forward_links: &HashMap<usize, Vec<(usize, f64)>>,
        min_strength: f64,
    ) -> Vec<usize> {
        let mut chain = vec![start];
        let mut visited = HashSet::new();
        visited.insert(start);
        let mut current = start;

        loop {
            let best_next = forward_links.get(&current)
                .map(|neighbors| {
                    neighbors.iter()
                        .filter(|(n, s)| *s >= min_strength && !visited.contains(n))
                        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                })
                .flatten();

            match best_next {
                Some(&(next, _)) => {
                    chain.push(next);
                    visited.insert(next);
                    current = next;
                }
                None => break,
            }
        }
        chain
    }

    let mut longest_chain: Vec<usize> = Vec::new();
    for &start in &chain_starts {
        let chain = find_longest_chain(start, &group_links, min_link_strength);
        if chain.len() > longest_chain.len() {
            longest_chain = chain;
        }
    }

    if params.verbose {
        println!("\n=== CHAIN ANALYSIS ===");
        println!("Chain start candidates (out>0, in=0): {:?}", chain_starts);
        println!("Longest chain length: {}", longest_chain.len());
        if longest_chain.len() <= 20 {
            println!("Longest chain: {:?}", longest_chain);
        } else {
            println!("Longest chain (first 10): {:?}", &longest_chain[..10]);
            println!("Longest chain (last 10): {:?}", &longest_chain[longest_chain.len()-10..]);
        }

        // Validate chain against ground truth (if available)
        let mut ground_truth_positions: Vec<Option<usize>> = Vec::new();
        for &group_idx in &longest_chain {
            // Get ground truth from first observation in this group
            let gt_pos = backbone_groups.get(group_idx)
                .and_then(|group| group.first())
                .and_then(|&obs_idx| observations.get(obs_idx))
                .and_then(|obs| obs.ground_truth.as_ref())
                .map(|gt| gt.residue_position);
            ground_truth_positions.push(gt_pos);
        }

        // Check if chain is sequential in ground truth
        let valid_gt: Vec<usize> = ground_truth_positions.iter()
            .filter_map(|&p| p)
            .collect();
        println!("Ground truth positions found: {} / {} groups", valid_gt.len(), longest_chain.len());
        if !valid_gt.is_empty() {
            let is_sequential = valid_gt.windows(2).all(|w| w[1] == w[0] + 1);
            println!("  First 10: {:?}", &valid_gt[..valid_gt.len().min(10)]);
            println!("  Last 10: {:?}", &valid_gt[valid_gt.len().saturating_sub(10)..]);
            println!("  Chain is sequential in GT: {}", is_sequential);
        }
        println!("======================\n");
    }

    if params.verbose && gly_anchors.len() >= 2 {
        println!("\n=== ANCHOR CHAIN DISTANCES ===");
        for i in 0..gly_anchors.len() {
            for j in (i+1)..gly_anchors.len() {
                let (g1, _, _, _) = gly_anchors[i];
                let (g2, _, _, _) = gly_anchors[j];
                let fwd_dist = find_chain_distance(*g1, *g2, &group_links, min_link_strength, 100);
                let bwd_dist = find_chain_distance(*g2, *g1, &group_links, min_link_strength, 100);
                if fwd_dist.is_some() || bwd_dist.is_some() {
                    println!("  GLY Group {} <-> GLY Group {}: fwd={:?}, bwd={:?}",
                        g1, g2, fwd_dist, bwd_dist);
                }
            }
        }
        println!("Expected GLY distances: 10->35=25, 35->47=12, 47->53=6");
        println!("==============================\n");
    }

    if params.verbose {
        println!("\n=== CHAIN-WALKING ANCHORS ===");
        println!("Found {} anchors (confidence >= {:.0}%)", anchors.len(), anchor_threshold * 100.0);
        for (group_idx, residue_type, conf, positions) in anchors.iter().take(10) {
            println!("  Group {}: {} ({:.1}%) -> positions {:?}",
                group_idx, residue_type, conf * 100.0, positions);
        }
        println!("=============================\n");
    }

    // Step 3: Use proline positions to identify chain boundaries
    // Prolines have no backbone NH, so they create natural breaks in triple-resonance chains
    let proline_positions: Vec<usize> = residue_types.iter()
        .enumerate()
        .filter(|(_, t)| *t == "PRO")
        .map(|(i, _)| i + 1)  // 1-indexed
        .collect();

    // Chain segments are ranges between prolines (and sequence start/end)
    // Position 1 (Met) also has no backbone NH, so chains start at position 2
    let mut chain_segments: Vec<(usize, usize)> = Vec::new();  // (start_pos, end_pos) inclusive
    let mut segment_start = 2;  // Skip position 1 (N-terminus)

    for &pro_pos in &proline_positions {
        if pro_pos > segment_start {
            chain_segments.push((segment_start, pro_pos - 1));
        }
        segment_start = pro_pos + 1;
    }
    // Add final segment
    if segment_start <= sequence.len() {
        chain_segments.push((segment_start, sequence.len()));
    }

    if params.verbose {
        println!("\n=== PROLINE-BASED CHAIN SEGMENTS ===");
        println!("Proline positions: {:?}", proline_positions);
        println!("Chain segments (residue ranges):");
        for (i, (start, end)) in chain_segments.iter().enumerate() {
            println!("  Segment {}: positions {}-{} ({} residues)", i, start, end, end - start + 1);
        }
        println!("=====================================\n");
    }

    // Initialize assignment tracking
    let mut group_assignments: HashMap<usize, i32> = HashMap::new();
    let mut position_assigned: HashSet<i32> = HashSet::new();

    // Mark proline positions as assigned (they have no backbone groups)
    for &pro_pos in &proline_positions {
        position_assigned.insert(pro_pos as i32);
    }
    // Also mark position 1 (N-terminus has no backbone NH)
    position_assigned.insert(1);

    // Match discovered chains to segments by length and assign directly
    // If the longest chain has 32 groups and segment 39-70 has 32 positions, we can assign directly!
    if !longest_chain.is_empty() {
        let chain_len = longest_chain.len();
        let matching_segment = chain_segments.iter()
            .find(|(start, end)| (*end - *start + 1) == chain_len);

        if let Some(&(seg_start, _seg_end)) = matching_segment {
            if params.verbose {
                println!("Longest chain ({} groups) matches segment starting at position {}", chain_len, seg_start);
                println!("Directly assigning chain to segment positions...");
            }

            // Directly assign chain groups to segment positions
            for (i, &group_idx) in longest_chain.iter().enumerate() {
                let position = (seg_start + i) as i32;
                group_assignments.insert(group_idx, position);
                position_assigned.insert(position);
            }

            if params.verbose {
                println!("Assigned {} groups from longest chain", longest_chain.len());

                // Verify: print first 10 assignments with their (H, N) shifts
                println!("Verifying chain assignments (first 10):");
                for (i, &group_idx) in longest_chain.iter().take(10).enumerate() {
                    let position = seg_start + i;
                    if let Some((h, n)) = backbone_groups.get(group_idx)
                        .and_then(|g| g.first())
                        .and_then(|&obs_idx| observations.get(obs_idx))
                        .and_then(|obs| {
                            let h = obs.dimensions.iter().find(|d| d.nucleus == NucleusType::H1).map(|d| d.shift)?;
                            let n = obs.dimensions.iter().find(|d| d.nucleus == NucleusType::N15).map(|d| d.shift)?;
                            Some((h, n))
                        })
                    {
                        let seq_char = sequence.chars().nth(position - 1).unwrap_or('?');
                        println!("  Group {} -> pos {} ({}): H={:.3}, N={:.2}",
                            group_idx, position, seq_char, h, n);
                    }
                }
            }
        }
    }

    // Step 4: Chain-walking from anchors (for remaining unassigned groups)

    // Helper: Walk chain in one direction from a starting group
    fn walk_chain(
        start_group: usize,
        start_pos: i32,
        links: &HashMap<usize, Vec<(usize, f64)>>,
        domain_size: usize,
        group_assignments: &mut HashMap<usize, i32>,
        position_assigned: &mut HashSet<i32>,
        direction: i32,  // +1 for forward, -1 for backward
    ) -> usize {
        let mut current_group = start_group;
        let mut current_pos = start_pos;
        let mut assigned_count = 0;

        loop {
            let next_pos = current_pos + direction;
            if next_pos < 1 || next_pos >= domain_size as i32 {
                break;  // Out of bounds
            }

            // Find next group with strongest link
            let neighbors = links.get(&current_group);
            if neighbors.is_none() || neighbors.unwrap().is_empty() {
                break;
            }

            // Find best unassigned neighbor with sufficient strength
            // Require aggregated strength >= 2.5 (roughly 2 carbons matching at 0.99+ strength)
            let min_link_strength = 2.5;
            let best_neighbor = neighbors.unwrap().iter()
                .filter(|(ng, strength)| !group_assignments.contains_key(ng) && *strength >= min_link_strength)
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            let Some(&(next_group, _strength)) = best_neighbor else { break };

            // Check if next position is available
            if position_assigned.contains(&next_pos) {
                break;  // Position already taken
            }

            // Assign next group to next position
            group_assignments.insert(next_group, next_pos);
            position_assigned.insert(next_pos);
            assigned_count += 1;

            current_group = next_group;
            current_pos = next_pos;
        }

        assigned_count
    }

    // Process anchors in order of quality
    for (group_idx, _residue_type, _conf, candidate_positions) in &anchors {
        if group_assignments.contains_key(group_idx) {
            continue;  // Already assigned by previous chain walk
        }

        // Find first available candidate position
        let Some(&chosen_pos) = candidate_positions.iter()
            .find(|&&pos| !position_assigned.contains(&(pos as i32)))
        else {
            continue;  // No available positions for this type
        };

        // Assign this anchor
        group_assignments.insert(*group_idx, chosen_pos as i32);
        position_assigned.insert(chosen_pos as i32);

        // Walk forward (using group_links: from -> to means from precedes to)
        walk_chain(*group_idx, chosen_pos as i32, &group_links, domain_size,
            &mut group_assignments, &mut position_assigned, 1);

        // Walk backward (using reverse_links: to -> from means from precedes to)
        walk_chain(*group_idx, chosen_pos as i32, &reverse_links, domain_size,
            &mut group_assignments, &mut position_assigned, -1);
    }

    if params.verbose {
        println!("\n=== CHAIN-WALKING RESULTS ===");
        println!("Assigned {} / {} backbone groups", group_assignments.len(), backbone_groups.len());

        // Show first 20 assignments
        let mut sorted_assignments: Vec<_> = group_assignments.iter().collect();
        sorted_assignments.sort_by_key(|(_, &pos)| pos);
        for (group_idx, pos) in sorted_assignments.iter().take(20) {
            let seq_char = if (**pos > 0) && ((**pos as usize) <= sequence.len()) {
                sequence.chars().nth(**pos as usize - 1).unwrap_or('?')
            } else { '?' };
            println!("  Group {} -> position {} ({})", group_idx, pos, seq_char);
        }
        println!("=============================\n");
    }

    // Step 5: DISABLED - chain-walking override
    // The group-level BP already set beliefs at lines 3826-3831.
    // The chain-walking code here was creating a SECOND set of assignments that
    // overrode the group-level BP results, causing conflicts.
    // Let's rely solely on group-level BP results for now.

    // TODO: Integrate chain-walking as a POST-HOC verification step, not an override

    // Extract assignments with backbone uniqueness constraint.
    //
    // Key insight: Different experiment types provide DIFFERENT information about a residue.
    // - HSQC-15N: only H-N correlation
    // - HNCA: H-N + CA carbon (more discriminative)
    // - HNCACO: H-N + CO carbon (carbonyl specific)
    //
    // Therefore, we maintain per-experiment-type uniqueness buckets so that the SAME residue
    // can be assigned by multiple experiment types (each providing complementary evidence).
    // If we merged all into one bucket, only one experiment could "claim" each residue.
    //
    // Future: This could be generalized to per-evidence-signature buckets based on what
    // atoms are observed, but per-experiment-type works well in practice.
    let mut assigned_hsqc_residues: HashSet<i32> = HashSet::new();
    let mut assigned_hnca_residues: HashSet<i32> = HashSet::new();
    let mut assigned_hncaco_residues: HashSet<i32> = HashSet::new();
    let mut results: Vec<ObservationAssignmentResult> = Vec::with_capacity(observations.len());

    // Helper to check if an observation is eligible for the backbone uniqueness pass.
    // Uses physics-based check: must have intra-residue backbone H-N anchor.
    // The experiment_type is used to route to the correct uniqueness bucket.
    let is_intra_backbone_hn = |obs: &Observation| -> bool {
        // Physics-based check: has backbone H-N anchor and all dimensions are intra-residue
        obs.is_nitrogen_anchored() && obs.is_purely_intra()
    };

    // Sort all backbone-type peaks by confidence for greedy assignment
    let mut backbone_indices: Vec<(usize, f64)> = observations.iter().enumerate()
        .filter(|(_, obs)| is_intra_backbone_hn(obs))
        .map(|(idx, _)| {
            let best_prob = beliefs[idx].iter().skip(1).fold(0.0f64, |a, &b| a.max(b));
            (idx, best_prob)
        })
        .collect();
    backbone_indices.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // First pass: Assign backbone peaks greedily with per-experiment-type uniqueness
    // Each experiment type maintains its own "one peak per residue" constraint
    let mut assigned: HashSet<usize> = HashSet::new();
    for (obs_idx, _) in &backbone_indices {
        let belief = &beliefs[*obs_idx];
        let obs = &observations[*obs_idx];

        // Get the appropriate uniqueness set based on experiment type
        // (Future: derive from atom_constraint signature instead)
        let assigned_residues = match obs.experiment_type {
            PeakExperimentType::Hsqc15N => &assigned_hsqc_residues,
            PeakExperimentType::Hnca => &assigned_hnca_residues,
            PeakExperimentType::Hncaco => &assigned_hncaco_residues,
            _ => &assigned_hsqc_residues,  // Default to HSQC bucket
        };

        // Find best available residue (not already assigned by THIS experiment type)
        let best = belief.iter().enumerate()
            .skip(1)  // Skip unassigned (index 0)
            .filter(|(r, _)| !assigned_residues.contains(&(*r as i32)))
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap());

        let (best_idx, best_prob) = match best {
            Some((idx, &prob)) => (idx, prob),
            None => (0, belief[0]),  // Fall back to unassigned if all residues taken
        };

        // Update the appropriate uniqueness set
        match obs.experiment_type {
            PeakExperimentType::Hsqc15N => { assigned_hsqc_residues.insert(best_idx as i32); }
            PeakExperimentType::Hnca => { assigned_hnca_residues.insert(best_idx as i32); }
            PeakExperimentType::Hncaco => { assigned_hncaco_residues.insert(best_idx as i32); }
            _ => {}
        }
        assigned.insert(*obs_idx);

        results.push(ObservationAssignmentResult {
            observation_id: obs.id,
            assigned_residue: best_idx as i32,
            confidence: best_prob,
            experiment_type: obs.experiment_type,
        });
    }

    // Second pass: Assign all other observations (no uniqueness constraint)
    for (obs_idx, obs) in observations.iter().enumerate() {
        if assigned.contains(&obs_idx) {
            continue;  // Already assigned in backbone pass
        }

        let belief = &beliefs[obs_idx];

        // Find best position (including unassigned at index 0)
        let (best_idx, &best_prob) = belief.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap_or((0, &0.0));

        // FIX: Don't let "unassigned" win when a real position has significant belief.
        // If best is unassigned (idx 0) but second-best position has belief > 0.2,
        // prefer the actual position. This handles end-of-sequence cases where
        // sequential evidence is weak but typing evidence is strong.
        let (final_idx, final_prob) = if best_idx == 0 {
            // Find best NON-unassigned position
            let best_real = belief.iter().enumerate()
                .skip(1)  // Skip unassigned
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap());

            if let Some((real_idx, &real_prob)) = best_real {
                if real_prob > 0.2 {
                    // Significant belief for a real position - use it
                    (real_idx, real_prob)
                } else {
                    (best_idx, best_prob)
                }
            } else {
                (best_idx, best_prob)
            }
        } else {
            (best_idx, best_prob)
        };

        results.push(ObservationAssignmentResult {
            observation_id: obs.id,
            assigned_residue: final_idx as i32,
            confidence: final_prob,
            experiment_type: obs.experiment_type,
        });
    }

    // Re-sort results by original observation order for consistency
    results.sort_by_key(|r| observations.iter().position(|o| o.id == r.observation_id).unwrap_or(0));

    // Debug: Print backbone peak assignments
    if params.verbose {
        println!("\n=== BACKBONE PEAK ASSIGNMENTS ===");
        for (obs, result) in observations.iter().zip(results.iter()) {
            if is_intra_backbone_hn(obs) {
                // Show experiment type for debugging (still useful for humans)
                let exp_name = format!("{:?}", obs.experiment_type);
                let shifts: Vec<_> = obs.dimensions.iter()
                    .map(|d| format!("{:?}={:.3}", d.nucleus, d.shift))
                    .collect();
                println!("  {}: {} -> residue {} (conf={:.3})",
                    exp_name, shifts.join(", "), result.assigned_residue, result.confidence);
            }
        }
        println!("================================\n");

        // Debug: Print CG-range carbon assignments
        println!("=== CG-RANGE CARBON ASSIGNMENTS ===");
        for (obs_idx, (obs, result)) in observations.iter().zip(results.iter()).enumerate() {
            let c_shift = obs.dimensions.iter()
                .find(|d| d.nucleus == NucleusType::C13)
                .map(|d| d.shift);
            if let Some(c) = c_shift {
                if c > 15.0 && c < 25.0 {
                    let h_shift = obs.dimensions.iter()
                        .find(|d| d.nucleus == NucleusType::H1)
                        .map(|d| d.shift)
                        .unwrap_or(0.0);
                    let seq_char = if result.assigned_residue > 0 && (result.assigned_residue as usize) <= sequence.len() {
                        sequence.chars().nth(result.assigned_residue as usize - 1).unwrap_or('?')
                    } else { '?' };
                    println!("  obs[{}] C={:.2} H={:.2} -> res {} ({}) conf={:.3}",
                        obs_idx, c, h_shift, result.assigned_residue, seq_char, result.confidence);
                }
            }
        }
        println!("===================================\n");
    }

    results
}

/// Compute typing scores for observations based on chemical shifts.
/// Now offset-aware: passes full residue_types array so inter-residue dimensions
/// (PrecedingResidue, FollowingResidue) can be scored against the correct residue types.
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

        // Check if this observation has backbone H/N (requires both Intra H1 and Intra N15)
        // This is used for N-terminus exclusion: position 1 has no backbone NH
        let has_backbone_hn = obs.dimensions.iter().any(|d|
            d.nucleus == NucleusType::H1 &&
            d.residue_offset == ResidueOffset::Intra &&
            matches!(&d.atom_constraint, AtomConstraint::Exact(s) if s == "H")
        ) && obs.dimensions.iter().any(|d|
            d.nucleus == NucleusType::N15 &&
            d.residue_offset == ResidueOffset::Intra &&
            matches!(&d.atom_constraint, AtomConstraint::Exact(s) if s == "N")
        );

        // Score against each residue POSITION (1-indexed)
        // The function now uses ResidueOffset to score each dimension against the
        // appropriate residue type (i for Intra, i-1 for PrecedingResidue, etc.)
        for r in 0..residue_types.len() {
            let candidate_pos = r + 1;  // 1-indexed position

            // N-TERMINUS EXCLUSION: Position 1 cannot have backbone NH
            // The N-terminus has -NH3+ instead of backbone amide NH
            if has_backbone_hn && candidate_pos == 1 {
                scores[candidate_pos] = 1e-20;  // Effectively zero
                continue;
            }

            let score = score_observation_for_residue_type(
                obs, candidate_pos, residue_types, kde, tol_params, iteration, max_iterations
            );
            scores[candidate_pos] = score.max(1e-10);
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

/// Score how well an observation matches a residue POSITION.
///
/// Physics-based: uses atom_constraint AND residue topology for hard constraints.
/// Now offset-aware: dimensions with ResidueOffset::PrecedingResidue score against
/// the preceding residue type (i-1), not the candidate position.
///
/// HARD CONSTRAINT: If an atom doesn't exist in the target residue type's topology,
/// that dimension contributes 0 probability.
fn score_observation_for_residue_type(
    obs: &Observation,
    candidate_pos: usize,        // 1-indexed position being evaluated
    residue_types: &[String],    // Full sequence (0-indexed)
    kde: &KDEDatabase,
    _tol_params: &NucleusToleranceParams,
    _iteration: usize,
    _max_iterations: usize,
) -> f64 {
    let mut log_score = 0.0;
    let debug_this = false;  // Disable verbose debug

    for dim in &obs.dimensions {
        // Determine which residue type this dimension should be scored against
        // based on its ResidueOffset - this is the key fix for inter-residue observations
        let target_res_type: Option<&str> = match dim.residue_offset {
            ResidueOffset::Intra => {
                // Score against candidate position
                residue_types.get(candidate_pos - 1).map(|s| s.as_str())
            },
            ResidueOffset::PrecedingResidue => {
                // Score against position before candidate (i-1)
                if candidate_pos >= 2 {
                    residue_types.get(candidate_pos - 2).map(|s| s.as_str())
                } else {
                    None  // No preceding residue for position 1
                }
            },
            ResidueOffset::FollowingResidue => {
                // Score against position after candidate (i+1)
                residue_types.get(candidate_pos).map(|s| s.as_str())
            },
            ResidueOffset::Unknown => {
                // Default to candidate position (e.g., NOESY where relationship is ambiguous)
                residue_types.get(candidate_pos - 1).map(|s| s.as_str())
            },
        };

        // If no valid target residue (edge case like position 1 for PrecedingResidue),
        // apply a penalty but don't completely reject
        let Some(res_type) = target_res_type else {
            // Edge case penalty: this position can't satisfy this dimension's offset
            // Use a very small probability in log space
            log_score += 1e-10_f64.ln();  // Strong but not absolute penalty
            continue;
        };

        // Get topology for the TARGET residue type (not necessarily candidate!)
        let topology = get_topology_by_three(res_type);

        // Physics-based: use atom_constraint instead of nucleus-based guessing
        let atom_candidates = atoms_from_constraint(&dim.atom_constraint, dim.nucleus);

        // HARD CONSTRAINT: Filter to atoms that actually exist in this residue type
        let valid_atoms: Vec<&str> = if let Some(topo) = topology {
            atom_candidates.iter()
                .filter(|&atom| topo.has_atom(atom))
                .copied()
                .collect()
        } else {
            // No topology found - fall back to all candidates
            atom_candidates.clone()
        };

        // HARD CONSTRAINT: If no valid atoms exist, this assignment is impossible
        if valid_atoms.is_empty() && topology.is_some() {
            if debug_this {
                println!("  HARD CONSTRAINT: pos {} offset={:?} -> {} has no atoms {:?} (candidates were {:?})",
                         candidate_pos, dim.residue_offset, res_type, valid_atoms, atom_candidates);
            }
            return 0.0;  // Impossible assignment - residue doesn't have these atoms
        }

        // Score using KDE among valid atoms only
        let mut best_density = 1e-10;
        let mut best_atom = "";

        for atom in &valid_atoms {
            let density = kde.density(res_type, atom, dim.shift);
            if density > best_density {
                best_density = density;
                best_atom = atom;
            }
        }

        if debug_this {
            println!("  SCORE: pos {} offset={:?} -> {} {:?} constraint={:?} shift={:.2} -> valid_atoms={:?}, best={}:{:.2e}",
                     candidate_pos, dim.residue_offset, res_type, dim.nucleus, dim.atom_constraint, dim.shift, valid_atoms, best_atom, best_density);
        }

        log_score += best_density.max(1e-10).ln();
    }

    let score = log_score.exp();
    if debug_this {
        println!("  SCORE: pos {} log={:.4} -> exp={:.2e}", candidate_pos, log_score, score);
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
                    // Backbone
                    "CA" => result.push("CA"),
                    "CB" => result.push("CB"),
                    "H" | "HN" => result.push("H"),
                    "HA" => {
                        result.push("HA");
                        result.push("HA2");
                        result.push("HA3");
                    }
                    "HA2" => result.push("HA2"),
                    "HA3" => result.push("HA3"),
                    "HB" => {
                        result.push("HB");
                        result.push("HB2");
                        result.push("HB3");
                    }
                    "HB2" => result.push("HB2"),
                    "HB3" => result.push("HB3"),
                    "N" => result.push("N"),
                    "C" => result.push("C"),
                    // Sidechain carbons (aliphatic + aromatic)
                    "CG" => result.push("CG"),
                    "CG1" => result.push("CG1"),
                    "CG2" => result.push("CG2"),
                    "CD" => result.push("CD"),
                    "CD1" => result.push("CD1"),
                    "CD2" => result.push("CD2"),
                    "CE" => result.push("CE"),
                    "CE1" => result.push("CE1"),
                    "CE2" => result.push("CE2"),
                    "CE3" => result.push("CE3"),
                    "CZ" => result.push("CZ"),
                    "CZ2" => result.push("CZ2"),
                    "CZ3" => result.push("CZ3"),
                    "CH2" => result.push("CH2"),
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
    // First check: if they share the same (C13, H1_direct) pair, they're the same carbon
    // This is a HARD constraint for 3D HSQC-TOCSY observations
    if obs_a.transfer_pathway == TransferPathway::ThroughBond
       && obs_b.transfer_pathway == TransferPathway::ThroughBond
    {
        // Get carbon shifts (if present)
        let ca_c = obs_a.dimensions.iter()
            .find(|d| d.nucleus == NucleusType::C13)
            .map(|d| d.shift);
        let cb_c = obs_b.dimensions.iter()
            .find(|d| d.nucleus == NucleusType::C13)
            .map(|d| d.shift);

        // For 3D HSQC-TOCSY-13C: dimensions are (C13, H1_direct, H1_tocsy)
        // The FIRST H1 dimension is the direct-attached proton
        let ha_protons = get_proton_shifts(obs_a);
        let hb_protons = get_proton_shifts(obs_b);

        // SAME-CARBON CONSTRAINT: If both have C13 AND first H1 matches, they're the same carbon!
        if let (Some(c_a), Some(c_b)) = (ca_c, cb_c) {
            if !ha_protons.is_empty() && !hb_protons.is_empty() {
                let h_a_direct = ha_protons[0];  // First proton is direct-attached
                let h_b_direct = hb_protons[0];

                // Check if same (C, H_direct) pair - HARD constraint, very high score
                if let (Some(c_score), Some(h_score)) = (
                    shifts_match(c_a, c_b, c_tol),
                    shifts_match(h_a_direct, h_b_direct, h_tol)
                ) {
                    // Same carbon atom! Return strong correlation (near 1.0)
                    return (c_score + h_score) / 2.0;
                }
            }
        }

        // Fall back to standard TOCSY correlation: require BOTH protons to match
        if ha_protons.len() >= 2 && hb_protons.len() >= 2 {
            // Check both orientations
            let match_direct = shifts_match(ha_protons[0], hb_protons[0], h_tol)
                .and_then(|s1| shifts_match(ha_protons[1], hb_protons[1], h_tol).map(|s2| (s1 + s2) / 2.0));
            let match_flipped = shifts_match(ha_protons[0], hb_protons[1], h_tol)
                .and_then(|s1| shifts_match(ha_protons[1], hb_protons[0], h_tol).map(|s2| (s1 + s2) / 2.0));

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
struct ObsSequentialLink {
    from_idx: usize,  // Observation index that is at position i
    to_idx: usize,    // Observation index that is at position i+1
    strength: f64,    // Carbon match quality (0-1)
}

// =============================================================================
// CACHED O(n²) STRUCTURES - Precomputed at max tolerance for fast iteration
// =============================================================================

/// Cached correlation pair - stores indices of observations that correlate at max tolerance.
/// Used to avoid O(n²) recomputation every iteration.
#[derive(Debug, Clone)]
struct CachedCorrelationPair {
    i: usize,
    j: usize,
}

/// Cached sequential link data - stores backbone group indices and their carbon shift diffs.
/// Allows filtering by current tolerance without recomputing O(n²) backbone grouping.
#[derive(Debug, Clone)]
struct CachedSequentialData {
    from_group_idx: usize,
    to_group_idx: usize,
    // Carbon shift diffs (stored for filtering) - None means no data for that atom
    ca_diff: Option<f64>,  // |inter_CA_from - intra_CA_to|
    cb_diff: Option<f64>,  // |inter_CB_from - intra_CB_to|
    co_diff: Option<f64>,  // |inter_CO_from - intra_CO_to|
    // Observation indices to propagate links to
    from_obs_indices: Vec<usize>,
    to_obs_indices: Vec<usize>,
}

/// Cached NOESY link data - stores proton shift diff for fast filtering.
#[derive(Debug, Clone)]
struct CachedNoesyPair {
    backbone_idx: usize,
    carbon_idx: usize,
    h_diff: f64,  // |H_backbone - H_carbon|
}

// =============================================================================
// PRECOMPUTE AND FILTER FUNCTIONS - O(n²) once, O(cached) per iteration
// =============================================================================

/// Precompute correlation pairs at max tolerance (iteration 0).
/// Returns pairs that correlate at the widest tolerance - these are candidates for all iterations.
fn precompute_correlation_pairs(
    observations: &[Observation],
    tol_params: &NucleusToleranceParams,
) -> Vec<CachedCorrelationPair> {
    let n = observations.len();
    let mut pairs = Vec::new();

    // Use iteration=0 for max (widest) tolerance
    for i in 0..n {
        for j in (i + 1)..n {
            let score = compute_observation_pair_correlation(
                &observations[i], &observations[j],
                tol_params, 0, 100
            );
            if score > 0.01 {
                pairs.push(CachedCorrelationPair { i, j });
            }
        }
    }

    pairs
}

/// Filter correlations from cached pairs using current tolerance.
/// Much faster than O(n²) - only iterates over pre-computed candidate pairs.
fn filter_correlations_from_cache(
    cached_pairs: &[CachedCorrelationPair],
    observations: &[Observation],
    tol_params: &NucleusToleranceParams,
    iteration: usize,
    max_iterations: usize,
) -> Vec<Vec<f64>> {
    let n = observations.len();
    let mut correlations = vec![vec![0.0; n]; n];

    // Self-correlations
    for i in 0..n {
        correlations[i][i] = 1.0;
    }

    // Only compute for cached pairs
    for pair in cached_pairs {
        let score = compute_observation_pair_correlation(
            &observations[pair.i], &observations[pair.j],
            tol_params, iteration, max_iterations
        );
        correlations[pair.i][pair.j] = score;
        correlations[pair.j][pair.i] = score;
    }

    correlations
}

/// Precompute NOESY backbone-carbon pairs at max tolerance.
fn precompute_noesy_pairs(
    observations: &[Observation],
    tol_params: &NucleusToleranceParams,
) -> Vec<CachedNoesyPair> {
    use crate::data::spin_system::TransferPathway;

    let h_max_tol = tol_params.tolerance_for(NucleusType::H1, 0, 100);
    let mut pairs = Vec::new();

    // Helper to check if observation is backbone-type (has H/N)
    let is_backbone = |obs: &Observation| -> bool {
        obs.dimensions.iter().any(|d| d.nucleus == NucleusType::H1) &&
        obs.dimensions.iter().any(|d| d.nucleus == NucleusType::N15)
    };

    // Helper to check if observation is carbon-type with NOESY pathway
    let is_noesy_carbon = |obs: &Observation| -> bool {
        obs.transfer_pathway == TransferPathway::ThroughSpace &&
        obs.dimensions.iter().any(|d| d.nucleus == NucleusType::H1) &&
        obs.dimensions.iter().any(|d| d.nucleus == NucleusType::C13)
    };

    // Get H shift from observation
    let get_h = |obs: &Observation| -> Option<f64> {
        obs.dimensions.iter()
            .find(|d| d.nucleus == NucleusType::H1)
            .map(|d| d.shift)
    };

    // Find backbone and NOESY carbon observations
    let backbone_obs: Vec<(usize, f64)> = observations.iter()
        .enumerate()
        .filter(|(_, obs)| is_backbone(obs))
        .filter_map(|(idx, obs)| get_h(obs).map(|h| (idx, h)))
        .collect();

    let noesy_carbon_obs: Vec<(usize, f64)> = observations.iter()
        .enumerate()
        .filter(|(_, obs)| is_noesy_carbon(obs))
        .filter_map(|(idx, obs)| get_h(obs).map(|h| (idx, h)))
        .collect();

    // Find pairs that match at max tolerance
    for (bb_idx, bb_h) in &backbone_obs {
        for (c_idx, c_h) in &noesy_carbon_obs {
            let h_diff = (bb_h - c_h).abs();
            if h_diff < h_max_tol {
                pairs.push(CachedNoesyPair {
                    backbone_idx: *bb_idx,
                    carbon_idx: *c_idx,
                    h_diff,
                });
            }
        }
    }

    pairs
}

/// Filter NOESY links from cached pairs using current tolerance.
fn filter_noesy_from_cache(
    cached_pairs: &[CachedNoesyPair],
    tol_params: &NucleusToleranceParams,
    iteration: usize,
    max_iterations: usize,
) -> Vec<(usize, usize, f64)> {
    let h_tol = tol_params.tolerance_for(NucleusType::H1, iteration, max_iterations);
    let mut links = Vec::new();

    for pair in cached_pairs {
        if pair.h_diff < h_tol {
            // Quality based on Gaussian
            let quality = (-0.5 * (pair.h_diff / h_tol).powi(2)).exp();
            links.push((pair.backbone_idx, pair.carbon_idx, quality));
        }
    }

    links
}

/// Compute sequential links from backbone-sequential transfer pathway observations.
///
/// Uses Bayesian likelihood ratio scoring with multi-atom evidence aggregation:
/// - Groups observations by backbone (H, N) anchor
/// - Collects all intra/inter CA, CB, CO shifts per backbone group
/// - For each backbone pair, computes Bayesian LR: match = +2.94, conflict = -2.94, missing = 0
/// - Only creates links with positive log-LR (at least one match, no conflicts)
///
/// This dramatically reduces false positives: if CA matches but CB conflicts, the link is rejected.
fn compute_sequential_links(
    observations: &[Observation],
    tol_params: &NucleusToleranceParams,
    iteration: usize,
    max_iterations: usize,
) -> Vec<ObsSequentialLink> {
    use crate::data::spin_system::{TransferPathway, ResidueOffset};
    use std::collections::HashMap;

    let c_tol = 0.3;  // Carbon tolerance for matching
    let h_tol = tol_params.tolerance_for(NucleusType::H1, iteration, max_iterations);
    let n_tol = tol_params.tolerance_for(NucleusType::N15, iteration, max_iterations);

    // === Step 1: Group observations by backbone (H, N) anchor ===
    // Key: (H*1000, N*10) as integers for grouping
    #[derive(Debug)]
    struct BackboneGroup {
        h: f64,
        n: f64,
        obs_indices: Vec<usize>,  // Observation indices in this group
        intra_ca: Vec<f64>,
        inter_ca: Vec<f64>,
        intra_cb: Vec<f64>,
        inter_cb: Vec<f64>,
        intra_co: Vec<f64>,
        inter_co: Vec<f64>,
    }

    let mut groups: Vec<BackboneGroup> = Vec::new();

    // Helper to find or create backbone group
    let find_or_create_group = |groups: &mut Vec<BackboneGroup>, h: f64, n: f64, h_tol: f64, n_tol: f64| -> usize {
        for (idx, g) in groups.iter().enumerate() {
            if (g.h - h).abs() < h_tol && (g.n - n).abs() < n_tol {
                return idx;
            }
        }
        groups.push(BackboneGroup {
            h, n,
            obs_indices: Vec::new(),
            intra_ca: Vec::new(), inter_ca: Vec::new(),
            intra_cb: Vec::new(), inter_cb: Vec::new(),
            intra_co: Vec::new(), inter_co: Vec::new(),
        });
        groups.len() - 1
    };

    // Process all backbone-sequential observations
    for (obs_idx, obs) in observations.iter().enumerate() {
        if obs.transfer_pathway != TransferPathway::BackboneSequential {
            continue;
        }

        // Get backbone anchor
        let h = obs.dimensions.iter().find(|d| d.nucleus == NucleusType::H1).map(|d| d.shift);
        let n = obs.dimensions.iter().find(|d| d.nucleus == NucleusType::N15).map(|d| d.shift);
        let (Some(h), Some(n)) = (h, n) else { continue };

        // Get carbon info
        let c_dim = obs.dimensions.iter().find(|d| d.nucleus == NucleusType::C13);
        let Some(c_dim) = c_dim else { continue };

        let c_shift = c_dim.shift;
        let is_intra = c_dim.residue_offset == ResidueOffset::Intra;

        // Determine atom type from hint or shift range
        let atom_type = c_dim.atom_hint.clone().unwrap_or_else(|| {
            if c_shift > 165.0 && c_shift < 185.0 {
                "C".to_string()  // CO
            } else if c_shift > 44.0 && c_shift < 75.0 {
                "CA".to_string()
            } else {
                "CB".to_string()
            }
        });

        // Find or create group
        let group_idx = find_or_create_group(&mut groups, h, n, h_tol, n_tol);
        let group = &mut groups[group_idx];
        group.obs_indices.push(obs_idx);

        // Add carbon to appropriate list
        match (atom_type.as_str(), is_intra) {
            ("CA", true) => group.intra_ca.push(c_shift),
            ("CA", false) => group.inter_ca.push(c_shift),
            ("CB", true) => group.intra_cb.push(c_shift),
            ("CB", false) => group.inter_cb.push(c_shift),
            ("C", true) => group.intra_co.push(c_shift),
            ("C", false) => group.inter_co.push(c_shift),
            _ => {}
        }
    }

    // === Step 2: Build sequential links between backbone groups with Bayesian LR ===
    let mut links = Vec::new();
    let log_lr_match = (0.95_f64 / 0.05).ln();      // +2.94
    let log_lr_mismatch = (0.05_f64 / 0.95).ln();  // -2.94

    // Helper to find best match between two shift lists
    let find_best_diff = |intra: &[f64], inter: &[f64]| -> Option<f64> {
        if intra.is_empty() || inter.is_empty() {
            return None;  // Missing data
        }
        let mut best = f64::MAX;
        for &a in intra {
            for &b in inter {
                best = best.min((a - b).abs());
            }
        }
        Some(best)
    };

    // For each pair of backbone groups
    for (from_idx, from_group) in groups.iter().enumerate() {
        for (to_idx, to_group) in groups.iter().enumerate() {
            if from_idx == to_idx { continue; }

            // Check: from_group's INTRA matches to_group's INTER
            // This means from_group precedes to_group in sequence

            let ca_diff = find_best_diff(&from_group.intra_ca, &to_group.inter_ca);
            let cb_diff = find_best_diff(&from_group.intra_cb, &to_group.inter_cb);
            let co_diff = find_best_diff(&from_group.intra_co, &to_group.inter_co);

            // Skip if no atoms to compare
            if ca_diff.is_none() && cb_diff.is_none() && co_diff.is_none() {
                continue;
            }

            // Compute Bayesian log-LR
            let mut log_lr = 0.0;
            let mut matched_atoms = Vec::new();
            let mut conflicting_atoms = Vec::new();

            if let Some(diff) = ca_diff {
                if diff < c_tol {
                    log_lr += log_lr_match;
                    matched_atoms.push("CA");
                } else {
                    log_lr += log_lr_mismatch;
                    conflicting_atoms.push("CA");
                }
            }

            if let Some(diff) = cb_diff {
                if diff < c_tol {
                    log_lr += log_lr_match;
                    matched_atoms.push("CB");
                } else {
                    log_lr += log_lr_mismatch;
                    conflicting_atoms.push("CB");
                }
            }

            if let Some(diff) = co_diff {
                if diff < c_tol {
                    log_lr += log_lr_match;
                    matched_atoms.push("C");
                } else {
                    log_lr += log_lr_mismatch;
                    conflicting_atoms.push("C");
                }
            }

            // Only create link if log-LR > 0 (more evidence FOR than AGAINST)
            if log_lr > 0.0 && !matched_atoms.is_empty() {
                // Create link from any observation in from_group to any in to_group
                // Use first observation from each group as representative
                if let (Some(&from_obs), Some(&to_obs)) = (from_group.obs_indices.first(), to_group.obs_indices.first()) {
                    // Strength based on log-LR, normalized
                    let strength = (log_lr / 8.83).min(1.0);  // Max log-LR for 3 matches ≈ 8.83

                    links.push(ObsSequentialLink {
                        from_idx: from_obs,
                        to_idx: to_obs,
                        strength,
                    });
                }
            }
        }
    }

    // Debug output
    println!("Backbone groups: {}, Sequential links (Bayesian LR): {}", groups.len(), links.len());

    links
}

/// Compute NOESY backbone-carbon correlations (Factor 3).
///
/// Finds observations where:
/// - One is a backbone observation (has H/N)
/// - One is a carbon observation (has H/C) with ThroughSpace pathway
/// - Their H shifts match (NOE correlation)
///
/// Returns: (backbone_idx, carbon_idx, quality)
fn compute_noesy_backbone_carbon(
    observations: &[Observation],
    tol_params: &NucleusToleranceParams,
    iteration: usize,
    max_iterations: usize,
) -> Vec<(usize, usize, f64)> {
    use crate::data::spin_system::TransferPathway;

    let h_tol = tol_params.tolerance_for(NucleusType::H1, iteration, max_iterations);
    let mut links = Vec::new();

    // Helper to check if observation is backbone-type (has H/N)
    let is_backbone = |obs: &Observation| -> bool {
        obs.dimensions.iter().any(|d| d.nucleus == NucleusType::H1) &&
        obs.dimensions.iter().any(|d| d.nucleus == NucleusType::N15)
    };

    // Helper to check if observation is carbon-type with NOESY pathway (has H/C and ThroughSpace)
    let is_noesy_carbon = |obs: &Observation| -> bool {
        obs.transfer_pathway == TransferPathway::ThroughSpace &&
        obs.dimensions.iter().any(|d| d.nucleus == NucleusType::H1) &&
        obs.dimensions.iter().any(|d| d.nucleus == NucleusType::C13)
    };

    // Get H shift from observation
    let get_h = |obs: &Observation| -> Option<f64> {
        obs.dimensions.iter()
            .find(|d| d.nucleus == NucleusType::H1)
            .map(|d| d.shift)
    };

    // Find backbone observations
    let backbone_obs: Vec<(usize, f64)> = observations.iter()
        .enumerate()
        .filter(|(_, obs)| is_backbone(obs))
        .filter_map(|(idx, obs)| get_h(obs).map(|h| (idx, h)))
        .collect();

    // Find NOESY carbon observations
    let noesy_carbon_obs: Vec<(usize, f64)> = observations.iter()
        .enumerate()
        .filter(|(_, obs)| is_noesy_carbon(obs))
        .filter_map(|(idx, obs)| get_h(obs).map(|h| (idx, h)))
        .collect();

    // Find correlations based on H shift matching
    for (bb_idx, bb_h) in &backbone_obs {
        for (c_idx, c_h) in &noesy_carbon_obs {
            let h_diff = (bb_h - c_h).abs();
            if h_diff < h_tol {
                // Quality based on Gaussian: closer match = higher quality
                let quality = (-0.5 * (h_diff / h_tol).powi(2)).exp();
                links.push((*bb_idx, *c_idx, quality));
            }
        }
    }

    links
}

/// Compute type confidence from per-position typing scores.
///
/// Aggregates scores by amino acid type (not position) and returns the best type
/// along with its confidence (proportion of total score mass).
///
/// Example: If typing_scores favor positions [3, 7, 12] which are all Glycine,
/// this returns ("GLY", high_confidence).
fn compute_type_confidence(
    typing_scores: &[f64],      // scores per position (domain_size elements, index 0 = unassigned)
    residue_types: &[String],   // amino acid type at each position (sequence.len() elements)
) -> Option<(String, f64)> {
    // Sum scores by amino acid type (not position)
    let mut type_scores: HashMap<String, f64> = HashMap::new();
    for (pos_idx, &score) in typing_scores.iter().enumerate().skip(1) {  // skip index 0 (unassigned)
        if let Some(aa) = residue_types.get(pos_idx - 1) {  // positions are 1-indexed
            *type_scores.entry(aa.clone()).or_default() += score;
        }
    }

    if type_scores.is_empty() {
        return None;
    }

    // Find best type and compute confidence
    let total: f64 = type_scores.values().sum();
    type_scores.into_iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(t, s)| (t, s / total.max(1e-10)))
}

/// Update beliefs using typing, correlation, sequential, NOESY, and sequence-type factors.
///
/// Factor 3 (NOESY sequential): If backbone B correlates with carbon C via NOESY, and C
/// has high belief at position R, then B gets boosted for position R+1 (B follows C).
///
/// Factor 5 (sequence-type constraint): If an observation is confidently typed as amino acid X,
/// penalize positions where X doesn't appear in the sequence. This creates "anchor points"
/// for rare residues like Glycine, Tryptophan, etc.
fn update_observation_beliefs_with_sequential(
    beliefs: &[Vec<f64>],
    typing_scores: &[Vec<f64>],
    correlation_scores: &[Vec<f64>],
    sequential_links: &[ObsSequentialLink],
    noesy_links: &[(usize, usize, f64)],  // (backbone_idx, carbon_idx, quality)
    type_to_positions: &HashMap<String, Vec<usize>>,
    residue_types: &[String],
    domain_size: usize,
    correlation_weight: f64,
    typing_weight: f64,
    sequential_weight: f64,
    sequence_type_weight: f64,
    sequence_type_threshold: f64,
) -> Vec<Vec<f64>> {
    let n = beliefs.len();
    let mut new_beliefs = vec![vec![0.0; domain_size]; n];

    for i in 0..n {
        // Start with typing prior
        for d in 0..domain_size {
            new_beliefs[i][d] = typing_scores[i][d].max(1e-10).ln() * typing_weight;
        }

        // Add correlation messages (same-residue factors)
        // DISABLED temporarily - correlation messages with uniform beliefs don't help and add noise
        // TODO: Re-enable once belief propagation is working correctly
        // for j in 0..n {
        //     if i == j { continue; }
        //
        //     let corr = correlation_scores[i][j];
        //     if corr > 0.01 {
        //         for d in 0..domain_size {
        //             new_beliefs[i][d] += beliefs[j][d].max(1e-10).ln() * corr * correlation_weight;
        //         }
        //     }
        // }

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
                // SEQUENTIAL: true backbone ordering
                // Logic: If link from A to B, and A has high belief at position d,
                // then B should be at d+1.
                //
                // Use RELATIVE boost: add (belief[d] - uniform) to position d+1
                // This way, if all beliefs are uniform, contribution is 0 (neutral)
                // If belief at d is HIGH, d+1 gets boosted
                // If belief at d is LOW, d+1 gets penalized
                let uniform_prob = 1.0 / domain_size as f64;

                if link.to_idx == i {
                    // I am the TO node: if FROM has high belief at d, I should be at d+1
                    for d in 1..domain_size - 1 {
                        let from_belief = beliefs[link.from_idx][d];
                        // Relative boost: how much above/below uniform
                        let relative = (from_belief / uniform_prob).max(1e-10).ln();
                        new_beliefs[i][d + 1] += relative * link.strength * sequential_weight;
                    }
                    // Position 0 and 1 get penalty for not fitting sequential chain
                    new_beliefs[i][0] -= link.strength * sequential_weight * 0.5;
                    new_beliefs[i][1] -= link.strength * sequential_weight * 0.5;
                }
                if link.from_idx == i {
                    // I am the FROM node: if TO has high belief at d, I should be at d-1
                    for d in 2..domain_size {
                        let to_belief = beliefs[link.to_idx][d];
                        let relative = (to_belief / uniform_prob).max(1e-10).ln();
                        new_beliefs[i][d - 1] += relative * link.strength * sequential_weight;
                    }
                    // Position 0 gets penalty for not fitting sequential chain
                    // Position domain_size-1 also doesn't fit (no next)
                    new_beliefs[i][0] -= link.strength * sequential_weight * 0.5;
                }
            }
        }

        // Factor 3: NOESY sequential
        // If backbone B correlates with carbon C via NOESY, and C has high belief at position R,
        // then B gets boosted for position R+1 (B is sequential to the residue containing C)
        // Logic: NOESY correlates protons within ~5Å. If backbone H correlates with a carbon's
        // attached H, and that carbon is intra-residue, then backbone is likely +1 position.
        for (bb_idx, c_idx, quality) in noesy_links {
            if *bb_idx == i {
                // This backbone correlates with carbon at c_idx via NOESY
                // If carbon has high belief at position R, boost backbone at R+1
                let uniform_prob = 1.0 / domain_size as f64;

                for r in 1..domain_size - 1 {
                    let carbon_prob = beliefs[*c_idx][r];
                    // Only boost if carbon has above-uniform belief (avoid noise)
                    if carbon_prob > uniform_prob * 3.0 {
                        let boost = (carbon_prob / uniform_prob).ln() * quality * sequential_weight;
                        new_beliefs[i][r + 1] += boost;
                    }
                }
            }
        }

        // Factor 5: Sequence-type constraint
        // If we're confident about the amino acid type, constrain to valid positions
        // This creates "anchor points" for rare residues (Gly, Trp, His, etc.)
        if sequence_type_weight > 0.0 {
            if let Some((best_type, confidence)) = compute_type_confidence(&typing_scores[i], residue_types) {
                if confidence > sequence_type_threshold {
                    if let Some(valid_positions) = type_to_positions.get(&best_type) {
                        // Penalize positions that don't match the typed amino acid
                        for d in 1..domain_size {
                            if !valid_positions.contains(&d) {
                                // Strong penalty for invalid positions
                                new_beliefs[i][d] -= sequence_type_weight * confidence;
                            }
                        }
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

/// Apply backbone uniqueness constraint: each residue gets at most one backbone NH.
///
/// This is a HARD CONSTRAINT that implements exclusion between backbone peaks.
/// For each residue position, the backbone peak with highest belief "wins" and
/// other backbone peaks have their belief for that residue set to near-zero.
///
/// This prevents multiple backbone peaks from being assigned to the same residue.
fn apply_backbone_uniqueness_factor(
    beliefs: &mut [Vec<f64>],
    backbone_indices: &[usize],
    domain_size: usize,
) {
    if backbone_indices.len() <= 1 {
        return;  // No competition with 0 or 1 backbone peak
    }

    // For each residue position (skip index 0 = unassigned)
    for residue in 1..domain_size {
        // Find which backbone peak has highest belief for this residue
        let mut best_backbone_idx: Option<usize> = None;
        let mut best_belief = 0.0;

        for &bb_idx in backbone_indices {
            let belief = beliefs[bb_idx][residue];
            if belief > best_belief {
                best_belief = belief;
                best_backbone_idx = Some(bb_idx);
            }
        }

        // Penalize all OTHER backbone peaks for this residue
        // Use strong penalty - this is a physical constraint
        if let Some(winner_idx) = best_backbone_idx {
            for &bb_idx in backbone_indices {
                if bb_idx != winner_idx {
                    // Suppress belief for this residue in losing backbone peaks
                    // Multiply by small factor to strongly discourage same-residue assignment
                    beliefs[bb_idx][residue] *= 0.01;
                }
            }
        }
    }

    // Re-normalize beliefs for backbone peaks after applying exclusion
    for &bb_idx in backbone_indices {
        let sum: f64 = beliefs[bb_idx].iter().sum();
        if sum > 0.0 {
            for v in &mut beliefs[bb_idx] {
                *v /= sum;
            }
        }
    }
}

/// Apply backbone GROUPING factor: observations in the same backbone group should have same belief.
///
/// This synchronizes beliefs among HNCA/HNCACB/HNCACO/HNCO observations from the same (H, N).
/// Within a group, we average beliefs and assign this average to all members.
fn apply_backbone_grouping_factor(
    beliefs: &mut [Vec<f64>],
    backbone_groups: &[Vec<usize>],
    domain_size: usize,
) {
    for group in backbone_groups {
        if group.len() <= 1 {
            continue;  // No grouping needed for single-member groups
        }

        // Compute average belief for this group using geometric mean (better for probabilities)
        let mut avg_beliefs = vec![1.0; domain_size];
        for d in 0..domain_size {
            let mut log_sum = 0.0;
            for &idx in group {
                log_sum += beliefs[idx][d].max(1e-20).ln();
            }
            avg_beliefs[d] = (log_sum / group.len() as f64).exp();
        }

        // Normalize average
        let sum: f64 = avg_beliefs.iter().sum();
        if sum > 0.0 {
            for v in &mut avg_beliefs {
                *v /= sum;
            }
        }

        // Blend each member's belief toward the group average (soft synchronization)
        // Use 0.7 weight toward average to allow individual evidence to still contribute
        let group_weight = 0.7;
        for &idx in group {
            for d in 0..domain_size {
                beliefs[idx][d] = (1.0 - group_weight) * beliefs[idx][d] + group_weight * avg_beliefs[d];
            }
        }
    }
}

/// Apply backbone GROUP uniqueness constraint: each residue gets at most one backbone GROUP.
///
/// This is a HARD CONSTRAINT that implements exclusion between backbone GROUPS (not observations).
/// For each residue position, the backbone group with highest average belief "wins" and
/// all observations in other groups have their belief for that residue set to near-zero.
fn apply_backbone_group_uniqueness_factor(
    beliefs: &mut [Vec<f64>],
    backbone_groups: &[Vec<usize>],
    domain_size: usize,
) {
    if backbone_groups.len() <= 1 {
        return;  // No competition with 0 or 1 backbone group
    }

    // Compute average belief for each group
    let group_beliefs: Vec<Vec<f64>> = backbone_groups.iter().map(|group| {
        if group.is_empty() {
            return vec![0.0; domain_size];
        }
        let mut avg = vec![0.0; domain_size];
        for d in 0..domain_size {
            avg[d] = group.iter().map(|&idx| beliefs[idx][d]).sum::<f64>() / group.len() as f64;
        }
        avg
    }).collect();

    // For each residue position (skip index 0 = unassigned)
    for residue in 1..domain_size {
        // Find which group has highest average belief for this residue
        let mut best_group_idx: Option<usize> = None;
        let mut best_belief = 0.0;

        for (group_idx, group_belief) in group_beliefs.iter().enumerate() {
            let belief = group_belief[residue];
            if belief > best_belief {
                best_belief = belief;
                best_group_idx = Some(group_idx);
            }
        }

        // Penalize all OTHER groups for this residue
        if let Some(winner_idx) = best_group_idx {
            for (group_idx, group) in backbone_groups.iter().enumerate() {
                if group_idx != winner_idx {
                    // Suppress belief for this residue in all observations of losing groups
                    for &obs_idx in group {
                        beliefs[obs_idx][residue] *= 0.01;
                    }
                }
            }
        }
    }

    // Re-normalize beliefs for all backbone observations after applying exclusion
    for group in backbone_groups {
        for &obs_idx in group {
            let sum: f64 = beliefs[obs_idx].iter().sum();
            if sum > 0.0 {
                for v in &mut beliefs[obs_idx] {
                    *v /= sum;
                }
            }
        }
    }
}

/// Apply SOFT backbone group uniqueness - gentler penalty during BP.
///
/// Uses softer suppression (0.3x instead of 0.01x) to allow recovery from early mistakes.
/// Full uniqueness is enforced at extraction time.
fn apply_soft_backbone_group_uniqueness(
    beliefs: &mut [Vec<f64>],
    backbone_groups: &[Vec<usize>],
    domain_size: usize,
) {
    if backbone_groups.len() <= 1 {
        return;
    }

    // Compute average belief for each group
    let group_beliefs: Vec<Vec<f64>> = backbone_groups.iter().map(|group| {
        if group.is_empty() {
            return vec![0.0; domain_size];
        }
        let mut avg = vec![0.0; domain_size];
        for d in 0..domain_size {
            avg[d] = group.iter().map(|&idx| beliefs[idx][d]).sum::<f64>() / group.len() as f64;
        }
        avg
    }).collect();

    // For each residue position (skip index 0 = unassigned)
    for residue in 1..domain_size {
        // Find which group has highest average belief for this residue
        let mut best_group_idx: Option<usize> = None;
        let mut best_belief = 0.0;

        for (group_idx, group_belief) in group_beliefs.iter().enumerate() {
            let belief = group_belief[residue];
            if belief > best_belief {
                best_belief = belief;
                best_group_idx = Some(group_idx);
            }
        }

        // Softly penalize other groups (0.3x instead of 0.01x)
        if let Some(winner_idx) = best_group_idx {
            for (group_idx, group) in backbone_groups.iter().enumerate() {
                if group_idx != winner_idx {
                    for &obs_idx in group {
                        beliefs[obs_idx][residue] *= 0.3;  // Soft penalty
                    }
                }
            }
        }
    }

    // Re-normalize
    for group in backbone_groups {
        for &obs_idx in group {
            let sum: f64 = beliefs[obs_idx].iter().sum();
            if sum > 0.0 {
                for v in &mut beliefs[obs_idx] {
                    *v /= sum;
                }
            }
        }
    }
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
    hncaco: &[UnlabeledPeak],
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
        hnco, hncaco, hnca, hncacb, cbcaconh, hbhaconh,
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
        let hncaco: Vec<UnlabeledPeak> = vec![];
        let hnca: Vec<UnlabeledPeak> = vec![];
        let hncacb: Vec<UnlabeledPeak> = vec![];
        let cbcaconh: Vec<UnlabeledPeak> = vec![];
        let hbhaconh: Vec<UnlabeledPeak> = vec![];

        let params = UnifiedAssignmentParams::default();
        let results = run_unified_assignment(
            &hsqc_15n, &hsqc_13c, &tocsy, &noesy,
            &hsqc_tocsy_15n, &hsqc_tocsy_13c,
            &hsqc_tocsy_15n_3d, &hsqc_tocsy_13c_3d,
            &hnco, &hncaco, &hnca, &hncacb, &cbcaconh, &hbhaconh,
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
