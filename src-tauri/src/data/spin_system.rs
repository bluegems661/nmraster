//! Spin system data structures for real NMR assignment.
//!
//! These structures support the real assignment workflow:
//! 1. Unlabeled peaks from experiments (no atom assignments)
//! 2. Spin systems grouped from TOCSY correlations
//! 3. Sequential links from NOESY contacts

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Type of NMR experiment for unlabeled peaks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PeakExperimentType {
    /// 15N-HSQC: Backbone N-H correlations
    Hsqc15N,
    /// 13C-HSQC: C-H correlations (CA-HA, CB-HB, etc.)
    Hsqc13C,
    /// TOCSY: H-H through-bond correlations within residue
    Tocsy,
    /// NOESY: H-H through-space correlations
    Noesy,
    /// 15N-HSQC-TOCSY 2D: (N, H_tocsy) - backbone NH anchored TOCSY
    HsqcTocsy15N,
    /// 15N-HSQC-TOCSY 3D: (N, H_backbone, H_tocsy)
    HsqcTocsy15N3D,
    /// 13C-HSQC-TOCSY 2D: (C, H_tocsy) - carbon anchored TOCSY
    HsqcTocsy13C,
    /// 13C-HSQC-TOCSY 3D: (C, H_anchor, H_tocsy)
    HsqcTocsy13C3D,

    // 3D Triple-resonance experiments for backbone assignment
    /// HNCO: (H, N, CO) - correlates NH(i) with CO(i-1)
    Hnco,
    /// HNCA: (H, N, CA) - correlates NH(i) with CA(i) strong, CA(i-1) weak
    Hnca,
    /// HNCACB: (H, N, CA/CB) - correlates NH(i) with CA/CB(i) positive, CA/CB(i-1) negative
    Hncacb,
    /// CBCACONH: (H, N, CA/CB) - correlates NH(i) with CA/CB(i-1) only
    Cbcaconh,
    /// HBHACONH: (H, N, HA/HB) - correlates NH(i) with HA/HB(i-1) only
    Hbhaconh,
}

// ============================================================================
// PHYSICS-BASED OBSERVATION MODEL
// ============================================================================
// These types encode the PHYSICS of NMR experiments rather than experiment names.
// Factor logic should reason about these properties, not experiment_type.

/// How magnetization was transferred between nuclei in an observation.
/// This encodes the PHYSICS of the NMR experiment, not the experiment name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransferPathway {
    /// One-bond J-coupling (e.g., HSQC). Nuclei are directly bonded.
    /// Implies: same atom pair, same residue.
    DirectBond,

    /// Multi-bond J-coupling (e.g., TOCSY). Nuclei are in the same spin system.
    /// Implies: same residue, any J-coupled atom pair within the spin system.
    ThroughBond,

    /// Dipolar coupling (e.g., NOESY). Nuclei are spatially close (<6Å).
    /// Implies: close in 3D space, may cross residue boundaries.
    ThroughSpace,

    /// Backbone J-pathway (triple-resonance experiments like HNCA, HNCACB).
    /// Specific backbone magnetization transfer: 1H → 15N → 13C.
    /// Some experiments show both intra (i) and inter (i-1) peaks.
    BackboneSequential,
}

/// Relationship of an observed nucleus to the "anchor" residue of the observation.
/// For backbone-anchored experiments, the anchor is defined by the H/N pair (residue i).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResidueOffset {
    /// This dimension observes the anchor residue (i).
    /// Examples: all HSQC dimensions, TOCSY (same spin system), strong HNCA CA.
    Intra,

    /// This dimension observes the preceding residue (i-1).
    /// Examples: weak HNCA CA, negative HNCACB CA/CB, all CBCACONH carbons.
    PrecedingResidue,

    /// This dimension observes the following residue (i+1).
    /// Rare in standard experiments, but could occur in certain pulse sequences.
    FollowingResidue,

    /// Residue relationship is ambiguous or unknown.
    /// Examples: NOESY (distance-based, may cross residues), unanchored TOCSY.
    Unknown,
}

/// Constraints on which atoms are physically possible for an observed dimension.
/// These are hard constraints derived from experiment physics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AtomConstraint {
    /// Must be exactly this atom (e.g., "H" for backbone amide, "N" for backbone nitrogen).
    Exact(String),

    /// Must be one of these atoms (e.g., ["CA", "CB"] for HNCACB carbon dimension).
    OneOf(Vec<String>),

    /// Any atom of this nucleus type (no constraint beyond nucleus).
    /// Used when atom identity must be inferred from chemical shift.
    Any,
}

impl AtomConstraint {
    /// Check if an atom name satisfies this constraint.
    pub fn allows(&self, atom: &str) -> bool {
        match self {
            AtomConstraint::Exact(a) => a == atom,
            AtomConstraint::OneOf(atoms) => atoms.iter().any(|a| a == atom),
            AtomConstraint::Any => true,
        }
    }

    /// Get allowed atom names, if constrained.
    pub fn allowed_atoms(&self) -> Option<Vec<&str>> {
        match self {
            AtomConstraint::Exact(a) => Some(vec![a.as_str()]),
            AtomConstraint::OneOf(atoms) => Some(atoms.iter().map(|s| s.as_str()).collect()),
            AtomConstraint::Any => None, // Caller should use full atom list for nucleus
        }
    }
}

/// An unlabeled peak from an NMR experiment.
///
/// This represents raw experimental data without atom assignments.
/// The assignment algorithm must determine which atom(s) produced each peak.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnlabeledPeak {
    pub id: Uuid,
    /// Peak positions in ppm for each dimension.
    /// - HSQC 15N: [N_ppm, H_ppm]
    /// - HSQC 13C: [C_ppm, H_ppm]
    /// - TOCSY: [H1_ppm, H2_ppm]
    /// - NOESY: [H1_ppm, H2_ppm]
    pub position_ppm: Vec<f64>,
    /// Peak intensity (arbitrary units, for relative comparison)
    pub intensity: f64,
    /// Type of experiment this peak came from
    pub experiment_type: PeakExperimentType,
    /// Optional linewidth in Hz (can help distinguish overlapped peaks)
    pub linewidth_hz: Option<Vec<f64>>,
}

impl UnlabeledPeak {
    /// Create a new unlabeled peak.
    pub fn new(position_ppm: Vec<f64>, intensity: f64, experiment_type: PeakExperimentType) -> Self {
        Self {
            id: Uuid::new_v4(),
            position_ppm,
            intensity,
            experiment_type,
            linewidth_hz: None,
        }
    }

    /// Create a 15N-HSQC peak (N, H).
    pub fn hsqc_15n(n_ppm: f64, h_ppm: f64, intensity: f64) -> Self {
        Self::new(vec![n_ppm, h_ppm], intensity, PeakExperimentType::Hsqc15N)
    }

    /// Create a 13C-HSQC peak (C, H).
    pub fn hsqc_13c(c_ppm: f64, h_ppm: f64, intensity: f64) -> Self {
        Self::new(vec![c_ppm, h_ppm], intensity, PeakExperimentType::Hsqc13C)
    }

    /// Create a TOCSY peak (H1, H2).
    pub fn tocsy(h1_ppm: f64, h2_ppm: f64, intensity: f64) -> Self {
        Self::new(vec![h1_ppm, h2_ppm], intensity, PeakExperimentType::Tocsy)
    }

    /// Create a NOESY peak (H1, H2).
    pub fn noesy(h1_ppm: f64, h2_ppm: f64, intensity: f64) -> Self {
        Self::new(vec![h1_ppm, h2_ppm], intensity, PeakExperimentType::Noesy)
    }

    /// Create a 15N-HSQC-TOCSY 2D peak (N, H_tocsy).
    /// Shows TOCSY correlations from backbone NH to aliphatic protons.
    pub fn hsqc_tocsy_15n(n_ppm: f64, h_tocsy_ppm: f64, intensity: f64) -> Self {
        Self::new(vec![n_ppm, h_tocsy_ppm], intensity, PeakExperimentType::HsqcTocsy15N)
    }

    /// Create a 15N-HSQC-TOCSY 3D peak (N, H_backbone, H_tocsy).
    /// Full 3D experiment with backbone H and TOCSY-correlated H.
    pub fn hsqc_tocsy_15n_3d(n_ppm: f64, h_bb_ppm: f64, h_tocsy_ppm: f64, intensity: f64) -> Self {
        Self::new(vec![n_ppm, h_bb_ppm, h_tocsy_ppm], intensity, PeakExperimentType::HsqcTocsy15N3D)
    }

    /// Create a 13C-HSQC-TOCSY 2D peak (C, H_tocsy).
    /// Shows TOCSY correlations from a carbon-attached proton to other protons.
    pub fn hsqc_tocsy_13c(c_ppm: f64, h_tocsy_ppm: f64, intensity: f64) -> Self {
        Self::new(vec![c_ppm, h_tocsy_ppm], intensity, PeakExperimentType::HsqcTocsy13C)
    }

    /// Create a 13C-HSQC-TOCSY 3D peak (C, H_anchor, H_tocsy).
    /// Full 3D experiment with anchor H and TOCSY-correlated H.
    pub fn hsqc_tocsy_13c_3d(c_ppm: f64, h_anchor_ppm: f64, h_tocsy_ppm: f64, intensity: f64) -> Self {
        Self::new(vec![c_ppm, h_anchor_ppm, h_tocsy_ppm], intensity, PeakExperimentType::HsqcTocsy13C3D)
    }

    // 3D Triple-resonance experiment constructors

    /// Create an HNCO peak (H, N, CO).
    /// Peak at NH(i) correlates with CO(i-1).
    pub fn hnco(h_ppm: f64, n_ppm: f64, co_ppm: f64, intensity: f64) -> Self {
        Self::new(vec![h_ppm, n_ppm, co_ppm], intensity, PeakExperimentType::Hnco)
    }

    /// Create an HNCA peak (H, N, CA).
    /// Peak at NH(i) correlates with CA.
    /// Strong intensity = intra-residue CA(i), weak (~30%) = inter-residue CA(i-1).
    pub fn hnca(h_ppm: f64, n_ppm: f64, ca_ppm: f64, intensity: f64) -> Self {
        Self::new(vec![h_ppm, n_ppm, ca_ppm], intensity, PeakExperimentType::Hnca)
    }

    /// Create an HNCACB peak (H, N, C).
    /// Peak at NH(i) correlates with CA or CB.
    /// Sign convention: positive intensity = intra-residue (i), negative = inter-residue (i-1).
    /// CA and CB are distinguished by chemical shift (~50-60 ppm for CA, ~20-70 ppm for CB).
    pub fn hncacb(h_ppm: f64, n_ppm: f64, c_ppm: f64, intensity: f64) -> Self {
        Self::new(vec![h_ppm, n_ppm, c_ppm], intensity, PeakExperimentType::Hncacb)
    }

    /// Create a CBCACONH peak (H, N, CA/CB).
    /// Peak at NH(i) shows CA(i-1) and CB(i-1) only (inter-residue).
    pub fn cbcaconh(h_ppm: f64, n_ppm: f64, c_ppm: f64, intensity: f64) -> Self {
        Self::new(vec![h_ppm, n_ppm, c_ppm], intensity, PeakExperimentType::Cbcaconh)
    }

    /// Create an HBHACONH peak (H, N, HA/HB).
    /// Peak at NH(i) shows HA(i-1) and HB(i-1) only (inter-residue protons).
    pub fn hbhaconh(h_ppm: f64, n_ppm: f64, h_ali_ppm: f64, intensity: f64) -> Self {
        Self::new(vec![h_ppm, n_ppm, h_ali_ppm], intensity, PeakExperimentType::Hbhaconh)
    }

    /// Get chemical shifts with nucleus labels based on experiment type.
    /// Returns [(nucleus_name, shift_ppm), ...]
    pub fn get_shifts_labeled(&self) -> Vec<(String, f64)> {
        match self.experiment_type {
            PeakExperimentType::Hsqc15N => {
                // [N, H]
                vec![
                    ("N".to_string(), self.position_ppm.get(0).copied().unwrap_or(0.0)),
                    ("H".to_string(), self.position_ppm.get(1).copied().unwrap_or(0.0)),
                ]
            }
            PeakExperimentType::Hsqc13C => {
                // [C, H]
                vec![
                    ("C".to_string(), self.position_ppm.get(0).copied().unwrap_or(0.0)),
                    ("H".to_string(), self.position_ppm.get(1).copied().unwrap_or(0.0)),
                ]
            }
            PeakExperimentType::Tocsy => {
                // [H1, H2]
                vec![
                    ("H1".to_string(), self.position_ppm.get(0).copied().unwrap_or(0.0)),
                    ("H2".to_string(), self.position_ppm.get(1).copied().unwrap_or(0.0)),
                ]
            }
            PeakExperimentType::Noesy => {
                // [H1, H2]
                vec![
                    ("H1".to_string(), self.position_ppm.get(0).copied().unwrap_or(0.0)),
                    ("H2".to_string(), self.position_ppm.get(1).copied().unwrap_or(0.0)),
                ]
            }
            PeakExperimentType::HsqcTocsy15N | PeakExperimentType::HsqcTocsy15N3D => {
                // [N, H_tocsy] or [N, H_bb, H_tocsy]
                if self.position_ppm.len() >= 3 {
                    vec![
                        ("N".to_string(), self.position_ppm[0]),
                        ("H".to_string(), self.position_ppm[1]),
                        ("H_tocsy".to_string(), self.position_ppm[2]),
                    ]
                } else {
                    vec![
                        ("N".to_string(), self.position_ppm.get(0).copied().unwrap_or(0.0)),
                        ("H_tocsy".to_string(), self.position_ppm.get(1).copied().unwrap_or(0.0)),
                    ]
                }
            }
            PeakExperimentType::HsqcTocsy13C | PeakExperimentType::HsqcTocsy13C3D => {
                // [C, H_tocsy] or [C, H_anchor, H_tocsy]
                if self.position_ppm.len() >= 3 {
                    vec![
                        ("C".to_string(), self.position_ppm[0]),
                        ("H".to_string(), self.position_ppm[1]),
                        ("H_tocsy".to_string(), self.position_ppm[2]),
                    ]
                } else {
                    vec![
                        ("C".to_string(), self.position_ppm.get(0).copied().unwrap_or(0.0)),
                        ("H_tocsy".to_string(), self.position_ppm.get(1).copied().unwrap_or(0.0)),
                    ]
                }
            }
            PeakExperimentType::Hnco => {
                // [H, N, CO]
                vec![
                    ("H".to_string(), self.position_ppm.get(0).copied().unwrap_or(0.0)),
                    ("N".to_string(), self.position_ppm.get(1).copied().unwrap_or(0.0)),
                    ("CO".to_string(), self.position_ppm.get(2).copied().unwrap_or(0.0)),
                ]
            }
            PeakExperimentType::Hnca => {
                // [H, N, CA]
                vec![
                    ("H".to_string(), self.position_ppm.get(0).copied().unwrap_or(0.0)),
                    ("N".to_string(), self.position_ppm.get(1).copied().unwrap_or(0.0)),
                    ("CA".to_string(), self.position_ppm.get(2).copied().unwrap_or(0.0)),
                ]
            }
            PeakExperimentType::Hncacb => {
                // [H, N, C] - C is CA or CB depending on shift range
                let c_shift = self.position_ppm.get(2).copied().unwrap_or(0.0);
                let c_name = if c_shift > 45.0 && c_shift < 70.0 { "CA" } else { "CB" };
                vec![
                    ("H".to_string(), self.position_ppm.get(0).copied().unwrap_or(0.0)),
                    ("N".to_string(), self.position_ppm.get(1).copied().unwrap_or(0.0)),
                    (c_name.to_string(), c_shift),
                ]
            }
            PeakExperimentType::Cbcaconh => {
                // [H, N, C] - C is CA or CB depending on shift range
                let c_shift = self.position_ppm.get(2).copied().unwrap_or(0.0);
                let c_name = if c_shift > 45.0 && c_shift < 70.0 { "CA" } else { "CB" };
                vec![
                    ("H".to_string(), self.position_ppm.get(0).copied().unwrap_or(0.0)),
                    ("N".to_string(), self.position_ppm.get(1).copied().unwrap_or(0.0)),
                    (c_name.to_string(), c_shift),
                ]
            }
            PeakExperimentType::Hbhaconh => {
                // [H, N, H_ali]
                vec![
                    ("H".to_string(), self.position_ppm.get(0).copied().unwrap_or(0.0)),
                    ("N".to_string(), self.position_ppm.get(1).copied().unwrap_or(0.0)),
                    ("H_ali".to_string(), self.position_ppm.get(2).copied().unwrap_or(0.0)),
                ]
            }
        }
    }
}

/// A candidate chemical shift with match score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShiftCandidate {
    /// Chemical shift value in ppm
    pub shift: f64,
    /// Match score (0.0 to 1.0) - how well this matches the spin system
    pub score: f64,
    /// Peak ID that this candidate comes from
    pub peak_id: Option<Uuid>,
}

impl ShiftCandidate {
    pub fn new(shift: f64, score: f64) -> Self {
        Self { shift, score, peak_id: None }
    }

    pub fn with_peak(shift: f64, score: f64, peak_id: Uuid) -> Self {
        Self { shift, score, peak_id: Some(peak_id) }
    }
}

/// A spin system representing all NMR signals from a single residue.
///
/// Built by:
/// 1. Using 15N-HSQC peak as anchor (defines H and N shifts)
/// 2. Finding TOCSY correlations from H to identify HA, HB, etc.
/// 3. Matching proton shifts to 13C-HSQC to get CA, CB, etc.
///
/// Uses **soft matching**: peaks can contribute to multiple spin systems
/// with different weights, allowing the factor graph to resolve ambiguity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpinSystem {
    pub id: Uuid,
    /// The 15N-HSQC peak that anchors this spin system
    pub anchor_peak_id: Uuid,
    /// Backbone amide proton shift from 15N-HSQC (ppm)
    pub h_shift: f64,
    /// Backbone nitrogen shift from 15N-HSQC (ppm)
    pub n_shift: f64,
    /// Whether this spin system has a visible backbone NH (false for N-terminal, proline)
    pub has_backbone_nh: bool,
    /// Proton chemical shifts keyed by atom name.
    /// Keys: "HA", "HB", "HB2", "HB3", "HG", etc.
    /// Values: chemical shift in ppm
    pub proton_shifts: HashMap<String, f64>,
    /// Carbon chemical shifts keyed by atom name (best match only).
    /// Keys: "CA", "CB", "CG", etc.
    /// Values: chemical shift in ppm
    pub carbon_shifts: HashMap<String, f64>,
    /// Carbon shift candidates with soft match scores.
    /// Multiple peaks can contribute to the same atom type with different scores.
    /// Keys: "CA", "CB", "CG", etc.
    /// Values: list of (shift, score) candidates
    pub carbon_candidates: HashMap<String, Vec<ShiftCandidate>>,
    /// Amino acid type scores from BMRB pattern matching.
    /// Keys: three-letter residue codes ("ALA", "VAL", etc.)
    /// Values: probability scores (0.0 to 1.0)
    pub amino_acid_scores: HashMap<String, f64>,
    /// Assigned residue sequence code (set after mapping to sequence)
    pub assigned_residue: Option<i32>,
    /// Confidence in the final assignment (0.0 to 1.0)
    pub assignment_confidence: f64,
}

impl SpinSystem {
    /// Create a new spin system from a 15N-HSQC anchor peak.
    pub fn new(anchor_peak: &UnlabeledPeak) -> Self {
        assert_eq!(
            anchor_peak.experiment_type,
            PeakExperimentType::Hsqc15N,
            "Spin system must be anchored by 15N-HSQC peak"
        );
        assert_eq!(
            anchor_peak.position_ppm.len(),
            2,
            "15N-HSQC peak must have 2 dimensions"
        );

        Self {
            id: Uuid::new_v4(),
            anchor_peak_id: anchor_peak.id,
            n_shift: anchor_peak.position_ppm[0],
            h_shift: anchor_peak.position_ppm[1],
            has_backbone_nh: anchor_peak.position_ppm[0] > 0.0, // N=0 means no backbone NH
            proton_shifts: HashMap::new(),
            carbon_shifts: HashMap::new(),
            carbon_candidates: HashMap::new(),
            amino_acid_scores: HashMap::new(),
            assigned_residue: None,
            assignment_confidence: 0.0,
        }
    }

    /// Add a proton shift to this spin system.
    pub fn add_proton_shift(&mut self, atom_name: &str, shift_ppm: f64) {
        self.proton_shifts.insert(atom_name.to_string(), shift_ppm);
    }

    /// Add a carbon shift to this spin system (hard assignment).
    pub fn add_carbon_shift(&mut self, atom_name: &str, shift_ppm: f64) {
        self.carbon_shifts.insert(atom_name.to_string(), shift_ppm);
    }

    /// Add a carbon candidate with soft matching score.
    /// Multiple candidates can contribute to the same atom type.
    pub fn add_carbon_candidate(&mut self, atom_name: &str, candidate: ShiftCandidate) {
        self.carbon_candidates
            .entry(atom_name.to_string())
            .or_insert_with(Vec::new)
            .push(candidate);
    }

    /// Get the best carbon candidate for an atom type (highest score).
    pub fn best_carbon_candidate(&self, atom_name: &str) -> Option<&ShiftCandidate> {
        self.carbon_candidates
            .get(atom_name)
            .and_then(|candidates| {
                candidates.iter().max_by(|a, b| a.score.partial_cmp(&b.score).unwrap())
            })
    }

    /// Get weighted average carbon shift for an atom type.
    /// Returns (weighted_shift, total_weight).
    pub fn weighted_carbon_shift(&self, atom_name: &str) -> Option<(f64, f64)> {
        self.carbon_candidates.get(atom_name).map(|candidates| {
            let total_weight: f64 = candidates.iter().map(|c| c.score).sum();
            if total_weight > 0.0 {
                let weighted_sum: f64 = candidates.iter().map(|c| c.shift * c.score).sum();
                (weighted_sum / total_weight, total_weight)
            } else {
                (0.0, 0.0)
            }
        })
    }

    /// Finalize carbon assignments by selecting best candidates.
    /// Call this after all soft assignments are made.
    pub fn finalize_carbon_assignments(&mut self) {
        for (atom_name, candidates) in &self.carbon_candidates {
            if let Some(best) = candidates.iter().max_by(|a, b| a.score.partial_cmp(&b.score).unwrap()) {
                self.carbon_shifts.insert(atom_name.clone(), best.shift);
            }
        }
    }

    /// Get the best amino acid type guess.
    pub fn best_amino_acid_type(&self) -> Option<(&str, f64)> {
        self.amino_acid_scores
            .iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(name, score)| (name.as_str(), *score))
    }

    /// Check if this spin system has CA/CB shifts (indicates good data).
    pub fn has_carbon_backbone(&self) -> bool {
        self.carbon_shifts.contains_key("CA")
    }

    /// Get the number of assigned atoms (protons + carbons + H + N).
    pub fn num_assigned_atoms(&self) -> usize {
        2 + self.proton_shifts.len() + self.carbon_shifts.len() // +2 for H and N
    }
}

/// Type of NOE connectivity pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NOEType {
    /// dαN: H(i) to HA(i-1) - most reliable for sequential assignment
    DAlphaN,
    /// dNN: H(i) to H(i-1) - present in alpha helices
    DNN,
    /// dβN: H(i) to HB(i-1) - additional sequential evidence
    DBetaN,
    /// dαN(i,i+1): HA(i) to H(i+1) - reverse direction
    DAlphaNReverse,
    /// Long-range or ambiguous NOE
    Other,
}

impl NOEType {
    /// Get the expected distance range for this NOE type (in Angstroms).
    pub fn expected_distance_range(&self) -> (f64, f64) {
        match self {
            NOEType::DAlphaN => (2.2, 3.5),          // Sequential dαN
            NOEType::DNN => (2.8, 4.5),              // Sequential dNN (helices)
            NOEType::DBetaN => (2.5, 4.0),           // Sequential dβN
            NOEType::DAlphaNReverse => (2.2, 3.5),   // Same as dαN
            NOEType::Other => (2.0, 6.0),            // Generic NOE range
        }
    }
}

/// A sequential link between two spin systems based on NOESY data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SequentialLink {
    /// ID of the spin system at position i
    pub from_spin_system: Uuid,
    /// ID of the spin system at position i+1
    pub to_spin_system: Uuid,
    /// Type of NOE evidence for this link
    pub noe_type: NOEType,
    /// Confidence in this sequential connection (0.0 to 1.0)
    pub confidence: f64,
    /// The NOESY peak(s) supporting this link
    pub supporting_peaks: Vec<Uuid>,
}

impl SequentialLink {
    /// Create a new sequential link.
    pub fn new(
        from_spin_system: Uuid,
        to_spin_system: Uuid,
        noe_type: NOEType,
        confidence: f64,
    ) -> Self {
        Self {
            from_spin_system,
            to_spin_system,
            noe_type,
            confidence,
            supporting_peaks: Vec::new(),
        }
    }

    /// Add a supporting NOESY peak to this link.
    pub fn add_supporting_peak(&mut self, peak_id: Uuid) {
        self.supporting_peaks.push(peak_id);
    }
}

/// Container for all spin systems and their connections.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SpinSystemList {
    pub id: Uuid,
    pub name: String,
    pub molecule_id: Uuid,
    /// All identified spin systems
    pub spin_systems: Vec<SpinSystem>,
    /// Sequential links between spin systems
    pub sequential_links: Vec<SequentialLink>,
}

impl SpinSystemList {
    /// Create a new spin system list.
    pub fn new(name: &str, molecule_id: Uuid) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.to_string(),
            molecule_id,
            spin_systems: Vec::new(),
            sequential_links: Vec::new(),
        }
    }

    /// Add a spin system.
    pub fn add_spin_system(&mut self, ss: SpinSystem) {
        self.spin_systems.push(ss);
    }

    /// Add a sequential link.
    pub fn add_sequential_link(&mut self, link: SequentialLink) {
        self.sequential_links.push(link);
    }

    /// Find a spin system by ID.
    pub fn get_spin_system(&self, id: &Uuid) -> Option<&SpinSystem> {
        self.spin_systems.iter().find(|ss| &ss.id == id)
    }

    /// Find a spin system by ID (mutable).
    pub fn get_spin_system_mut(&mut self, id: &Uuid) -> Option<&mut SpinSystem> {
        self.spin_systems.iter_mut().find(|ss| &ss.id == id)
    }

    /// Get all links originating from a spin system.
    pub fn get_links_from(&self, spin_system_id: &Uuid) -> Vec<&SequentialLink> {
        self.sequential_links
            .iter()
            .filter(|link| &link.from_spin_system == spin_system_id)
            .collect()
    }

    /// Get all links pointing to a spin system.
    pub fn get_links_to(&self, spin_system_id: &Uuid) -> Vec<&SequentialLink> {
        self.sequential_links
            .iter()
            .filter(|link| &link.to_spin_system == spin_system_id)
            .collect()
    }

    /// Get number of spin systems.
    pub fn num_spin_systems(&self) -> usize {
        self.spin_systems.len()
    }

    /// Get number of sequential links.
    pub fn num_links(&self) -> usize {
        self.sequential_links.len()
    }

    /// Get spin systems that have been assigned to residues.
    pub fn assigned_spin_systems(&self) -> Vec<&SpinSystem> {
        self.spin_systems
            .iter()
            .filter(|ss| ss.assigned_residue.is_some())
            .collect()
    }

    /// Get assignment coverage (fraction of spin systems assigned).
    pub fn assignment_coverage(&self) -> f64 {
        if self.spin_systems.is_empty() {
            return 0.0;
        }
        self.assigned_spin_systems().len() as f64 / self.spin_systems.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unlabeled_peak_creation() {
        let peak = UnlabeledPeak::hsqc_15n(120.5, 8.21, 1000.0);
        assert_eq!(peak.experiment_type, PeakExperimentType::Hsqc15N);
        assert_eq!(peak.position_ppm[0], 120.5);
        assert_eq!(peak.position_ppm[1], 8.21);
    }

    #[test]
    fn test_spin_system_creation() {
        let anchor = UnlabeledPeak::hsqc_15n(120.5, 8.21, 1000.0);
        let mut ss = SpinSystem::new(&anchor);

        assert_eq!(ss.n_shift, 120.5);
        assert_eq!(ss.h_shift, 8.21);

        ss.add_proton_shift("HA", 4.35);
        ss.add_proton_shift("HB", 1.42);
        ss.add_carbon_shift("CA", 53.1);
        ss.add_carbon_shift("CB", 19.0);

        assert_eq!(ss.num_assigned_atoms(), 6); // H, N, HA, HB, CA, CB
        assert!(ss.has_carbon_backbone());
    }

    #[test]
    fn test_sequential_link() {
        let ss1_id = Uuid::new_v4();
        let ss2_id = Uuid::new_v4();

        let mut link = SequentialLink::new(ss1_id, ss2_id, NOEType::DAlphaN, 0.95);
        link.add_supporting_peak(Uuid::new_v4());

        assert_eq!(link.noe_type, NOEType::DAlphaN);
        assert_eq!(link.supporting_peaks.len(), 1);
    }

    #[test]
    fn test_spin_system_list() {
        let mol_id = Uuid::new_v4();
        let mut list = SpinSystemList::new("test", mol_id);

        let anchor1 = UnlabeledPeak::hsqc_15n(120.5, 8.21, 1000.0);
        let anchor2 = UnlabeledPeak::hsqc_15n(119.8, 8.19, 900.0);

        let ss1 = SpinSystem::new(&anchor1);
        let ss2 = SpinSystem::new(&anchor2);
        let ss1_id = ss1.id;
        let ss2_id = ss2.id;

        list.add_spin_system(ss1);
        list.add_spin_system(ss2);

        let link = SequentialLink::new(ss1_id, ss2_id, NOEType::DAlphaN, 0.9);
        list.add_sequential_link(link);

        assert_eq!(list.num_spin_systems(), 2);
        assert_eq!(list.num_links(), 1);
        assert_eq!(list.get_links_from(&ss1_id).len(), 1);
        assert_eq!(list.get_links_to(&ss2_id).len(), 1);
    }

    #[test]
    fn test_amino_acid_scoring() {
        let anchor = UnlabeledPeak::hsqc_15n(120.5, 8.21, 1000.0);
        let mut ss = SpinSystem::new(&anchor);

        ss.amino_acid_scores.insert("ALA".to_string(), 0.92);
        ss.amino_acid_scores.insert("VAL".to_string(), 0.45);
        ss.amino_acid_scores.insert("GLY".to_string(), 0.12);

        let (best, score) = ss.best_amino_acid_type().unwrap();
        assert_eq!(best, "ALA");
        assert!((score - 0.92).abs() < 0.001);
    }
}
