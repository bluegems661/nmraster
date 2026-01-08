//! Amino acid typing from spin system chemical shifts.
//!
//! Scores spin systems against BMRB statistics to determine the most likely
//! amino acid type. Carbon shifts (CA, CB) are particularly discriminating.
//!
//! This module supports two scoring methods:
//! - Gaussian scoring (traditional): Uses mean/std_dev from BMRB statistics
//! - KDE scoring (recommended): Uses full empirical distributions from BMRB

use std::collections::HashMap;

use crate::data::SpinSystem;
use crate::inference::scoring::{GaussianScorer, KDEScorer, ShiftScorer};
use crate::testdata::BMRBDatabase;

/// Parameters for amino acid typing.
#[derive(Debug, Clone)]
pub struct TypingParams {
    /// Weight for backbone H shift match (least discriminating)
    pub h_weight: f64,
    /// Weight for backbone N shift match
    pub n_weight: f64,
    /// Weight for carbon shifts (CA, CB are most discriminating)
    pub carbon_weight: f64,
    /// Weight for sidechain proton shifts
    pub proton_weight: f64,
    /// Penalty per missing expected atom
    pub missing_atom_penalty: f64,
    /// Penalty per extra observed atom (not expected for residue)
    pub extra_atom_penalty: f64,
    /// Minimum score to consider a type valid (0.0-1.0)
    pub min_score_threshold: f64,
}

impl Default for TypingParams {
    fn default() -> Self {
        Self {
            h_weight: 0.5,       // Backbone H is least discriminating
            n_weight: 0.8,       // Backbone N is somewhat useful
            carbon_weight: 3.0,  // Carbons are highly discriminating
            proton_weight: 1.5,  // Sidechain protons help
            missing_atom_penalty: -1.0,  // Penalty for missing expected atoms
            extra_atom_penalty: -2.0,    // Stronger penalty for extra atoms
            min_score_threshold: 0.01,   // Minimum score to include
        }
    }
}

/// Expected atoms for each amino acid type.
/// Returns (carbons, protons) that should be observable in 13C-HSQC.
/// Only includes carbons with directly attached protons (not quaternary or carbonyl).
fn get_expected_atoms(residue: &str) -> (Vec<&'static str>, Vec<&'static str>) {
    match residue {
        "GLY" => (vec!["CA"], vec!["HA2", "HA3"]),
        "ALA" => (vec!["CA", "CB"], vec!["HA", "HB"]),
        "VAL" => (vec!["CA", "CB", "CG1", "CG2"], vec!["HA", "HB", "HG1", "HG2"]),
        "LEU" => (vec!["CA", "CB", "CG", "CD1", "CD2"], vec!["HA", "HB", "HG", "HD1", "HD2"]),
        "ILE" => (vec!["CA", "CB", "CG1", "CG2", "CD1"], vec!["HA", "HB", "HG1", "HG2", "HD1"]),
        "PRO" => (vec!["CA", "CB", "CG", "CD"], vec!["HA", "HB", "HG", "HD"]),
        "MET" => (vec!["CA", "CB", "CG", "CE"], vec!["HA", "HB", "HG", "HE"]),
        // Aromatic residues - only carbons with attached H
        "PHE" => (vec!["CA", "CB", "CD1", "CD2", "CE1", "CE2", "CZ"], vec!["HA", "HB", "HD1", "HD2", "HE1", "HE2", "HZ"]),
        "TYR" => (vec!["CA", "CB", "CD1", "CD2", "CE1", "CE2"], vec!["HA", "HB", "HD1", "HD2", "HE1", "HE2"]),
        "TRP" => (vec!["CA", "CB", "CD1", "CE3", "CZ2", "CZ3", "CH2"], vec!["HA", "HB", "HD1", "HE3", "HZ2", "HZ3", "HH2"]),
        "HIS" => (vec!["CA", "CB", "CD2", "CE1"], vec!["HA", "HB", "HD2", "HE1"]),
        "SER" => (vec!["CA", "CB"], vec!["HA", "HB"]),
        "THR" => (vec!["CA", "CB", "CG2"], vec!["HA", "HB", "HG2"]),
        "CYS" => (vec!["CA", "CB"], vec!["HA", "HB"]),
        "ASN" => (vec!["CA", "CB"], vec!["HA", "HB"]),  // CG is carbonyl
        "GLN" => (vec!["CA", "CB", "CG"], vec!["HA", "HB", "HG"]),  // CD is carbonyl
        "ASP" => (vec!["CA", "CB"], vec!["HA", "HB"]),  // CG is carbonyl
        "GLU" => (vec!["CA", "CB", "CG"], vec!["HA", "HB", "HG"]),  // CD is carbonyl
        "LYS" => (vec!["CA", "CB", "CG", "CD", "CE"], vec!["HA", "HB", "HG", "HD", "HE"]),
        "ARG" => (vec!["CA", "CB", "CG", "CD"], vec!["HA", "HB", "HG", "HD"]),  // CZ is guanidinium
        _ => (vec!["CA", "CB"], vec!["HA", "HB"]),
    }
}

/// Score a spin system against all amino acid types using Gaussian scoring.
///
/// Returns a HashMap of residue three-letter codes to probability scores.
/// Scores are normalized to sum to 1.0.
///
/// For more accurate scoring that captures multi-modal distributions,
/// use `score_amino_acid_types_kde` instead.
pub fn score_amino_acid_types(
    spin_system: &SpinSystem,
    bmrb: &BMRBDatabase,
    params: &TypingParams,
) -> HashMap<String, f64> {
    let residue_names = bmrb.residue_names();
    let mut raw_scores = HashMap::new();

    for residue in &residue_names {
        let score = score_single_residue_type(spin_system, residue, bmrb, params);
        if score > params.min_score_threshold {
            raw_scores.insert(residue.clone(), score);
        }
    }

    // Normalize scores to sum to 1.0
    normalize_scores(&mut raw_scores);

    raw_scores
}

// =============================================================================
// Scorer-based typing functions (supports KDE and Gaussian)
// =============================================================================

/// Standard list of amino acid residue names.
const RESIDUE_NAMES: [&str; 20] = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
    "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
    "THR", "TRP", "TYR", "VAL",
];

/// Score a spin system against all amino acid types using a ShiftScorer.
///
/// This is the generic version that accepts any scorer (Gaussian or KDE).
/// Scores are normalized to sum to 1.0.
pub fn score_amino_acid_types_with_scorer(
    spin_system: &SpinSystem,
    scorer: &dyn ShiftScorer,
    params: &TypingParams,
) -> HashMap<String, f64> {
    let mut raw_scores = HashMap::new();

    for residue in RESIDUE_NAMES {
        let score = score_single_residue_with_scorer(spin_system, residue, scorer, params);
        if score > params.min_score_threshold {
            raw_scores.insert(residue.to_string(), score);
        }
    }

    normalize_scores(&mut raw_scores);
    raw_scores
}

/// Score a spin system using KDE-based empirical distributions.
///
/// This is the recommended method as it captures multi-modal distributions
/// (e.g., CYS CB with/without disulfide bonds).
pub fn score_amino_acid_types_kde(
    spin_system: &SpinSystem,
    params: &TypingParams,
) -> HashMap<String, f64> {
    let scorer = KDEScorer::new();
    score_amino_acid_types_with_scorer(spin_system, &scorer, params)
}

/// Score a spin system using Gaussian approximations.
///
/// This is the traditional method using mean/std_dev from BMRB.
/// Faster but less accurate for multi-modal distributions.
pub fn score_amino_acid_types_gaussian(
    spin_system: &SpinSystem,
    params: &TypingParams,
) -> HashMap<String, f64> {
    let scorer = GaussianScorer::new();
    score_amino_acid_types_with_scorer(spin_system, &scorer, params)
}

/// Minimum probability to avoid log(0).
const MIN_PROB: f64 = 1e-10;

/// Score a single residue type using a ShiftScorer.
fn score_single_residue_with_scorer(
    spin_system: &SpinSystem,
    residue: &str,
    scorer: &dyn ShiftScorer,
    params: &TypingParams,
) -> f64 {
    let mut total_log_prob = 0.0;
    let mut total_weight = 0.0;

    // Score backbone H (always available)
    let h_score = scorer.score(residue, "H", spin_system.h_shift).max(MIN_PROB);
    total_log_prob += params.h_weight * h_score.ln();
    total_weight += params.h_weight;

    // Score backbone N (always available)
    let n_score = scorer.score(residue, "N", spin_system.n_shift).max(MIN_PROB);
    total_log_prob += params.n_weight * n_score.ln();
    total_weight += params.n_weight;

    // Get expected atoms for this residue type
    let (expected_carbons, expected_protons) = get_expected_atoms(residue);

    // Score ALL observed carbon shifts against this residue's expected atoms
    let mut observed_carbons: Vec<&String> = spin_system.carbon_shifts.keys().collect();
    for (atom, candidates) in &spin_system.carbon_candidates {
        if !candidates.is_empty() && !observed_carbons.contains(&atom) {
            observed_carbons.push(atom);
        }
    }

    for carbon_atom in &observed_carbons {
        // Get the observed shift for this carbon
        let observed_shift = spin_system.carbon_shifts.get(*carbon_atom)
            .or_else(|| spin_system.carbon_candidates.get(*carbon_atom)
                .and_then(|c| c.first().map(|x| &x.shift)));

        if let Some(&shift) = observed_shift {
            // Check if this carbon type is expected for this residue
            let is_expected = expected_carbons.iter()
                .any(|&e| normalize_atom_name(carbon_atom) == normalize_atom_name(e));

            if is_expected {
                // Score this carbon directly against the expected atom
                let prob = scorer.score(residue, carbon_atom, shift).max(MIN_PROB);
                total_log_prob += params.carbon_weight * prob.ln();
                total_weight += params.carbon_weight;
            } else {
                // This carbon is NOT expected - find best matching expected atom
                // and score the observed shift against it
                let mut best_prob = MIN_PROB;
                for &expected_atom in &expected_carbons {
                    let prob = scorer.score(residue, expected_atom, shift);
                    if prob > best_prob {
                        best_prob = prob;
                    }
                }
                // Penalize unexpected carbons (they shouldn't be there for this residue)
                total_log_prob += params.carbon_weight * best_prob.max(MIN_PROB).ln();
                total_log_prob += params.extra_atom_penalty; // Extra penalty for unexpected carbon
                total_weight += params.carbon_weight;
            }
        }
    }

    // Penalize if fewer carbons observed than expected (missing evidence)
    let observed_c_count = observed_carbons.len();
    let expected_c_count = expected_carbons.len();
    if observed_c_count < expected_c_count {
        // Missing carbons - small penalty (might just be missing data)
        let missing = expected_c_count - observed_c_count;
        total_log_prob += params.missing_atom_penalty * missing as f64;
        total_weight += missing as f64 * 0.5; // Lower weight for missing atoms
    }

    // Score ALL observed proton shifts
    for (proton_atom, &shift) in &spin_system.proton_shifts {
        // Map HA to HA2 for GLY
        let atom_name = if residue == "GLY" && proton_atom == "HA" {
            "HA2"
        } else {
            proton_atom.as_str()
        };

        let proton_score = scorer.score(residue, atom_name, shift).max(MIN_PROB);
        total_log_prob += params.proton_weight * proton_score.ln();
        total_weight += params.proton_weight;
    }

    // Convert log probability back to probability
    if total_weight > 0.0 {
        (total_log_prob / total_weight).exp()
    } else {
        1.0 / 20.0 // Uniform prior if no data
    }
}

/// Normalize atom names for comparison (CG1 -> CG, HD1 -> HD, etc.)
fn normalize_atom_name(atom: &str) -> String {
    let s = atom.to_uppercase();
    // Remove trailing digits for comparison
    s.trim_end_matches(|c: char| c.is_ascii_digit()).to_string()
}

/// Score a carbon atom using soft matching with a ShiftScorer.
fn score_carbon_soft_with_scorer(
    spin_system: &SpinSystem,
    residue: &str,
    atom: &str,
    scorer: &dyn ShiftScorer,
) -> Option<(f64, f64)> {
    // First check if we have soft candidates
    if let Some(candidates) = spin_system.carbon_candidates.get(atom) {
        if !candidates.is_empty() {
            let mut weighted_log_prob = 0.0;
            let mut total_weight = 0.0;

            for candidate in candidates {
                let prob = scorer.score(residue, atom, candidate.shift).max(MIN_PROB);
                weighted_log_prob += candidate.score * prob.ln();
                total_weight += candidate.score;
            }

            if total_weight > 0.0 {
                return Some((weighted_log_prob / total_weight, total_weight.min(1.0)));
            }
        }
    }

    // Fall back to hard assignment if no candidates
    if let Some(&shift) = spin_system.carbon_shifts.get(atom) {
        let prob = scorer.score(residue, atom, shift).max(MIN_PROB);
        return Some((prob.ln(), 1.0));
    }

    None
}

/// Score a spin system against a single amino acid type.
///
/// Uses **soft matching**: carbon candidates contribute weighted by their match scores,
/// allowing peaks shared between spin systems to influence typing proportionally.
///
/// Note: This is the legacy Gaussian scoring function. For better accuracy with
/// multi-modal distributions, use `score_amino_acid_types_kde` instead.
fn score_single_residue_type(
    spin_system: &SpinSystem,
    residue: &str,
    bmrb: &BMRBDatabase,
    params: &TypingParams,
) -> f64 {
    let mut total_log_prob = 0.0;
    let mut total_weight = 0.0;

    // Score backbone H (always available)
    if let Some(stats) = bmrb.get(residue, "H") {
        let prob = gaussian_probability(spin_system.h_shift, stats.mean, stats.std_dev);
        total_log_prob += params.h_weight * prob.ln();
        total_weight += params.h_weight;
    }

    // Score backbone N (always available)
    if let Some(stats) = bmrb.get(residue, "N") {
        let prob = gaussian_probability(spin_system.n_shift, stats.mean, stats.std_dev);
        total_log_prob += params.n_weight * prob.ln();
        total_weight += params.n_weight;
    }

    // Score all observed carbons
    for (carbon_atom, &shift) in &spin_system.carbon_shifts {
        if let Some(stats) = bmrb.get(residue, carbon_atom) {
            let prob = gaussian_probability(shift, stats.mean, stats.std_dev);
            total_log_prob += params.carbon_weight * prob.ln();
            total_weight += params.carbon_weight;
        }
    }

    // Score carbon candidates (soft matching)
    for (carbon_atom, candidates) in &spin_system.carbon_candidates {
        if let Some(stats) = bmrb.get(residue, carbon_atom) {
            for candidate in candidates {
                let prob = gaussian_probability(candidate.shift, stats.mean, stats.std_dev);
                total_log_prob += params.carbon_weight * candidate.score * prob.ln();
                total_weight += params.carbon_weight * candidate.score;
            }
        }
    }

    // Score all observed protons
    for (proton_atom, &shift) in &spin_system.proton_shifts {
        let atom_name = if residue == "GLY" && proton_atom == "HA" { "HA2" } else { proton_atom.as_str() };
        if let Some(stats) = bmrb.get(residue, atom_name) {
            let prob = gaussian_probability(shift, stats.mean, stats.std_dev);
            total_log_prob += params.proton_weight * prob.ln();
            total_weight += params.proton_weight;
        }
    }

    // Convert log probability back to probability
    if total_weight > 0.0 {
        (total_log_prob / total_weight).exp()
    } else {
        1.0 / 20.0 // Uniform prior if no data
    }
}

/// Score a carbon atom using soft matching with candidates.
///
/// Returns (weighted_log_prob, effective_weight) or None if no data.
/// Uses all candidates weighted by their match scores.
fn score_carbon_soft(
    spin_system: &SpinSystem,
    residue: &str,
    atom: &str,
    bmrb: &BMRBDatabase,
) -> Option<(f64, f64)> {
    let stats = bmrb.get(residue, atom)?;

    // First check if we have soft candidates
    if let Some(candidates) = spin_system.carbon_candidates.get(atom) {
        if !candidates.is_empty() {
            // Use weighted combination of all candidates
            let mut weighted_log_prob = 0.0;
            let mut total_weight = 0.0;

            for candidate in candidates {
                let prob = gaussian_probability(candidate.shift, stats.mean, stats.std_dev);
                // Weight the log probability by the candidate's match score
                weighted_log_prob += candidate.score * prob.ln();
                total_weight += candidate.score;
            }

            if total_weight > 0.0 {
                return Some((weighted_log_prob / total_weight, total_weight.min(1.0)));
            }
        }
    }

    // Fall back to hard assignment if no candidates
    if let Some(&shift) = spin_system.carbon_shifts.get(atom) {
        let prob = gaussian_probability(shift, stats.mean, stats.std_dev);
        return Some((prob.ln(), 1.0));
    }

    None
}

/// Calculate Gaussian probability density.
fn gaussian_probability(value: f64, mean: f64, std_dev: f64) -> f64 {
    let z = (value - mean) / std_dev;
    let prob = (-0.5 * z * z).exp() / (std_dev * (2.0 * std::f64::consts::PI).sqrt());

    // Avoid numerical issues with very small probabilities
    prob.max(1e-10)
}

/// Normalize scores to sum to 1.0.
fn normalize_scores(scores: &mut HashMap<String, f64>) {
    let sum: f64 = scores.values().sum();
    if sum > 0.0 {
        for score in scores.values_mut() {
            *score /= sum;
        }
    }
}

/// Type all spin systems in a list and update their amino_acid_scores.
pub fn type_all_spin_systems(
    spin_systems: &mut [SpinSystem],
    bmrb: &BMRBDatabase,
    params: &TypingParams,
) {
    for ss in spin_systems.iter_mut() {
        let scores = score_amino_acid_types(ss, bmrb, params);
        ss.amino_acid_scores = scores;
    }

    log_typing_statistics(spin_systems);
}

/// Type all spin systems using a ShiftScorer.
pub fn type_all_spin_systems_with_scorer(
    spin_systems: &mut [SpinSystem],
    scorer: &dyn ShiftScorer,
    params: &TypingParams,
) {
    for ss in spin_systems.iter_mut() {
        let scores = score_amino_acid_types_with_scorer(ss, scorer, params);
        ss.amino_acid_scores = scores;
    }

    log_typing_statistics(spin_systems);
}

/// Type all spin systems using KDE scoring (recommended).
pub fn type_all_spin_systems_kde(spin_systems: &mut [SpinSystem], params: &TypingParams) {
    let scorer = KDEScorer::new();
    type_all_spin_systems_with_scorer(spin_systems, &scorer, params);
}

/// Log typing statistics.
fn log_typing_statistics(spin_systems: &[SpinSystem]) {
    let typed_count = spin_systems
        .iter()
        .filter(|ss| {
            ss.best_amino_acid_type()
                .map(|(_, score)| score > 0.2)
                .unwrap_or(false)
        })
        .count();

    tracing::info!(
        "Typed {} of {} spin systems with confidence > 20%",
        typed_count,
        spin_systems.len()
    );
}

/// Get the top N most likely amino acid types for a spin system.
pub fn get_top_types(spin_system: &SpinSystem, n: usize) -> Vec<(&str, f64)> {
    let mut types: Vec<_> = spin_system
        .amino_acid_scores
        .iter()
        .map(|(name, &score)| (name.as_str(), score))
        .collect();

    types.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    types.truncate(n);
    types
}

/// Calculate typing entropy (higher = more ambiguous).
pub fn typing_entropy(spin_system: &SpinSystem) -> f64 {
    spin_system
        .amino_acid_scores
        .values()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.ln())
        .sum()
}

/// Maximum possible entropy (uniform over 20 amino acids).
pub fn max_typing_entropy() -> f64 {
    let p: f64 = 1.0 / 20.0;
    20.0 * (-p * p.ln())
}

/// Normalized typing entropy (0.0 = certain, 1.0 = uniform).
pub fn normalized_typing_entropy(spin_system: &SpinSystem) -> f64 {
    typing_entropy(spin_system) / max_typing_entropy()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::UnlabeledPeak;

    fn make_alanine_spin_system() -> SpinSystem {
        // Create a spin system with ALA-like shifts
        let anchor = UnlabeledPeak::hsqc_15n(123.0, 8.2, 1000.0);
        let mut ss = SpinSystem::new(&anchor);

        // ALA-typical shifts from BMRB
        ss.add_proton_shift("HA", 4.35);
        ss.add_proton_shift("HB", 1.39);
        ss.add_carbon_shift("CA", 53.2);
        ss.add_carbon_shift("CB", 19.0);

        ss
    }

    fn make_glycine_spin_system() -> SpinSystem {
        // Create a spin system with GLY-like shifts
        let anchor = UnlabeledPeak::hsqc_15n(109.0, 8.3, 1000.0);
        let mut ss = SpinSystem::new(&anchor);

        // GLY-typical shifts (no CB!)
        ss.add_proton_shift("HA", 3.97);
        ss.add_carbon_shift("CA", 45.3);
        // Note: no CB

        ss
    }

    fn make_serine_spin_system() -> SpinSystem {
        // Create a spin system with SER-like shifts
        let anchor = UnlabeledPeak::hsqc_15n(116.0, 8.2, 1000.0);
        let mut ss = SpinSystem::new(&anchor);

        // SER-typical shifts (high CB!)
        ss.add_proton_shift("HA", 4.47);
        ss.add_proton_shift("HB", 3.89);
        ss.add_carbon_shift("CA", 58.5);
        ss.add_carbon_shift("CB", 63.5);

        ss
    }

    #[test]
    fn test_alanine_typing() {
        let bmrb = BMRBDatabase::load_embedded();
        let params = TypingParams::default();

        let ss = make_alanine_spin_system();
        let scores = score_amino_acid_types(&ss, &bmrb, &params);

        // ALA should score highest
        let best = scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();

        assert_eq!(
            best.0, "ALA",
            "ALA should score highest, got {} with {:.3}",
            best.0, best.1
        );
        assert!(
            *best.1 > 0.3,
            "ALA score should be > 0.3, got {:.3}",
            best.1
        );
    }

    #[test]
    fn test_glycine_typing() {
        let bmrb = BMRBDatabase::load_embedded();
        let params = TypingParams::default();

        let ss = make_glycine_spin_system();
        let scores = score_amino_acid_types(&ss, &bmrb, &params);

        // GLY should score highest due to unique CA and no CB
        let best = scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();

        assert_eq!(
            best.0, "GLY",
            "GLY should score highest, got {} with {:.3}",
            best.0, best.1
        );
    }

    #[test]
    fn test_serine_typing() {
        let bmrb = BMRBDatabase::load_embedded();
        let params = TypingParams::default();

        let ss = make_serine_spin_system();
        let scores = score_amino_acid_types(&ss, &bmrb, &params);

        // SER or THR should score highest (similar CB)
        let best = scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();

        assert!(
            best.0 == "SER" || best.0 == "THR",
            "SER or THR should score highest, got {} with {:.3}",
            best.0,
            best.1
        );
    }

    #[test]
    fn test_scores_normalized() {
        let bmrb = BMRBDatabase::load_embedded();
        let params = TypingParams::default();

        let ss = make_alanine_spin_system();
        let scores = score_amino_acid_types(&ss, &bmrb, &params);

        let sum: f64 = scores.values().sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "Scores should sum to 1.0, got {:.3}",
            sum
        );
    }

    #[test]
    fn test_typing_entropy() {
        let bmrb = BMRBDatabase::load_embedded();
        let params = TypingParams::default();

        let mut ss = make_alanine_spin_system();
        ss.amino_acid_scores = score_amino_acid_types(&ss, &bmrb, &params);

        let entropy = normalized_typing_entropy(&ss);
        assert!(
            entropy < 0.7,
            "ALA typing entropy should be low, got {:.3}",
            entropy
        );
    }

    #[test]
    fn test_get_top_types() {
        let bmrb = BMRBDatabase::load_embedded();
        let params = TypingParams::default();

        let mut ss = make_alanine_spin_system();
        ss.amino_acid_scores = score_amino_acid_types(&ss, &bmrb, &params);

        let top3 = get_top_types(&ss, 3);
        assert_eq!(top3.len(), 3);
        assert_eq!(top3[0].0, "ALA");
        assert!(top3[0].1 >= top3[1].1);
        assert!(top3[1].1 >= top3[2].1);
    }

    // =========================================================================
    // KDE-based typing tests
    // =========================================================================

    #[test]
    fn test_kde_alanine_typing() {
        let params = TypingParams::default();
        let ss = make_alanine_spin_system();
        let scores = score_amino_acid_types_kde(&ss, &params);

        let best = scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();

        assert_eq!(
            best.0, "ALA",
            "KDE: ALA should score highest, got {} with {:.3}",
            best.0, best.1
        );
    }

    #[test]
    fn test_kde_glycine_typing() {
        let params = TypingParams::default();
        let ss = make_glycine_spin_system();
        let scores = score_amino_acid_types_kde(&ss, &params);

        let best = scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();

        assert_eq!(
            best.0, "GLY",
            "KDE: GLY should score highest, got {} with {:.3}",
            best.0, best.1
        );
    }

    #[test]
    fn test_kde_serine_typing() {
        let params = TypingParams::default();
        let ss = make_serine_spin_system();
        let scores = score_amino_acid_types_kde(&ss, &params);

        let best = scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();

        assert!(
            best.0 == "SER" || best.0 == "THR",
            "KDE: SER or THR should score highest, got {} with {:.3}",
            best.0,
            best.1
        );
    }

    #[test]
    fn test_kde_vs_gaussian_agreement() {
        // Both scorers should agree on clear-cut cases
        let params = TypingParams::default();

        for (name, ss) in [
            ("ALA", make_alanine_spin_system()),
            ("GLY", make_glycine_spin_system()),
        ] {
            let gaussian_scores = score_amino_acid_types_gaussian(&ss, &params);
            let kde_scores = score_amino_acid_types_kde(&ss, &params);

            let gaussian_best = gaussian_scores
                .iter()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap();
            let kde_best = kde_scores
                .iter()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap();

            assert_eq!(
                gaussian_best.0, kde_best.0,
                "For {}: Gaussian ({}) and KDE ({}) should agree",
                name, gaussian_best.0, kde_best.0
            );
        }
    }

    fn make_cysteine_spin_system_reduced() -> SpinSystem {
        // CYS with reduced (free thiol) CB shift ~27 ppm
        let anchor = UnlabeledPeak::hsqc_15n(118.0, 8.3, 1000.0);
        let mut ss = SpinSystem::new(&anchor);
        ss.add_proton_shift("HA", 4.5);
        ss.add_proton_shift("HB", 3.0);
        ss.add_carbon_shift("CA", 58.0);
        ss.add_carbon_shift("CB", 27.0); // Reduced CYS
        ss
    }

    fn make_cysteine_spin_system_oxidized() -> SpinSystem {
        // CYS with oxidized (disulfide) CB shift ~40 ppm
        let anchor = UnlabeledPeak::hsqc_15n(120.0, 8.5, 1000.0);
        let mut ss = SpinSystem::new(&anchor);
        ss.add_proton_shift("HA", 4.8);
        ss.add_proton_shift("HB", 3.2);
        ss.add_carbon_shift("CA", 55.0);
        ss.add_carbon_shift("CB", 40.0); // Oxidized CYS
        ss
    }

    #[test]
    fn test_kde_cysteine_bimodal() {
        // KDE should give similar scores for both reduced and oxidized CYS CB shifts
        // This tests that KDE captures the bimodal distribution
        use crate::inference::scoring::KDEScorer;

        let scorer = KDEScorer::new();

        // CYS CB is bimodal: ~27 ppm (reduced) and ~40 ppm (oxidized)
        let score_27 = scorer.score("CYS", "CB", 27.0);
        let score_33 = scorer.score("CYS", "CB", 33.0); // Valley between modes
        let score_40 = scorer.score("CYS", "CB", 40.0);

        println!(
            "KDE CYS CB scores: d(27)={:.4}, d(33)={:.4}, d(40)={:.4}",
            score_27, score_33, score_40
        );

        // Both modes should have non-trivial density
        assert!(
            score_27 > 0.01,
            "KDE should capture reduced CYS CB mode at ~27 ppm"
        );
        assert!(
            score_40 > 0.01,
            "KDE should capture oxidized CYS CB mode at ~40 ppm"
        );

        // The valley might have lower density (testing bimodality)
        // Note: this depends on how spread the data is
        println!("Ratio 27/33: {:.2}, 40/33: {:.2}", score_27 / score_33.max(0.001), score_40 / score_33.max(0.001));
    }
}
