//! Scoring functions for NMR assignments.
//!
//! This module provides both Gaussian-based scoring (using mean/std_dev from BMRB)
//! and KDE-based scoring (using full empirical distributions) for chemical shift
//! probability evaluation.

use crate::data::BMRBStatistics;
use crate::testdata::{BMRBDatabase, KDEDatabase};

/// Score an assignment based on BMRB statistics.
pub fn score_bmrb_consistency(
    shift: f64,
    residue_name: &str,
    atom_name: &str,
    stats: &[BMRBStatistics],
) -> f64 {
    let stat = match BMRBStatistics::for_atom(stats, residue_name, atom_name) {
        Some(s) => s,
        None => return 0.0, // Unknown atom type, no penalty
    };

    // Gaussian probability
    let z = (shift - stat.mean) / stat.std_dev;
    (-0.5 * z * z).exp()
}

/// Score peak position consistency.
pub fn score_peak_consistency(shift: f64, peak_position: f64, tolerance: f64) -> f64 {
    let diff = (shift - peak_position).abs();
    if diff <= tolerance {
        1.0 - (diff / tolerance)
    } else {
        0.0
    }
}

/// Score sequential connectivity between two residues.
pub fn score_sequential_connectivity(
    shift_i: f64,
    shift_i_plus_1: f64,
    expected_diff: f64,
    tolerance: f64,
) -> f64 {
    let actual_diff = shift_i_plus_1 - shift_i;
    let error = (actual_diff - expected_diff).abs();

    if error <= tolerance {
        1.0 - (error / tolerance)
    } else {
        (-error / tolerance).exp()
    }
}

/// Combined scoring function for an assignment.
#[derive(Debug, Clone)]
pub struct AssignmentScore {
    pub bmrb_score: f64,
    pub peak_score: f64,
    pub connectivity_score: f64,
    pub total: f64,
}

impl AssignmentScore {
    pub fn compute(
        bmrb: f64,
        peak: f64,
        connectivity: f64,
        weights: &ScoreWeights,
    ) -> Self {
        let total = weights.bmrb * bmrb
            + weights.peak * peak
            + weights.connectivity * connectivity;

        Self {
            bmrb_score: bmrb,
            peak_score: peak,
            connectivity_score: connectivity,
            total,
        }
    }
}

/// Weights for different scoring components.
#[derive(Debug, Clone)]
pub struct ScoreWeights {
    pub bmrb: f64,
    pub peak: f64,
    pub connectivity: f64,
}

impl Default for ScoreWeights {
    fn default() -> Self {
        Self {
            bmrb: 1.0,
            peak: 2.0,
            connectivity: 1.5,
        }
    }
}

// =============================================================================
// ShiftScorer Trait - Unified interface for chemical shift scoring
// =============================================================================

/// Trait for scoring chemical shifts against expected distributions.
///
/// This trait provides a unified interface for different scoring methods:
/// - `GaussianScorer`: Uses mean/std_dev from BMRB statistics (simple, fast)
/// - `KDEScorer`: Uses full empirical distributions from BMRB (accurate, handles multi-modal)
pub trait ShiftScorer: Send + Sync {
    /// Score a chemical shift for a residue/atom combination.
    ///
    /// Returns a probability density value (higher = more likely).
    fn score(&self, residue: &str, atom: &str, shift: f64) -> f64;

    /// Score using log probability (more numerically stable for products).
    ///
    /// Returns log(density) or a large negative value for unlikely shifts.
    fn log_score(&self, residue: &str, atom: &str, shift: f64) -> f64;
}

/// Gaussian scorer using mean/std_dev from BMRB statistics.
///
/// This is the traditional approach: assumes chemical shifts follow a
/// Gaussian distribution with the mean and standard deviation from BMRB.
/// Simple and fast, but doesn't capture multi-modal distributions (e.g., CYS CB).
pub struct GaussianScorer {
    bmrb: BMRBDatabase,
}

impl GaussianScorer {
    /// Create a new Gaussian scorer with the embedded BMRB database.
    pub fn new() -> Self {
        Self {
            bmrb: BMRBDatabase::load_embedded(),
        }
    }

    /// Create from an existing BMRBDatabase reference.
    pub fn from_database(bmrb: BMRBDatabase) -> Self {
        Self { bmrb }
    }
}

impl Default for GaussianScorer {
    fn default() -> Self {
        Self::new()
    }
}

impl ShiftScorer for GaussianScorer {
    fn score(&self, residue: &str, atom: &str, shift: f64) -> f64 {
        if let Some(stats) = self.bmrb.get(residue, atom) {
            let z = (shift - stats.mean) / stats.std_dev;
            // Gaussian PDF (unnormalized, but consistent)
            (-0.5 * z * z).exp()
        } else {
            0.0
        }
    }

    fn log_score(&self, residue: &str, atom: &str, shift: f64) -> f64 {
        if let Some(stats) = self.bmrb.get(residue, atom) {
            let z = (shift - stats.mean) / stats.std_dev;
            -0.5 * z * z
        } else {
            -100.0
        }
    }
}

/// KDE scorer using precomputed empirical distributions from BMRB.
///
/// This approach uses Kernel Density Estimation on real BMRB observations
/// to capture the true shape of chemical shift distributions, including:
/// - Multi-modal patterns (e.g., CYS CB with/without disulfide)
/// - Skewed distributions
/// - Non-Gaussian tails
pub struct KDEScorer {
    kde: KDEDatabase,
}

impl KDEScorer {
    /// Create a new KDE scorer with the embedded KDE database.
    pub fn new() -> Self {
        Self {
            kde: KDEDatabase::load_embedded(),
        }
    }

    /// Create from an existing KDEDatabase reference.
    pub fn from_database(kde: KDEDatabase) -> Self {
        Self { kde }
    }
}

impl Default for KDEScorer {
    fn default() -> Self {
        Self::new()
    }
}

impl ShiftScorer for KDEScorer {
    fn score(&self, residue: &str, atom: &str, shift: f64) -> f64 {
        self.kde.density(residue, atom, shift)
    }

    fn log_score(&self, residue: &str, atom: &str, shift: f64) -> f64 {
        self.kde.log_density(residue, atom, shift)
    }
}

/// Enum wrapper to allow runtime selection of scorer type.
pub enum DynamicScorer {
    Gaussian(GaussianScorer),
    KDE(KDEScorer),
}

impl DynamicScorer {
    /// Create a Gaussian scorer.
    pub fn gaussian() -> Self {
        Self::Gaussian(GaussianScorer::new())
    }

    /// Create a KDE scorer.
    pub fn kde() -> Self {
        Self::KDE(KDEScorer::new())
    }
}

impl ShiftScorer for DynamicScorer {
    fn score(&self, residue: &str, atom: &str, shift: f64) -> f64 {
        match self {
            Self::Gaussian(s) => s.score(residue, atom, shift),
            Self::KDE(s) => s.score(residue, atom, shift),
        }
    }

    fn log_score(&self, residue: &str, atom: &str, shift: f64) -> f64 {
        match self {
            Self::Gaussian(s) => s.log_score(residue, atom, shift),
            Self::KDE(s) => s.log_score(residue, atom, shift),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bmrb_scoring() {
        let stats = BMRBStatistics::default_protein_stats();

        // CA at expected value should score high
        let score_good = score_bmrb_consistency(56.0, "ALA", "CA", &stats);
        assert!(score_good > 0.8);

        // CA far from expected should score low
        let score_bad = score_bmrb_consistency(100.0, "ALA", "CA", &stats);
        assert!(score_bad < 0.01);
    }

    #[test]
    fn test_peak_scoring() {
        let score_exact = score_peak_consistency(8.0, 8.0, 0.1);
        assert!((score_exact - 1.0).abs() < 1e-10);

        let score_close = score_peak_consistency(8.05, 8.0, 0.1);
        assert!(score_close > 0.4);

        let score_far = score_peak_consistency(8.5, 8.0, 0.1);
        assert!(score_far < 1e-10);
    }

    #[test]
    fn test_connectivity_scoring() {
        // Expected CA(i) - CA(i-1) difference is ~0 for same residue types
        let score_good = score_sequential_connectivity(54.0, 54.5, 0.0, 2.0);
        assert!(score_good > 0.7);
    }

    #[test]
    fn test_gaussian_scorer() {
        let scorer = GaussianScorer::new();

        // ALA CA at typical value should score high
        let score = scorer.score("ALA", "CA", 53.0);
        assert!(score > 0.5, "ALA CA at 53 ppm should score well");

        // ALA CA far from typical should score low
        let score_bad = scorer.score("ALA", "CA", 100.0);
        assert!(score_bad < 0.01, "ALA CA at 100 ppm should score poorly");
    }

    #[test]
    fn test_kde_scorer() {
        let scorer = KDEScorer::new();

        // ALA CA at typical value should score high
        let score = scorer.score("ALA", "CA", 53.0);
        assert!(score > 0.1, "ALA CA at 53 ppm should have high KDE density");

        // ALA CA far from typical should score low
        let score_bad = scorer.score("ALA", "CA", 100.0);
        assert_eq!(score_bad, 0.0, "ALA CA at 100 ppm should be out of range");
    }

    #[test]
    fn test_scorer_trait_polymorphism() {
        // Test that both scorers work through the trait interface
        let scorers: Vec<Box<dyn ShiftScorer>> = vec![
            Box::new(GaussianScorer::new()),
            Box::new(KDEScorer::new()),
        ];

        for scorer in &scorers {
            let score = scorer.score("ALA", "CA", 53.0);
            assert!(score > 0.0, "Both scorers should give positive score for typical shift");
        }
    }

    #[test]
    fn test_dynamic_scorer() {
        let gaussian = DynamicScorer::gaussian();
        let kde = DynamicScorer::kde();

        // Both should give reasonable scores for typical shifts
        let g_score = gaussian.score("ALA", "CA", 53.0);
        let k_score = kde.score("ALA", "CA", 53.0);

        assert!(g_score > 0.0);
        assert!(k_score > 0.0);

        // Log scores should also work
        let g_log = gaussian.log_score("ALA", "CA", 53.0);
        let k_log = kde.log_score("ALA", "CA", 53.0);

        assert!(g_log > -100.0);
        assert!(k_log > -100.0);
    }

    #[test]
    fn test_cys_cb_bimodal_kde_advantage() {
        // CYS CB is bimodal: ~27 ppm (reduced) and ~40 ppm (oxidized)
        // KDE should capture this better than Gaussian
        let gaussian = GaussianScorer::new();
        let kde = KDEScorer::new();

        // Get scores at both modes
        let kde_27 = kde.score("CYS", "CB", 27.0);
        let kde_40 = kde.score("CYS", "CB", 40.0);

        // KDE should give non-trivial density at both modes
        println!("KDE CYS CB: d(27)={:.4}, d(40)={:.4}", kde_27, kde_40);
        assert!(kde_27 > 0.01, "KDE should capture reduced CYS CB mode");
        assert!(kde_40 > 0.01, "KDE should capture oxidized CYS CB mode");

        // Gaussian will score one mode much higher than the other
        // (depending on where the mean is)
        let gauss_27 = gaussian.score("CYS", "CB", 27.0);
        let gauss_40 = gaussian.score("CYS", "CB", 40.0);
        println!("Gaussian CYS CB: d(27)={:.4}, d(40)={:.4}", gauss_27, gauss_40);
    }
}
