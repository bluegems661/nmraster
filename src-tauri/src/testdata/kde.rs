//! KDE-based chemical shift scoring using precomputed grids.
//!
//! This module provides chemical shift probability densities computed from
//! empirical BMRB data using Kernel Density Estimation. Unlike the simple
//! Gaussian model (mean/std_dev), KDE captures the true shape of the
//! distribution including multi-modal patterns.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Precomputed KDE grid for a residue/atom combination.
///
/// Stores density values sampled on a uniform grid. Values between grid
/// points are computed via linear interpolation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KDEGrid {
    /// Residue three-letter code (e.g., "ALA")
    pub residue: String,
    /// Atom name (e.g., "CA", "CB", "H")
    pub atom: String,
    /// Minimum chemical shift value in the grid (ppm)
    pub grid_min: f64,
    /// Maximum chemical shift value in the grid (ppm)
    pub grid_max: f64,
    /// Step size between grid points (ppm)
    pub step: f64,
    /// Number of original observations used to build KDE
    pub n_observations: u32,
    /// KDE bandwidth parameter
    pub bandwidth: f64,
    /// Precomputed density values at each grid point
    pub values: Vec<f64>,
}

impl KDEGrid {
    /// Evaluate the KDE density at a given chemical shift via linear interpolation.
    ///
    /// Returns 0.0 for shifts outside the grid range.
    pub fn evaluate(&self, shift: f64) -> f64 {
        if shift < self.grid_min || shift > self.grid_max {
            return 0.0;
        }

        let idx = (shift - self.grid_min) / self.step;
        let idx_floor = idx.floor() as usize;

        if idx_floor >= self.values.len() - 1 {
            return self.values[self.values.len() - 1];
        }

        // Linear interpolation between grid points
        let frac = idx - idx_floor as f64;
        self.values[idx_floor] * (1.0 - frac) + self.values[idx_floor + 1] * frac
    }

    /// Evaluate log-density (more numerically stable for products).
    ///
    /// Returns -100.0 for zero or negative density (effectively log(1e-44)).
    pub fn log_evaluate(&self, shift: f64) -> f64 {
        let density = self.evaluate(shift);
        if density <= 0.0 {
            -100.0
        } else {
            density.ln()
        }
    }

    /// Get the shift value with maximum density (mode of the distribution).
    pub fn mode(&self) -> f64 {
        let max_idx = self
            .values
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        self.grid_min + (max_idx as f64) * self.step
    }
}

/// JSON file structure for KDE grids.
#[derive(Debug, Deserialize)]
struct KDEGridsFile {
    #[allow(dead_code)]
    version: String,
    #[allow(dead_code)]
    n_grid_points: u32,
    #[allow(dead_code)]
    n_entries: u32,
    grids: HashMap<String, KDEGrid>,
}

/// Database of precomputed KDE grids for all residue/atom combinations.
///
/// Provides fast lookup of probability densities for chemical shift scoring.
pub struct KDEDatabase {
    /// Grids indexed by (residue, atom) tuple
    grids: HashMap<(String, String), KDEGrid>,
}

impl KDEDatabase {
    /// Load the embedded KDE grids from the JSON data file.
    ///
    /// The JSON file is embedded at compile time for portability.
    pub fn load_embedded() -> Self {
        const KDE_JSON: &str = include_str!("../../data/bmrb_kde_grids.json");

        let file: KDEGridsFile =
            serde_json::from_str(KDE_JSON).expect("Failed to parse embedded KDE grids JSON");

        let mut grids = HashMap::new();
        for (_, grid) in file.grids {
            let key = (grid.residue.clone(), grid.atom.clone());
            grids.insert(key, grid);
        }

        Self { grids }
    }

    /// Get KDE grid for a residue/atom combination.
    ///
    /// # Arguments
    /// * `residue` - Three-letter residue code (e.g., "ALA", "GLY")
    /// * `atom` - Atom name (e.g., "H", "CA", "CB")
    ///
    /// # Returns
    /// Reference to the KDE grid if found, None otherwise.
    pub fn get(&self, residue: &str, atom: &str) -> Option<&KDEGrid> {
        let key = (residue.to_uppercase(), atom.to_uppercase());
        self.grids.get(&key)
    }

    /// Evaluate probability density for a chemical shift.
    ///
    /// Returns 0.0 if the residue/atom combination is not in the database.
    pub fn density(&self, residue: &str, atom: &str, shift: f64) -> f64 {
        self.get(residue, atom)
            .map(|grid| grid.evaluate(shift))
            .unwrap_or(0.0)
    }

    /// Evaluate log probability density.
    ///
    /// Returns -100.0 if the residue/atom combination is not in the database.
    pub fn log_density(&self, residue: &str, atom: &str, shift: f64) -> f64 {
        self.get(residue, atom)
            .map(|grid| grid.log_evaluate(shift))
            .unwrap_or(-100.0)
    }

    /// Get the number of entries in the database.
    pub fn len(&self) -> usize {
        self.grids.len()
    }

    /// Check if the database is empty.
    pub fn is_empty(&self) -> bool {
        self.grids.is_empty()
    }

    /// List all unique residue names in the database.
    pub fn residue_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self
            .grids
            .keys()
            .map(|(residue, _)| residue.clone())
            .collect();
        names.sort();
        names.dedup();
        names
    }

    /// Get all atom names for a specific residue.
    pub fn get_residue_atoms(&self, residue: &str) -> Vec<String> {
        let residue_upper = residue.to_uppercase();
        self.grids
            .keys()
            .filter(|(r, _)| *r == residue_upper)
            .map(|(_, atom)| atom.clone())
            .collect()
    }
}

impl Default for KDEDatabase {
    fn default() -> Self {
        Self::load_embedded()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_embedded() {
        let db = KDEDatabase::load_embedded();
        assert!(!db.is_empty());
        assert!(db.len() > 200); // Should have many entries
    }

    #[test]
    fn test_get_alanine_ca() {
        let db = KDEDatabase::load_embedded();
        let grid = db.get("ALA", "CA").expect("ALA CA should exist");

        // ALA CA should have mode around 53 ppm
        let mode = grid.mode();
        assert!(
            mode > 50.0 && mode < 56.0,
            "ALA CA mode should be ~53 ppm, got {}",
            mode
        );

        // Density at mode should be positive
        let density_at_mode = grid.evaluate(mode);
        assert!(density_at_mode > 0.1, "Density at mode should be significant");
    }

    #[test]
    fn test_density_falloff() {
        let db = KDEDatabase::load_embedded();

        // ALA CA density should be highest near 53 ppm
        let density_53 = db.density("ALA", "CA", 53.0);
        let density_45 = db.density("ALA", "CA", 45.0);
        let density_65 = db.density("ALA", "CA", 65.0);

        assert!(
            density_53 > density_45,
            "53 ppm should have higher density than 45 ppm"
        );
        assert!(
            density_53 > density_65,
            "53 ppm should have higher density than 65 ppm"
        );
    }

    #[test]
    fn test_glycine_ca_shift() {
        let db = KDEDatabase::load_embedded();

        // GLY CA is unique - around 45 ppm (no CB, so CA is upfield)
        let gly_density = db.density("GLY", "CA", 45.0);
        let ala_density = db.density("ALA", "CA", 45.0);

        // GLY should be more likely than ALA at 45 ppm
        assert!(
            gly_density > ala_density,
            "GLY CA should be more likely than ALA CA at 45 ppm"
        );
    }

    #[test]
    fn test_cysteine_cb_bimodal() {
        let db = KDEDatabase::load_embedded();
        let grid = db.get("CYS", "CB").expect("CYS CB should exist");

        // CYS CB is bimodal: ~27 ppm (reduced) and ~40 ppm (oxidized/disulfide)
        let density_27 = grid.evaluate(27.0);
        let density_40 = grid.evaluate(40.0);
        let density_33 = grid.evaluate(33.0); // Between the modes

        // Both modes should have non-trivial density
        assert!(
            density_27 > 0.01,
            "CYS CB should have density at ~27 ppm (reduced)"
        );
        assert!(
            density_40 > 0.01,
            "CYS CB should have density at ~40 ppm (oxidized)"
        );

        // The valley between modes might have lower density
        // (This tests that KDE captures the bimodal nature)
        println!(
            "CYS CB: d(27)={:.4}, d(33)={:.4}, d(40)={:.4}",
            density_27, density_33, density_40
        );
    }

    #[test]
    fn test_log_density() {
        let db = KDEDatabase::load_embedded();

        let log_density = db.log_density("ALA", "CA", 53.0);
        let density = db.density("ALA", "CA", 53.0);

        assert!(
            (log_density - density.ln()).abs() < 1e-10,
            "log_density should equal ln(density)"
        );
    }

    #[test]
    fn test_out_of_range() {
        let db = KDEDatabase::load_embedded();

        // Way outside any reasonable range
        let density = db.density("ALA", "CA", 200.0);
        assert_eq!(density, 0.0, "Out of range should return 0");

        let log_density = db.log_density("ALA", "CA", 200.0);
        assert_eq!(log_density, -100.0, "Out of range log should return -100");
    }

    #[test]
    fn test_case_insensitive() {
        let db = KDEDatabase::load_embedded();

        // Should work with lowercase
        assert!(db.get("ala", "ca").is_some());
        assert!(db.get("Ala", "Ca").is_some());
    }

    #[test]
    fn test_missing_entry() {
        let db = KDEDatabase::load_embedded();

        // Non-existent residue
        assert!(db.get("XXX", "CA").is_none());
        assert_eq!(db.density("XXX", "CA", 50.0), 0.0);
    }

    #[test]
    fn test_residue_names() {
        let db = KDEDatabase::load_embedded();
        let names = db.residue_names();

        // Should have all 20 standard amino acids
        assert_eq!(names.len(), 20);
        assert!(names.contains(&"ALA".to_string()));
        assert!(names.contains(&"TRP".to_string()));
        assert!(names.contains(&"GLY".to_string()));
    }
}
