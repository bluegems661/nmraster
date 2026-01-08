//! BMRB chemical shift statistics database.
//!
//! Provides access to chemical shift statistics from the BioMagResBank (BMRB)
//! for generating realistic synthetic NMR spectra.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Chemical shift statistics for a specific atom type in a specific residue.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtomShiftStatistics {
    pub residue_name: String,
    pub atom_name: String,
    pub nucleus: String,
    pub mean: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub count: u32,
}

/// Root structure for the BMRB statistics JSON file.
#[derive(Debug, Deserialize)]
struct BMRBStatsFile {
    #[allow(dead_code)]
    version: String,
    #[allow(dead_code)]
    source: String,
    #[allow(dead_code)]
    description: String,
    statistics: Vec<AtomShiftStatistics>,
}

/// Database of BMRB chemical shift statistics.
///
/// Provides fast lookup of expected chemical shifts by residue and atom name.
pub struct BMRBDatabase {
    /// Statistics indexed by (residue_name, atom_name)
    stats: HashMap<(String, String), AtomShiftStatistics>,
    /// All statistics for quick iteration
    all_stats: Vec<AtomShiftStatistics>,
}

impl BMRBDatabase {
    /// Load the embedded BMRB statistics from the data file.
    ///
    /// The JSON file is embedded at compile time for portability.
    pub fn load_embedded() -> Self {
        const BMRB_JSON: &str = include_str!("../../data/bmrb_stats.json");

        let file: BMRBStatsFile = serde_json::from_str(BMRB_JSON)
            .expect("Failed to parse embedded BMRB statistics JSON");

        let mut stats = HashMap::new();
        for stat in &file.statistics {
            let key = (stat.residue_name.clone(), stat.atom_name.clone());
            stats.insert(key, stat.clone());
        }

        Self {
            stats,
            all_stats: file.statistics,
        }
    }

    /// Get statistics for a specific residue and atom combination.
    ///
    /// # Arguments
    /// * `residue` - Three-letter residue code (e.g., "ALA", "GLY")
    /// * `atom` - Atom name (e.g., "H", "CA", "HB")
    ///
    /// # Returns
    /// Statistics if found, None otherwise.
    pub fn get(&self, residue: &str, atom: &str) -> Option<&AtomShiftStatistics> {
        let key = (residue.to_uppercase(), atom.to_uppercase());
        self.stats.get(&key)
    }

    /// Get all atom statistics for a specific residue.
    ///
    /// # Arguments
    /// * `residue` - Three-letter residue code (e.g., "ALA", "GLY")
    ///
    /// # Returns
    /// Vector of references to all atoms for that residue.
    pub fn get_residue_atoms(&self, residue: &str) -> Vec<&AtomShiftStatistics> {
        let residue_upper = residue.to_uppercase();
        self.all_stats
            .iter()
            .filter(|s| s.residue_name == residue_upper)
            .collect()
    }

    /// Get all atom statistics for a specific nucleus type.
    ///
    /// # Arguments
    /// * `nucleus` - Nucleus type ("H", "C", or "N")
    ///
    /// # Returns
    /// Vector of references to all atoms of that nucleus type.
    pub fn get_nucleus_atoms(&self, nucleus: &str) -> Vec<&AtomShiftStatistics> {
        let nucleus_upper = nucleus.to_uppercase();
        self.all_stats
            .iter()
            .filter(|s| s.nucleus == nucleus_upper)
            .collect()
    }

    /// Get all proton (1H) atoms for a specific residue.
    ///
    /// Useful for generating 1D proton spectra.
    pub fn get_residue_protons(&self, residue: &str) -> Vec<&AtomShiftStatistics> {
        let residue_upper = residue.to_uppercase();
        self.all_stats
            .iter()
            .filter(|s| s.residue_name == residue_upper && s.nucleus == "H")
            .collect()
    }

    /// Get backbone atoms (H, HA, N, CA, CB, C) for a specific residue.
    ///
    /// Note: GLY has HA2/HA3 instead of HA.
    pub fn get_backbone_atoms(&self, residue: &str) -> Vec<&AtomShiftStatistics> {
        let residue_upper = residue.to_uppercase();
        let backbone_atoms = if residue_upper == "GLY" {
            vec!["H", "HA2", "HA3", "N", "CA", "C"]
        } else {
            vec!["H", "HA", "N", "CA", "CB", "C"]
        };

        self.all_stats
            .iter()
            .filter(|s| {
                s.residue_name == residue_upper
                    && backbone_atoms.contains(&s.atom_name.as_str())
            })
            .collect()
    }

    /// Get the expected chemical shift (mean value) for a residue-atom pair.
    ///
    /// Convenience method that returns just the mean shift value.
    pub fn get_shift(&self, residue: &str, atom: &str) -> Option<f64> {
        self.get(residue, atom).map(|s| s.mean)
    }

    /// List all unique residue names in the database.
    pub fn residue_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.all_stats
            .iter()
            .map(|s| s.residue_name.clone())
            .collect();
        names.sort();
        names.dedup();
        names
    }

    /// Get total number of entries in the database.
    pub fn len(&self) -> usize {
        self.all_stats.len()
    }

    /// Check if the database is empty.
    pub fn is_empty(&self) -> bool {
        self.all_stats.is_empty()
    }
}

impl Default for BMRBDatabase {
    fn default() -> Self {
        Self::load_embedded()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_embedded() {
        let db = BMRBDatabase::load_embedded();
        assert!(!db.is_empty());
        assert!(db.len() > 200); // Should have many entries
    }

    #[test]
    fn test_get_alanine_ca() {
        let db = BMRBDatabase::load_embedded();
        let stats = db.get("ALA", "CA").expect("ALA CA should exist");

        // ALA CA should be around 53 ppm
        assert!(stats.mean > 50.0 && stats.mean < 56.0);
        assert_eq!(stats.nucleus, "C");
    }

    #[test]
    fn test_get_glycine_ha2() {
        let db = BMRBDatabase::load_embedded();
        let stats = db.get("GLY", "HA2").expect("GLY HA2 should exist");

        // GLY HA should be around 4 ppm
        assert!(stats.mean > 3.0 && stats.mean < 5.0);
        assert_eq!(stats.nucleus, "H");
    }

    #[test]
    fn test_get_residue_protons() {
        let db = BMRBDatabase::load_embedded();
        let protons = db.get_residue_protons("ALA");

        // ALA should have H, HA, HB protons
        assert!(protons.len() >= 3);
        assert!(protons.iter().all(|p| p.nucleus == "H"));
    }

    #[test]
    fn test_residue_names() {
        let db = BMRBDatabase::load_embedded();
        let names = db.residue_names();

        // Should have all 20 standard amino acids
        assert_eq!(names.len(), 20);
        assert!(names.contains(&"ALA".to_string()));
        assert!(names.contains(&"TRP".to_string()));
    }

    #[test]
    fn test_case_insensitive() {
        let db = BMRBDatabase::load_embedded();

        // Should work with lowercase
        assert!(db.get("ala", "ca").is_some());
        assert!(db.get("Ala", "Ca").is_some());
    }
}
