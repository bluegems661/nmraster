//! Experiment metadata and chemical shift data structures.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::spectrum::NucleusType;

/// Metadata for an NMR experiment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experiment {
    pub id: Uuid,
    pub name: String,
    pub experiment_type: String,
    pub date: DateTime<Utc>,
    pub spectrometer_frequency_mhz: f64,
    pub temperature_k: Option<f64>,
    pub ph: Option<f64>,
    pub sample_conditions: Option<String>,
    pub notes: Option<String>,
}

impl Experiment {
    pub fn new(name: &str, experiment_type: &str, frequency_mhz: f64) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.to_string(),
            experiment_type: experiment_type.to_string(),
            date: Utc::now(),
            spectrometer_frequency_mhz: frequency_mhz,
            temperature_k: Some(298.0), // Default 25°C
            ph: None,
            sample_conditions: None,
            notes: None,
        }
    }
}

/// A list of chemical shifts for a molecule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChemicalShiftList {
    pub id: Uuid,
    pub name: String,
    pub molecule_id: Uuid,
    pub experiment_id: Option<Uuid>,
    pub shifts: Vec<ChemicalShift>,
}

impl ChemicalShiftList {
    pub fn new(name: &str, molecule_id: Uuid) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.to_string(),
            molecule_id,
            experiment_id: None,
            shifts: Vec::new(),
        }
    }

    /// Add a chemical shift to the list.
    pub fn add_shift(&mut self, shift: ChemicalShift) {
        self.shifts.push(shift);
    }

    /// Get shifts for a specific atom.
    pub fn get_shift_for_atom(&self, atom_id: &Uuid) -> Option<&ChemicalShift> {
        self.shifts.iter().find(|s| &s.atom_id == atom_id)
    }

    /// Get all shifts for a residue.
    pub fn get_shifts_for_residue(&self, residue_seq_code: i32) -> Vec<&ChemicalShift> {
        self.shifts
            .iter()
            .filter(|s| s.residue_seq_code == residue_seq_code)
            .collect()
    }
}

/// A single chemical shift assignment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChemicalShift {
    pub id: Uuid,
    pub atom_id: Uuid,
    pub atom_name: String,
    pub residue_seq_code: i32,
    pub residue_name: String,
    pub chain_code: String,
    /// Chemical shift value in ppm
    pub value: f64,
    /// Uncertainty in ppm
    pub error: Option<f64>,
    /// Ambiguity code (1 = unique, 2+ = ambiguous)
    pub ambiguity_code: u8,
    /// Confidence in the assignment (0-1)
    pub confidence: f64,
    /// Nucleus type
    pub nucleus: NucleusType,
}

impl ChemicalShift {
    pub fn new(
        atom_id: Uuid,
        atom_name: &str,
        residue_seq_code: i32,
        residue_name: &str,
        chain_code: &str,
        value: f64,
        nucleus: NucleusType,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            atom_id,
            atom_name: atom_name.to_string(),
            residue_seq_code,
            residue_name: residue_name.to_string(),
            chain_code: chain_code.to_string(),
            value,
            error: None,
            ambiguity_code: 1,
            confidence: 1.0,
            nucleus,
        }
    }
}

/// BMRB statistics for expected chemical shifts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BMRBStatistics {
    pub residue_name: String,
    pub atom_name: String,
    pub nucleus: NucleusType,
    pub mean: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub count: u32,
}

impl BMRBStatistics {
    /// Load default BMRB statistics for common atoms.
    /// In a real implementation, this would load from a database.
    pub fn default_protein_stats() -> Vec<Self> {
        vec![
            // Backbone H
            Self {
                residue_name: "*".to_string(),
                atom_name: "H".to_string(),
                nucleus: NucleusType::H1,
                mean: 8.2,
                std_dev: 0.6,
                min: 6.0,
                max: 10.5,
                count: 100000,
            },
            Self {
                residue_name: "*".to_string(),
                atom_name: "HA".to_string(),
                nucleus: NucleusType::H1,
                mean: 4.3,
                std_dev: 0.4,
                min: 3.5,
                max: 5.5,
                count: 100000,
            },
            // Backbone N
            Self {
                residue_name: "*".to_string(),
                atom_name: "N".to_string(),
                nucleus: NucleusType::N15,
                mean: 120.0,
                std_dev: 4.0,
                min: 100.0,
                max: 140.0,
                count: 100000,
            },
            // Backbone C
            Self {
                residue_name: "*".to_string(),
                atom_name: "CA".to_string(),
                nucleus: NucleusType::C13,
                mean: 56.0,
                std_dev: 4.0,
                min: 44.0,
                max: 66.0,
                count: 100000,
            },
            Self {
                residue_name: "*".to_string(),
                atom_name: "CB".to_string(),
                nucleus: NucleusType::C13,
                mean: 35.0,
                std_dev: 10.0,
                min: 15.0,
                max: 75.0,
                count: 100000,
            },
            Self {
                residue_name: "*".to_string(),
                atom_name: "C".to_string(),
                nucleus: NucleusType::C13,
                mean: 175.0,
                std_dev: 2.0,
                min: 170.0,
                max: 180.0,
                count: 100000,
            },
        ]
    }

    /// Get statistics for a specific atom type.
    pub fn for_atom<'a>(stats: &'a [Self], residue_name: &str, atom_name: &str) -> Option<&'a Self> {
        // First try exact match
        stats
            .iter()
            .find(|s| s.residue_name == residue_name && s.atom_name == atom_name)
            .or_else(|| {
                // Fall back to wildcard residue
                stats
                    .iter()
                    .find(|s| s.residue_name == "*" && s.atom_name == atom_name)
            })
    }
}

/// Relaxation data for dynamics analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelaxationData {
    pub id: Uuid,
    pub residue_seq_code: i32,
    pub atom_name: String,
    pub data_type: RelaxationType,
    pub field_strength_mhz: f64,
    pub temperature_k: f64,
    pub value: f64,
    pub error: f64,
}

/// Type of relaxation measurement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RelaxationType {
    T1,
    T2,
    R1,
    R2,
    HetNOE,
    R1rho,
}

/// CPMG relaxation dispersion data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CPMGDispersion {
    pub id: Uuid,
    pub residue_seq_code: i32,
    pub atom_name: String,
    pub field_strength_mhz: f64,
    pub temperature_k: f64,
    /// (nu_cpmg_hz, r2eff, error)
    pub dispersion_points: Vec<(f64, f64, f64)>,
    /// Fitted parameters if available
    pub fitted_parameters: Option<CPMGFitParameters>,
}

/// Fitted parameters for CPMG dispersion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CPMGFitParameters {
    pub model: String,  // e.g., "CR72", "TSMFK01"
    pub kex: f64,       // Exchange rate
    pub pb: f64,        // Population of minor state
    pub delta_omega_ppm: f64,  // Chemical shift difference
    pub r20: f64,       // Intrinsic R2
    pub chi_squared: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_experiment_creation() {
        let exp = Experiment::new("hsqc_test", "HSQC", 600.0);
        assert_eq!(exp.spectrometer_frequency_mhz, 600.0);
        assert_eq!(exp.temperature_k, Some(298.0));
    }

    #[test]
    fn test_chemical_shift_list() {
        let mol_id = Uuid::new_v4();
        let mut list = ChemicalShiftList::new("test_shifts", mol_id);

        let atom_id = Uuid::new_v4();
        let shift = ChemicalShift::new(
            atom_id,
            "CA",
            1,
            "ALA",
            "A",
            52.5,
            NucleusType::C13,
        );

        list.add_shift(shift);

        let found = list.get_shift_for_atom(&atom_id);
        assert!(found.is_some());
        assert_eq!(found.unwrap().value, 52.5);
    }

    #[test]
    fn test_bmrb_lookup() {
        let stats = BMRBStatistics::default_protein_stats();

        let ca_stats = BMRBStatistics::for_atom(&stats, "ALA", "CA");
        assert!(ca_stats.is_some());
        assert!((ca_stats.unwrap().mean - 56.0).abs() < 1.0);
    }
}
