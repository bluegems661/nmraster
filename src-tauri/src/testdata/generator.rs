//! Synthetic NMR spectrum generator using BMRB statistics.
//!
//! Generates realistic test spectra with known ground truth peaks
//! for validating peak picking and assignment algorithms.

use ndarray::{Array1, Array2};
use rand::Rng;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::data::molecule::Molecule;
use crate::data::spectrum::{
    ExperimentType, NucleusType, Spectrum1D, Spectrum2D, SpectrumMetadata,
};

use super::bmrb::BMRBDatabase;
use super::peaks::{AtomAssignment, GroundTruthPeak, GroundTruthPeaks};

/// Parameters for spectrum generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectrumGeneratorParams {
    /// Spectrometer frequency in MHz (e.g., 600.0, 800.0)
    pub spectrometer_frequency_mhz: f64,

    /// Number of points in spectrum (e.g., 8192 for 1D, [256, 1024] for 2D)
    pub num_points: Vec<usize>,

    /// Spectral width in ppm for each dimension
    pub spectral_width_ppm: Vec<f64>,

    /// Carrier offset in ppm for each dimension
    pub carrier_offset_ppm: Vec<f64>,

    /// Line width in Hz (full width at half maximum)
    pub line_width_hz: f64,

    /// Base intensity for peaks (before any scaling)
    pub base_intensity: f64,

    /// Noise level as fraction of base intensity (0.0 = no noise)
    pub noise_level: f64,
}

impl Default for SpectrumGeneratorParams {
    fn default() -> Self {
        Self::default_1d()
    }
}

impl SpectrumGeneratorParams {
    /// Default parameters for 1D proton spectrum.
    pub fn default_1d() -> Self {
        Self {
            spectrometer_frequency_mhz: 600.0,
            num_points: vec![8192],
            spectral_width_ppm: vec![14.0],
            carrier_offset_ppm: vec![4.7],
            line_width_hz: 5.0,
            base_intensity: 1.0,
            noise_level: 0.0,
        }
    }

    /// Default parameters for 2D 15N-HSQC spectrum.
    pub fn default_hsqc() -> Self {
        Self {
            spectrometer_frequency_mhz: 600.0,
            num_points: vec![256, 1024],
            spectral_width_ppm: vec![35.0, 4.0], // N: 100-135, H: 6-10
            carrier_offset_ppm: vec![117.5, 8.0],
            line_width_hz: 15.0, // Broader for 2D
            base_intensity: 1.0,
            noise_level: 0.02, // 2% noise for realistic baseline
        }
    }

    /// Default parameters for 2D 13C-HSQC spectrum.
    pub fn default_chsqc() -> Self {
        Self {
            spectrometer_frequency_mhz: 600.0,
            num_points: vec![512, 1024],
            spectral_width_ppm: vec![80.0, 10.0], // C: 10-90 ppm, H: 0-10 ppm
            carrier_offset_ppm: vec![50.0, 5.0],
            line_width_hz: 20.0, // Broader for 13C
            base_intensity: 1.0,
            noise_level: 0.02, // 2% noise for realistic baseline
        }
    }

    /// Default parameters for 2D NOESY spectrum.
    pub fn default_noesy() -> Self {
        Self {
            spectrometer_frequency_mhz: 600.0,
            num_points: vec![512, 2048],
            spectral_width_ppm: vec![12.0, 12.0],
            carrier_offset_ppm: vec![4.7, 4.7],
            line_width_hz: 10.0,
            base_intensity: 1.0,
            noise_level: 0.02, // 2% noise for realistic baseline
        }
    }

    /// Default parameters for 2D TOCSY spectrum.
    pub fn default_tocsy() -> Self {
        Self {
            spectrometer_frequency_mhz: 600.0,
            num_points: vec![512, 2048],
            spectral_width_ppm: vec![12.0, 12.0],
            carrier_offset_ppm: vec![4.7, 4.7],
            line_width_hz: 10.0,
            base_intensity: 1.0,
            noise_level: 0.02, // 2% noise for realistic baseline
        }
    }
}

/// Synthetic spectrum generator using BMRB statistics.
pub struct SpectrumGenerator {
    bmrb: BMRBDatabase,
    params: SpectrumGeneratorParams,
}

impl SpectrumGenerator {
    /// Create a new spectrum generator.
    pub fn new(params: SpectrumGeneratorParams) -> Self {
        Self {
            bmrb: BMRBDatabase::load_embedded(),
            params,
        }
    }

    /// Create a generator with default 1D parameters.
    pub fn new_1d() -> Self {
        Self::new(SpectrumGeneratorParams::default_1d())
    }

    /// Get reference to the BMRB database.
    pub fn bmrb(&self) -> &BMRBDatabase {
        &self.bmrb
    }

    /// Generate a 1D proton spectrum with ground truth peaks.
    ///
    /// Places Lorentzian peaks at BMRB mean positions for all protons
    /// in the molecule.
    pub fn generate_1d_proton(&self, mol: &Molecule) -> (Spectrum1D, GroundTruthPeaks) {
        let n = self.params.num_points[0];
        let sw_ppm = self.params.spectral_width_ppm[0];
        let offset_ppm = self.params.carrier_offset_ppm[0];
        let sf = self.params.spectrometer_frequency_mhz;
        let sw_hz = sw_ppm * sf;

        // Create PPM axis (high to low, standard NMR convention)
        let ppm_axis = Array1::linspace(offset_ppm + sw_ppm / 2.0, offset_ppm - sw_ppm / 2.0, n);

        // Initialize spectrum data
        let mut real = Array1::zeros(n);
        let mut ground_truth = GroundTruthPeaks::new_1d(mol.name.clone(), mol.sequence.clone());

        // Process each residue using public API
        for residue in mol.residues() {
            let res_name = &residue.residue_name;

            // Get all protons for this residue type
            let protons = self.bmrb.get_residue_protons(res_name);

            for stats in protons {
                let shift_ppm = stats.mean;

                // Skip if outside spectral window
                if shift_ppm < offset_ppm - sw_ppm / 2.0 || shift_ppm > offset_ppm + sw_ppm / 2.0 {
                    continue;
                }

                // Add Lorentzian peak to spectrum
                self.add_lorentzian_1d(&mut real, &ppm_axis, shift_ppm, self.params.base_intensity);

                // Record ground truth
                let peak = GroundTruthPeak::new_1d(
                    shift_ppm,
                    self.params.base_intensity,
                    residue.sequence_code,
                    res_name,
                    &stats.atom_name,
                );
                ground_truth.add_peak(peak);
            }
        }

        // Normalize to max = 1.0
        let max_val = real.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if max_val > 0.0 {
            real.mapv_inplace(|v| v / max_val);
        }

        // Create spectrum metadata
        let metadata = SpectrumMetadata {
            id: Uuid::new_v4(),
            name: format!("{}_1H", mol.name),
            experiment_type: ExperimentType::Proton1D,
            nucleus_types: vec![NucleusType::H1],
            spectral_width_hz: vec![sw_hz],
            carrier_offset_ppm: vec![offset_ppm],
            spectrometer_frequency_mhz: sf,
            original_points: vec![n],
            processed_points: vec![n],
        };

        let spectrum = Spectrum1D {
            metadata,
            real,
            imag: None,
            ppm_axis,
            is_processed: true, // Already in frequency domain
        };

        (spectrum, ground_truth)
    }

    /// Generate predicted peak positions for a molecule (no spectrum data).
    ///
    /// Useful for quick validation of expected peaks without generating
    /// full spectrum arrays.
    pub fn generate_peak_list_1d(&self, mol: &Molecule) -> GroundTruthPeaks {
        let sw_ppm = self.params.spectral_width_ppm[0];
        let offset_ppm = self.params.carrier_offset_ppm[0];

        let mut ground_truth = GroundTruthPeaks::new_1d(mol.name.clone(), mol.sequence.clone());

        for residue in mol.residues() {
            let res_name = &residue.residue_name;

            let protons = self.bmrb.get_residue_protons(res_name);

            for stats in protons {
                let shift_ppm = stats.mean;

                // Skip if outside spectral window
                if shift_ppm < offset_ppm - sw_ppm / 2.0 || shift_ppm > offset_ppm + sw_ppm / 2.0 {
                    continue;
                }

                let peak = GroundTruthPeak::new_1d(
                    shift_ppm,
                    self.params.base_intensity,
                    residue.sequence_code,
                    res_name,
                    &stats.atom_name,
                );
                ground_truth.add_peak(peak);
            }
        }

        ground_truth
    }

    /// Generate expected HSQC peaks for a molecule.
    ///
    /// Returns peaks for backbone H-N correlations only (no sidechain NH2).
    pub fn generate_peak_list_hsqc(&self, mol: &Molecule) -> GroundTruthPeaks {
        let mut ground_truth = GroundTruthPeaks::new_hsqc(mol.name.clone(), mol.sequence.clone());

        for residue in mol.residues() {
            let res_name = &residue.residue_name;

            // Skip N-terminal (no amide) and proline (no amide H)
            if residue.sequence_code == 1 || res_name == "PRO" {
                continue;
            }

            // Get backbone H and N shifts
            let h_stats = self.bmrb.get(res_name, "H");
            let n_stats = self.bmrb.get(res_name, "N");

            if let (Some(h), Some(n)) = (h_stats, n_stats) {
                let peak = GroundTruthPeak::new_hsqc(
                    n.mean,
                    h.mean,
                    self.params.base_intensity,
                    residue.sequence_code,
                    res_name,
                );
                ground_truth.add_peak(peak);
            }

            // Add sidechain NH2 for Asn and Gln (two peaks)
            if res_name == "ASN" {
                if let (Some(hd21), Some(nd2)) =
                    (self.bmrb.get("ASN", "HD21"), self.bmrb.get("ASN", "ND2"))
                {
                    let peak = GroundTruthPeak {
                        position_ppm: vec![nd2.mean, hd21.mean],
                        intensity: self.params.base_intensity * 0.5,
                        assignments: vec![
                            AtomAssignment {
                                residue_seq_code: residue.sequence_code,
                                residue_name: res_name.clone(),
                                atom_name: "ND2".to_string(),
                            },
                            AtomAssignment {
                                residue_seq_code: residue.sequence_code,
                                residue_name: res_name.clone(),
                                atom_name: "HD21".to_string(),
                            },
                        ],
                        line_width_hz: None,
                    };
                    ground_truth.add_peak(peak);
                }
                if let (Some(hd22), Some(nd2)) =
                    (self.bmrb.get("ASN", "HD22"), self.bmrb.get("ASN", "ND2"))
                {
                    let peak = GroundTruthPeak {
                        position_ppm: vec![nd2.mean, hd22.mean],
                        intensity: self.params.base_intensity * 0.5,
                        assignments: vec![
                            AtomAssignment {
                                residue_seq_code: residue.sequence_code,
                                residue_name: res_name.clone(),
                                atom_name: "ND2".to_string(),
                            },
                            AtomAssignment {
                                residue_seq_code: residue.sequence_code,
                                residue_name: res_name.clone(),
                                atom_name: "HD22".to_string(),
                            },
                        ],
                        line_width_hz: None,
                    };
                    ground_truth.add_peak(peak);
                }
            }

            if res_name == "GLN" {
                if let (Some(he21), Some(ne2)) =
                    (self.bmrb.get("GLN", "HE21"), self.bmrb.get("GLN", "NE2"))
                {
                    let peak = GroundTruthPeak {
                        position_ppm: vec![ne2.mean, he21.mean],
                        intensity: self.params.base_intensity * 0.5,
                        assignments: vec![
                            AtomAssignment {
                                residue_seq_code: residue.sequence_code,
                                residue_name: res_name.clone(),
                                atom_name: "NE2".to_string(),
                            },
                            AtomAssignment {
                                residue_seq_code: residue.sequence_code,
                                residue_name: res_name.clone(),
                                atom_name: "HE21".to_string(),
                            },
                        ],
                        line_width_hz: None,
                    };
                    ground_truth.add_peak(peak);
                }
                if let (Some(he22), Some(ne2)) =
                    (self.bmrb.get("GLN", "HE22"), self.bmrb.get("GLN", "NE2"))
                {
                    let peak = GroundTruthPeak {
                        position_ppm: vec![ne2.mean, he22.mean],
                        intensity: self.params.base_intensity * 0.5,
                        assignments: vec![
                            AtomAssignment {
                                residue_seq_code: residue.sequence_code,
                                residue_name: res_name.clone(),
                                atom_name: "NE2".to_string(),
                            },
                            AtomAssignment {
                                residue_seq_code: residue.sequence_code,
                                residue_name: res_name.clone(),
                                atom_name: "HE22".to_string(),
                            },
                        ],
                        line_width_hz: None,
                    };
                    ground_truth.add_peak(peak);
                }
            }
        }

        ground_truth
    }

    /// Generate expected 13C-HSQC peaks for a molecule.
    ///
    /// Returns peaks for all C-H correlations (CA-HA, CB-HB, sidechain).
    pub fn generate_peak_list_chsqc(&self, mol: &Molecule) -> GroundTruthPeaks {
        let mut ground_truth = GroundTruthPeaks::new_chsqc(mol.name.clone(), mol.sequence.clone());

        for residue in mol.residues() {
            let res_name = &residue.residue_name;

            // CA-HA correlation (all residues)
            let ca_stats = self.bmrb.get(res_name, "CA");
            let ha_stats = if res_name == "GLY" {
                // Glycine has HA2 and HA3
                self.bmrb.get("GLY", "HA2")
            } else {
                self.bmrb.get(res_name, "HA")
            };

            if let (Some(ca), Some(ha)) = (ca_stats, ha_stats) {
                let peak = GroundTruthPeak {
                    position_ppm: vec![ca.mean, ha.mean], // [C, H]
                    intensity: self.params.base_intensity,
                    assignments: vec![
                        AtomAssignment {
                            residue_seq_code: residue.sequence_code,
                            residue_name: res_name.clone(),
                            atom_name: "CA".to_string(),
                        },
                        AtomAssignment {
                            residue_seq_code: residue.sequence_code,
                            residue_name: res_name.clone(),
                            atom_name: ha.atom_name.clone(),
                        },
                    ],
                    line_width_hz: None,
                };
                ground_truth.add_peak(peak);
            }

            // Glycine HA3
            if res_name == "GLY" {
                if let (Some(ca), Some(ha3)) = (self.bmrb.get("GLY", "CA"), self.bmrb.get("GLY", "HA3")) {
                    let peak = GroundTruthPeak {
                        position_ppm: vec![ca.mean, ha3.mean],
                        intensity: self.params.base_intensity,
                        assignments: vec![
                            AtomAssignment {
                                residue_seq_code: residue.sequence_code,
                                residue_name: res_name.clone(),
                                atom_name: "CA".to_string(),
                            },
                            AtomAssignment {
                                residue_seq_code: residue.sequence_code,
                                residue_name: res_name.clone(),
                                atom_name: "HA3".to_string(),
                            },
                        ],
                        line_width_hz: None,
                    };
                    ground_truth.add_peak(peak);
                }
            }

            // CB-HB correlation (most residues, not GLY/ALA special cases)
            let cb_stats = self.bmrb.get(res_name, "CB");
            let hb_stats = self.bmrb.get(res_name, "HB");
            let hb2_stats = self.bmrb.get(res_name, "HB2");
            let hb3_stats = self.bmrb.get(res_name, "HB3");

            if let Some(cb) = cb_stats {
                // Single HB (e.g., ILE, THR, VAL)
                if let Some(hb) = hb_stats {
                    let peak = GroundTruthPeak {
                        position_ppm: vec![cb.mean, hb.mean],
                        intensity: self.params.base_intensity,
                        assignments: vec![
                            AtomAssignment {
                                residue_seq_code: residue.sequence_code,
                                residue_name: res_name.clone(),
                                atom_name: "CB".to_string(),
                            },
                            AtomAssignment {
                                residue_seq_code: residue.sequence_code,
                                residue_name: res_name.clone(),
                                atom_name: "HB".to_string(),
                            },
                        ],
                        line_width_hz: None,
                    };
                    ground_truth.add_peak(peak);
                }

                // HB2/HB3 (methylene groups)
                if let Some(hb2) = hb2_stats {
                    let peak = GroundTruthPeak {
                        position_ppm: vec![cb.mean, hb2.mean],
                        intensity: self.params.base_intensity,
                        assignments: vec![
                            AtomAssignment {
                                residue_seq_code: residue.sequence_code,
                                residue_name: res_name.clone(),
                                atom_name: "CB".to_string(),
                            },
                            AtomAssignment {
                                residue_seq_code: residue.sequence_code,
                                residue_name: res_name.clone(),
                                atom_name: "HB2".to_string(),
                            },
                        ],
                        line_width_hz: None,
                    };
                    ground_truth.add_peak(peak);
                }
                if let Some(hb3) = hb3_stats {
                    let peak = GroundTruthPeak {
                        position_ppm: vec![cb.mean, hb3.mean],
                        intensity: self.params.base_intensity,
                        assignments: vec![
                            AtomAssignment {
                                residue_seq_code: residue.sequence_code,
                                residue_name: res_name.clone(),
                                atom_name: "CB".to_string(),
                            },
                            AtomAssignment {
                                residue_seq_code: residue.sequence_code,
                                residue_name: res_name.clone(),
                                atom_name: "HB3".to_string(),
                            },
                        ],
                        line_width_hz: None,
                    };
                    ground_truth.add_peak(peak);
                }
            }

            // Additional sidechain C-H correlations based on residue type
            self.add_sidechain_chsqc_peaks(&mut ground_truth, residue.sequence_code, res_name);
        }

        ground_truth
    }

    /// Add sidechain C-H HSQC peaks for specific residue types.
    fn add_sidechain_chsqc_peaks(
        &self,
        ground_truth: &mut GroundTruthPeaks,
        seq_code: i32,
        res_name: &str,
    ) {
        // Define C-H pairs for each residue type's sidechain
        let ch_pairs: Vec<(&str, &str)> = match res_name {
            "ARG" => vec![("CG", "HG2"), ("CG", "HG3"), ("CD", "HD2"), ("CD", "HD3")],
            "ASN" | "ASP" => vec![], // No aliphatic sidechain C-H beyond CB
            "GLN" | "GLU" => vec![("CG", "HG2"), ("CG", "HG3")],
            "HIS" => vec![("CD2", "HD2"), ("CE1", "HE1")], // Aromatic
            "ILE" => vec![("CG1", "HG12"), ("CG1", "HG13"), ("CG2", "HG2"), ("CD1", "HD1")],
            "LEU" => vec![("CG", "HG"), ("CD1", "HD1"), ("CD2", "HD2")],
            "LYS" => vec![("CG", "HG2"), ("CG", "HG3"), ("CD", "HD2"), ("CD", "HD3"), ("CE", "HE2"), ("CE", "HE3")],
            "MET" => vec![("CG", "HG2"), ("CG", "HG3"), ("CE", "HE")],
            "PHE" | "TYR" => vec![("CD1", "HD1"), ("CD2", "HD2"), ("CE1", "HE1"), ("CE2", "HE2")],
            "PRO" => vec![("CG", "HG2"), ("CG", "HG3"), ("CD", "HD2"), ("CD", "HD3")],
            "SER" => vec![], // Only CB which is handled above
            "THR" => vec![("CG2", "HG2")],
            "TRP" => vec![("CD1", "HD1"), ("CE3", "HE3"), ("CZ2", "HZ2"), ("CZ3", "HZ3"), ("CH2", "HH2")],
            "VAL" => vec![("CG1", "HG1"), ("CG2", "HG2")],
            _ => vec![],
        };

        for (c_atom, h_atom) in ch_pairs {
            if let (Some(c_stats), Some(h_stats)) = (self.bmrb.get(res_name, c_atom), self.bmrb.get(res_name, h_atom)) {
                let peak = GroundTruthPeak {
                    position_ppm: vec![c_stats.mean, h_stats.mean],
                    intensity: self.params.base_intensity * 0.8, // Sidechain slightly weaker
                    assignments: vec![
                        AtomAssignment {
                            residue_seq_code: seq_code,
                            residue_name: res_name.to_string(),
                            atom_name: c_atom.to_string(),
                        },
                        AtomAssignment {
                            residue_seq_code: seq_code,
                            residue_name: res_name.to_string(),
                            atom_name: h_atom.to_string(),
                        },
                    ],
                    line_width_hz: None,
                };
                ground_truth.add_peak(peak);
            }
        }
    }

    /// Generate expected TOCSY peaks for a molecule.
    ///
    /// Cross-peaks between all protons within the same residue (spin system).
    pub fn generate_peak_list_tocsy(&self, mol: &Molecule) -> GroundTruthPeaks {
        let mut ground_truth = GroundTruthPeaks::new_tocsy(mol.name.clone(), mol.sequence.clone());

        for residue in mol.residues() {
            let res_name = &residue.residue_name;

            // Get all protons for this residue
            let protons = self.bmrb.get_residue_protons(res_name);

            // Create cross-peaks between all proton pairs
            for (i, h1) in protons.iter().enumerate() {
                // Diagonal peak
                let diag = GroundTruthPeak::new_cross_peak(
                    h1.mean,
                    h1.mean,
                    self.params.base_intensity,
                    AtomAssignment {
                        residue_seq_code: residue.sequence_code,
                        residue_name: res_name.clone(),
                        atom_name: h1.atom_name.clone(),
                    },
                    AtomAssignment {
                        residue_seq_code: residue.sequence_code,
                        residue_name: res_name.clone(),
                        atom_name: h1.atom_name.clone(),
                    },
                );
                ground_truth.add_peak(diag);

                // Cross-peaks
                for h2 in protons.iter().skip(i + 1) {
                    // Upper triangle
                    let cross = GroundTruthPeak::new_cross_peak(
                        h1.mean,
                        h2.mean,
                        self.params.base_intensity * 0.5, // Cross-peaks weaker
                        AtomAssignment {
                            residue_seq_code: residue.sequence_code,
                            residue_name: res_name.clone(),
                            atom_name: h1.atom_name.clone(),
                        },
                        AtomAssignment {
                            residue_seq_code: residue.sequence_code,
                            residue_name: res_name.clone(),
                            atom_name: h2.atom_name.clone(),
                        },
                    );
                    ground_truth.add_peak(cross);

                    // Lower triangle (symmetric)
                    let cross_sym = GroundTruthPeak::new_cross_peak(
                        h2.mean,
                        h1.mean,
                        self.params.base_intensity * 0.5,
                        AtomAssignment {
                            residue_seq_code: residue.sequence_code,
                            residue_name: res_name.clone(),
                            atom_name: h2.atom_name.clone(),
                        },
                        AtomAssignment {
                            residue_seq_code: residue.sequence_code,
                            residue_name: res_name.clone(),
                            atom_name: h1.atom_name.clone(),
                        },
                    );
                    ground_truth.add_peak(cross_sym);
                }
            }
        }

        ground_truth
    }

    /// Generate expected NOESY peaks for a molecule.
    ///
    /// Includes intraresidue NOEs and sequential (i to i+1) backbone NOEs.
    pub fn generate_peak_list_noesy(&self, mol: &Molecule) -> GroundTruthPeaks {
        let mut ground_truth = GroundTruthPeaks::new_noesy(mol.name.clone(), mol.sequence.clone());

        // Collect residue data for sequential NOEs
        let residues: Vec<_> = mol.residues().collect();

        for (idx, residue) in residues.iter().enumerate() {
            let res_name = &residue.residue_name;

            // Intraresidue NOEs: H-HA, H-HB, HA-HB
            let h = self.bmrb.get(res_name, "H");
            let ha = if res_name == "GLY" {
                self.bmrb.get("GLY", "HA2")
            } else {
                self.bmrb.get(res_name, "HA")
            };
            let hb = self.bmrb.get(res_name, "HB");

            // H-HA (strong intraresidue)
            if let (Some(h_s), Some(ha_s)) = (h, ha) {
                self.add_symmetric_noesy_peaks(
                    &mut ground_truth,
                    h_s.mean,
                    ha_s.mean,
                    0.6, // Strong NOE
                    residue.sequence_code,
                    res_name,
                    &h_s.atom_name,
                    &ha_s.atom_name,
                );
            }

            // H-HB (medium intraresidue, if HB exists)
            if let (Some(h_s), Some(hb_s)) = (h, hb) {
                self.add_symmetric_noesy_peaks(
                    &mut ground_truth,
                    h_s.mean,
                    hb_s.mean,
                    0.3,
                    residue.sequence_code,
                    res_name,
                    &h_s.atom_name,
                    &hb_s.atom_name,
                );
            }

            // Sequential NOEs (daN: Hi to HAi-1)
            if idx > 0 {
                let prev_residue = &residues[idx - 1];
                let prev_res_name = &prev_residue.residue_name;

                let prev_ha = if prev_res_name == "GLY" {
                    self.bmrb.get("GLY", "HA2")
                } else {
                    self.bmrb.get(prev_res_name, "HA")
                };

                // daN: H(i) - HA(i-1)
                if let (Some(h_s), Some(ha_prev)) = (h, prev_ha) {
                    // Cross-peak Hi - HAi-1
                    let cross = GroundTruthPeak::new_cross_peak(
                        h_s.mean,
                        ha_prev.mean,
                        self.params.base_intensity * 0.4, // Sequential NOE
                        AtomAssignment {
                            residue_seq_code: residue.sequence_code,
                            residue_name: res_name.clone(),
                            atom_name: h_s.atom_name.clone(),
                        },
                        AtomAssignment {
                            residue_seq_code: prev_residue.sequence_code,
                            residue_name: prev_res_name.clone(),
                            atom_name: ha_prev.atom_name.clone(),
                        },
                    );
                    ground_truth.add_peak(cross);

                    // Symmetric
                    let cross_sym = GroundTruthPeak::new_cross_peak(
                        ha_prev.mean,
                        h_s.mean,
                        self.params.base_intensity * 0.4,
                        AtomAssignment {
                            residue_seq_code: prev_residue.sequence_code,
                            residue_name: prev_res_name.clone(),
                            atom_name: ha_prev.atom_name.clone(),
                        },
                        AtomAssignment {
                            residue_seq_code: residue.sequence_code,
                            residue_name: res_name.clone(),
                            atom_name: h_s.atom_name.clone(),
                        },
                    );
                    ground_truth.add_peak(cross_sym);
                }
            }
        }

        ground_truth
    }

    // ========================================================================
    // Private helper methods
    // ========================================================================

    /// Add a Lorentzian peak to the 1D spectrum.
    fn add_lorentzian_1d(
        &self,
        data: &mut Array1<f64>,
        ppm_axis: &Array1<f64>,
        center_ppm: f64,
        intensity: f64,
    ) {
        let sf = self.params.spectrometer_frequency_mhz;
        let lw_hz = self.params.line_width_hz;
        let lw_ppm = lw_hz / sf;

        // Lorentzian FWHM = 2 * gamma, so gamma = lw_ppm / 2
        let gamma = lw_ppm / 2.0;

        for (i, &ppm) in ppm_axis.iter().enumerate() {
            let delta = ppm - center_ppm;
            // Lorentzian: L(x) = (gamma/pi) / (delta^2 + gamma^2)
            // Normalized to unit height at center
            let lorentz = gamma * gamma / (delta * delta + gamma * gamma);
            data[i] += intensity * lorentz;
        }
    }

    /// Add symmetric NOESY cross-peaks (both (a,b) and (b,a)).
    fn add_symmetric_noesy_peaks(
        &self,
        ground_truth: &mut GroundTruthPeaks,
        h1_ppm: f64,
        h2_ppm: f64,
        intensity_factor: f64,
        residue_seq_code: i32,
        residue_name: &str,
        atom1_name: &str,
        atom2_name: &str,
    ) {
        let intensity = self.params.base_intensity * intensity_factor;

        // Cross-peak
        let cross = GroundTruthPeak::new_cross_peak(
            h1_ppm,
            h2_ppm,
            intensity,
            AtomAssignment {
                residue_seq_code,
                residue_name: residue_name.to_string(),
                atom_name: atom1_name.to_string(),
            },
            AtomAssignment {
                residue_seq_code,
                residue_name: residue_name.to_string(),
                atom_name: atom2_name.to_string(),
            },
        );
        ground_truth.add_peak(cross);

        // Symmetric cross-peak
        let cross_sym = GroundTruthPeak::new_cross_peak(
            h2_ppm,
            h1_ppm,
            intensity,
            AtomAssignment {
                residue_seq_code,
                residue_name: residue_name.to_string(),
                atom_name: atom2_name.to_string(),
            },
            AtomAssignment {
                residue_seq_code,
                residue_name: residue_name.to_string(),
                atom_name: atom1_name.to_string(),
            },
        );
        ground_truth.add_peak(cross_sym);
    }

    // ========================================================================
    // 2D Spectrum Generation
    // ========================================================================

    /// Generate a 2D 15N-HSQC spectrum with full intensity array.
    ///
    /// Returns the spectrum and ground truth peak list.
    pub fn generate_2d_hsqc(&self, mol: &Molecule) -> (Spectrum2D, GroundTruthPeaks) {
        // 1. Generate peak list using existing method
        let peaks = self.generate_peak_list_hsqc(mol);

        // 2. Extract dimensions from params
        let n_f1 = self.params.num_points[0]; // 15N (indirect)
        let n_f2 = self.params.num_points[1]; // 1H (direct)

        let sw_f1 = self.params.spectral_width_ppm[0];
        let sw_f2 = self.params.spectral_width_ppm[1];
        let offset_f1 = self.params.carrier_offset_ppm[0];
        let offset_f2 = self.params.carrier_offset_ppm[1];

        // 3. Create 2D intensity array
        let mut data = Array2::zeros((n_f1, n_f2));

        // 4. Build PPM axes (high to low, standard NMR convention)
        let ppm_f1 = Array1::linspace(offset_f1 + sw_f1 / 2.0, offset_f1 - sw_f1 / 2.0, n_f1);
        let ppm_f2 = Array1::linspace(offset_f2 + sw_f2 / 2.0, offset_f2 - sw_f2 / 2.0, n_f2);

        // 5. Add each peak as 2D Gaussian
        for peak in &peaks.peaks {
            let center_f1 = peak.position_ppm[0]; // N
            let center_f2 = peak.position_ppm[1]; // H
            self.add_2d_gaussian(&mut data, &ppm_f1, &ppm_f2, center_f1, center_f2, peak.intensity);
        }

        // 6. Optionally add noise
        if self.params.noise_level > 0.0 {
            self.add_gaussian_noise_2d(&mut data, self.params.noise_level);
        }

        // 7. Normalize to max = 1.0
        let max_val = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if max_val > 0.0 {
            data.mapv_inplace(|v| v / max_val);
        }

        // 8. Create metadata
        let sf = self.params.spectrometer_frequency_mhz;
        let sw_f1_hz = sw_f1 * sf * 60.8 / 600.0; // Convert 15N ppm to Hz (approx)
        let sw_f2_hz = sw_f2 * sf; // 1H Hz

        let metadata = SpectrumMetadata {
            id: Uuid::new_v4(),
            name: format!("{}_HSQC", mol.name),
            experiment_type: ExperimentType::HSQC,
            nucleus_types: vec![NucleusType::N15, NucleusType::H1],
            spectral_width_hz: vec![sw_f1_hz, sw_f2_hz],
            carrier_offset_ppm: vec![offset_f1, offset_f2],
            spectrometer_frequency_mhz: sf,
            original_points: vec![n_f1, n_f2],
            processed_points: vec![n_f1, n_f2],
        };

        let spectrum = Spectrum2D {
            metadata,
            data,
            ppm_axis_f1: ppm_f1,
            ppm_axis_f2: ppm_f2,
            is_processed: true,
        };

        (spectrum, peaks)
    }

    /// Generate a 2D 13C-HSQC spectrum with full intensity array.
    pub fn generate_2d_chsqc(&self, mol: &Molecule) -> (Spectrum2D, GroundTruthPeaks) {
        // 1. Generate peak list using existing method
        let peaks = self.generate_peak_list_chsqc(mol);

        // 2. Extract dimensions
        let n_f1 = self.params.num_points[0]; // 13C (indirect)
        let n_f2 = self.params.num_points[1]; // 1H (direct)

        let sw_f1 = self.params.spectral_width_ppm[0];
        let sw_f2 = self.params.spectral_width_ppm[1];
        let offset_f1 = self.params.carrier_offset_ppm[0];
        let offset_f2 = self.params.carrier_offset_ppm[1];

        // 3. Create 2D array
        let mut data = Array2::zeros((n_f1, n_f2));
        let ppm_f1 = Array1::linspace(offset_f1 + sw_f1 / 2.0, offset_f1 - sw_f1 / 2.0, n_f1);
        let ppm_f2 = Array1::linspace(offset_f2 + sw_f2 / 2.0, offset_f2 - sw_f2 / 2.0, n_f2);

        // 4. Add peaks
        for peak in &peaks.peaks {
            let center_f1 = peak.position_ppm[0]; // C
            let center_f2 = peak.position_ppm[1]; // H
            self.add_2d_gaussian(&mut data, &ppm_f1, &ppm_f2, center_f1, center_f2, peak.intensity);
        }

        // 5. Add noise
        if self.params.noise_level > 0.0 {
            self.add_gaussian_noise_2d(&mut data, self.params.noise_level);
        }

        // 6. Normalize
        let max_val = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if max_val > 0.0 {
            data.mapv_inplace(|v| v / max_val);
        }

        // 7. Metadata
        let sf = self.params.spectrometer_frequency_mhz;
        let metadata = SpectrumMetadata {
            id: Uuid::new_v4(),
            name: format!("{}_CHSQC", mol.name),
            experiment_type: ExperimentType::HSQC_13C,
            nucleus_types: vec![NucleusType::C13, NucleusType::H1],
            spectral_width_hz: vec![sw_f1 * sf * 150.9 / 600.0, sw_f2 * sf],
            carrier_offset_ppm: vec![offset_f1, offset_f2],
            spectrometer_frequency_mhz: sf,
            original_points: vec![n_f1, n_f2],
            processed_points: vec![n_f1, n_f2],
        };

        let spectrum = Spectrum2D {
            metadata,
            data,
            ppm_axis_f1: ppm_f1,
            ppm_axis_f2: ppm_f2,
            is_processed: true,
        };

        (spectrum, peaks)
    }

    /// Generate a 2D TOCSY spectrum with full intensity array.
    pub fn generate_2d_tocsy(&self, mol: &Molecule) -> (Spectrum2D, GroundTruthPeaks) {
        let peaks = self.generate_peak_list_tocsy(mol);

        let n_f1 = self.params.num_points[0];
        let n_f2 = self.params.num_points[1];
        let sw_f1 = self.params.spectral_width_ppm[0];
        let sw_f2 = self.params.spectral_width_ppm[1];
        let offset_f1 = self.params.carrier_offset_ppm[0];
        let offset_f2 = self.params.carrier_offset_ppm[1];

        let mut data = Array2::zeros((n_f1, n_f2));
        let ppm_f1 = Array1::linspace(offset_f1 + sw_f1 / 2.0, offset_f1 - sw_f1 / 2.0, n_f1);
        let ppm_f2 = Array1::linspace(offset_f2 + sw_f2 / 2.0, offset_f2 - sw_f2 / 2.0, n_f2);

        for peak in &peaks.peaks {
            let center_f1 = peak.position_ppm[0];
            let center_f2 = peak.position_ppm[1];
            self.add_2d_gaussian(&mut data, &ppm_f1, &ppm_f2, center_f1, center_f2, peak.intensity);
        }

        if self.params.noise_level > 0.0 {
            self.add_gaussian_noise_2d(&mut data, self.params.noise_level);
        }

        let max_val = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if max_val > 0.0 {
            data.mapv_inplace(|v| v / max_val);
        }

        let sf = self.params.spectrometer_frequency_mhz;
        let metadata = SpectrumMetadata {
            id: Uuid::new_v4(),
            name: format!("{}_TOCSY", mol.name),
            experiment_type: ExperimentType::TOCSY,
            nucleus_types: vec![NucleusType::H1, NucleusType::H1],
            spectral_width_hz: vec![sw_f1 * sf, sw_f2 * sf],
            carrier_offset_ppm: vec![offset_f1, offset_f2],
            spectrometer_frequency_mhz: sf,
            original_points: vec![n_f1, n_f2],
            processed_points: vec![n_f1, n_f2],
        };

        let spectrum = Spectrum2D {
            metadata,
            data,
            ppm_axis_f1: ppm_f1,
            ppm_axis_f2: ppm_f2,
            is_processed: true,
        };

        (spectrum, peaks)
    }

    /// Generate a 2D NOESY spectrum with full intensity array.
    pub fn generate_2d_noesy(&self, mol: &Molecule) -> (Spectrum2D, GroundTruthPeaks) {
        let peaks = self.generate_peak_list_noesy(mol);

        let n_f1 = self.params.num_points[0];
        let n_f2 = self.params.num_points[1];
        let sw_f1 = self.params.spectral_width_ppm[0];
        let sw_f2 = self.params.spectral_width_ppm[1];
        let offset_f1 = self.params.carrier_offset_ppm[0];
        let offset_f2 = self.params.carrier_offset_ppm[1];

        let mut data = Array2::zeros((n_f1, n_f2));
        let ppm_f1 = Array1::linspace(offset_f1 + sw_f1 / 2.0, offset_f1 - sw_f1 / 2.0, n_f1);
        let ppm_f2 = Array1::linspace(offset_f2 + sw_f2 / 2.0, offset_f2 - sw_f2 / 2.0, n_f2);

        for peak in &peaks.peaks {
            let center_f1 = peak.position_ppm[0];
            let center_f2 = peak.position_ppm[1];
            self.add_2d_gaussian(&mut data, &ppm_f1, &ppm_f2, center_f1, center_f2, peak.intensity);
        }

        if self.params.noise_level > 0.0 {
            self.add_gaussian_noise_2d(&mut data, self.params.noise_level);
        }

        let max_val = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if max_val > 0.0 {
            data.mapv_inplace(|v| v / max_val);
        }

        let sf = self.params.spectrometer_frequency_mhz;
        let metadata = SpectrumMetadata {
            id: Uuid::new_v4(),
            name: format!("{}_NOESY", mol.name),
            experiment_type: ExperimentType::NOESY,
            nucleus_types: vec![NucleusType::H1, NucleusType::H1],
            spectral_width_hz: vec![sw_f1 * sf, sw_f2 * sf],
            carrier_offset_ppm: vec![offset_f1, offset_f2],
            spectrometer_frequency_mhz: sf,
            original_points: vec![n_f1, n_f2],
            processed_points: vec![n_f1, n_f2],
        };

        let spectrum = Spectrum2D {
            metadata,
            data,
            ppm_axis_f1: ppm_f1,
            ppm_axis_f2: ppm_f2,
            is_processed: true,
        };

        (spectrum, peaks)
    }

    /// Add a 2D Gaussian peak to the spectrum.
    ///
    /// Uses the linewidth from params, converting from Hz to ppm.
    fn add_2d_gaussian(
        &self,
        data: &mut Array2<f64>,
        ppm_f1: &Array1<f64>,
        ppm_f2: &Array1<f64>,
        center_f1: f64,
        center_f2: f64,
        intensity: f64,
    ) {
        // Convert linewidth FWHM (Hz) to Gaussian sigma (ppm)
        // FWHM = 2.355 * sigma, so sigma = FWHM / 2.355
        // Also need to convert Hz to ppm: ppm = Hz / SF (MHz)
        let sf = self.params.spectrometer_frequency_mhz;
        let fwhm_ppm = self.params.line_width_hz / sf;
        let sigma = fwhm_ppm / 2.355;

        // For 2D we use slightly different linewidths in each dimension
        // F1 (indirect) is typically broader than F2 (direct)
        let sigma_f1 = sigma * 2.0; // Broader in indirect dimension
        let sigma_f2 = sigma;

        // Only iterate over points within 5 sigma of peak center
        for (i, &f1) in ppm_f1.iter().enumerate() {
            let d1 = (f1 - center_f1) / sigma_f1;
            if d1.abs() > 5.0 {
                continue;
            }

            for (j, &f2) in ppm_f2.iter().enumerate() {
                let d2 = (f2 - center_f2) / sigma_f2;
                if d2.abs() > 5.0 {
                    continue;
                }

                // 2D Gaussian: I * exp(-((f1-c1)²/(2σ1²) + (f2-c2)²/(2σ2²)))
                let gauss = intensity * (-0.5 * (d1 * d1 + d2 * d2)).exp();
                data[[i, j]] += gauss;
            }
        }
    }

    /// Add Gaussian white noise to a 2D spectrum.
    fn add_gaussian_noise_2d(&self, data: &mut Array2<f64>, noise_level: f64) {
        let mut rng = rand::rng();
        let noise_std = self.params.base_intensity * noise_level;

        for val in data.iter_mut() {
            // Box-Muller transform for Gaussian noise
            let u1: f64 = rng.random();
            let u2: f64 = rng.random();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            *val += z * noise_std;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_molecule() -> Molecule {
        // Simple 3-residue peptide: ACD
        Molecule::from_sequence("test", "ACD", "A")
    }

    #[test]
    fn test_generate_1d_proton() {
        let gen = SpectrumGenerator::new_1d();
        let mol = make_test_molecule();

        let (spectrum, ground_truth) = gen.generate_1d_proton(&mol);

        // Check spectrum properties
        assert_eq!(spectrum.len(), 8192);
        assert!(spectrum.is_processed);
        assert!(!spectrum.real.iter().all(|&v| v == 0.0)); // Has signal

        // Check ground truth
        assert!(!ground_truth.is_empty());
        assert_eq!(ground_truth.experiment_type, "1H");

        // Should have peaks for H, HA, HB for ALA (3 protons minimum)
        // Plus ASP and CYS protons
        assert!(ground_truth.len() >= 9);
    }

    #[test]
    fn test_generate_peak_list_hsqc() {
        let gen = SpectrumGenerator::new(SpectrumGeneratorParams::default_hsqc());
        let mol = make_test_molecule();

        let ground_truth = gen.generate_peak_list_hsqc(&mol);

        // ACD: A is N-terminal (no amide), so only C and D have HSQC peaks
        // = 2 backbone peaks
        assert!(ground_truth.len() >= 2);
        assert_eq!(ground_truth.experiment_type, "HSQC");
        assert_eq!(ground_truth.dimensions, 2);
    }

    #[test]
    fn test_generate_peak_list_tocsy() {
        let gen = SpectrumGenerator::new(SpectrumGeneratorParams::default_tocsy());
        let mol = make_test_molecule();

        let ground_truth = gen.generate_peak_list_tocsy(&mol);

        // TOCSY should have many cross-peaks
        assert!(ground_truth.len() > 10);
        assert_eq!(ground_truth.experiment_type, "TOCSY");
    }

    #[test]
    fn test_generate_peak_list_noesy() {
        let gen = SpectrumGenerator::new(SpectrumGeneratorParams::default_noesy());
        let mol = make_test_molecule();

        let ground_truth = gen.generate_peak_list_noesy(&mol);

        // NOESY should have intraresidue and sequential NOEs
        assert!(ground_truth.len() > 5);
        assert_eq!(ground_truth.experiment_type, "NOESY");
    }

    #[test]
    fn test_peak_positions_reasonable() {
        let gen = SpectrumGenerator::new_1d();
        let mol = make_test_molecule();

        let ground_truth = gen.generate_peak_list_1d(&mol);

        // All peaks should be in reasonable 1H range (0-12 ppm)
        for peak in &ground_truth.peaks {
            let ppm = peak.position_ppm[0];
            assert!(ppm > 0.0 && ppm < 12.0, "Peak at {} ppm is out of range", ppm);
        }
    }

    #[test]
    fn test_generate_2d_hsqc() {
        let gen = SpectrumGenerator::new(SpectrumGeneratorParams::default_hsqc());
        let mol = make_test_molecule();

        let (spectrum, ground_truth) = gen.generate_2d_hsqc(&mol);

        // Check spectrum dimensions
        let (n_f1, n_f2) = spectrum.data.dim();
        assert_eq!(n_f1, 256); // Default indirect points
        assert_eq!(n_f2, 1024); // Default direct points

        // Check PPM axes have correct length
        assert_eq!(spectrum.ppm_axis_f1.len(), n_f1);
        assert_eq!(spectrum.ppm_axis_f2.len(), n_f2);

        // Check data has signal (not all zeros)
        let max_val = spectrum.data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(max_val > 0.0, "Spectrum should have positive signal");

        // Normalized to max = 1.0
        assert!((max_val - 1.0).abs() < 1e-10, "Spectrum should be normalized to max=1.0");

        // Check ground truth matches
        assert!(ground_truth.len() >= 2, "Should have at least 2 backbone peaks");

        // Verify peaks are at expected positions (signal should be non-zero near peak centers)
        for peak in &ground_truth.peaks {
            let n_ppm = peak.position_ppm[0]; // 15N
            let h_ppm = peak.position_ppm[1]; // 1H

            // Find closest indices
            let (i_f1, _) = spectrum.ppm_axis_f1.iter().enumerate()
                .min_by(|(_, a), (_, b)| {
                    ((*a - n_ppm).abs()).partial_cmp(&((*b - n_ppm).abs())).unwrap()
                }).unwrap();
            let (i_f2, _) = spectrum.ppm_axis_f2.iter().enumerate()
                .min_by(|(_, a), (_, b)| {
                    ((*a - h_ppm).abs()).partial_cmp(&((*b - h_ppm).abs())).unwrap()
                }).unwrap();

            // Check signal exists near this peak
            let signal_at_peak = spectrum.data[[i_f1, i_f2]];
            assert!(signal_at_peak > 0.1, "Signal at peak ({}, {}) should be > 0.1, got {}",
                    n_ppm, h_ppm, signal_at_peak);
        }
    }

    #[test]
    fn test_generate_2d_tocsy() {
        let gen = SpectrumGenerator::new(SpectrumGeneratorParams::default_tocsy());
        let mol = make_test_molecule();

        let (spectrum, ground_truth) = gen.generate_2d_tocsy(&mol);

        // Check dimensions
        let (n_f1, n_f2) = spectrum.data.dim();
        assert_eq!(n_f1, 512);
        assert_eq!(n_f2, 2048);

        // Should have many cross-peaks
        assert!(ground_truth.len() > 10, "TOCSY should have many cross-peaks");

        // Check data has signal
        let max_val = spectrum.data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(max_val > 0.0);
    }
}
