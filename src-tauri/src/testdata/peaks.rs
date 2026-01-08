//! Ground truth peak structures for test data validation.
//!
//! These structures store the known "correct" peak positions and assignments
//! for synthetic spectra, enabling validation of peak picking and assignment.

use serde::{Deserialize, Serialize};

/// Assignment of a peak to a specific atom in a molecule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtomAssignment {
    /// Residue sequence number (1-based)
    pub residue_seq_code: i32,
    /// Three-letter residue name (e.g., "ALA")
    pub residue_name: String,
    /// Atom name (e.g., "CA", "H", "HB")
    pub atom_name: String,
}

/// A ground truth peak with known position and assignment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroundTruthPeak {
    /// Chemical shift position(s) in ppm.
    /// - 1D: `[H_ppm]`
    /// - 2D HSQC: `[N_ppm, H_ppm]` (F1, F2)
    /// - 2D NOESY/TOCSY: `[H1_ppm, H2_ppm]` (F1, F2)
    pub position_ppm: Vec<f64>,

    /// Peak intensity (normalized, typically 0.0-1.0)
    pub intensity: f64,

    /// Atom(s) giving rise to this peak.
    /// - Most peaks: single atom
    /// - 2D cross-peaks: two atoms (from, to)
    /// - Degenerate peaks: multiple atoms at same position
    pub assignments: Vec<AtomAssignment>,

    /// Peak line width in Hz (optional metadata)
    #[serde(default)]
    pub line_width_hz: Option<f64>,
}

/// Collection of ground truth peaks for a generated spectrum.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroundTruthPeaks {
    /// Molecule name
    pub molecule_name: String,

    /// Amino acid sequence (single-letter)
    pub sequence: String,

    /// Experiment type ("1H", "HSQC", "NOESY", "TOCSY")
    pub experiment_type: String,

    /// Number of dimensions
    pub dimensions: u8,

    /// Axis labels in ppm units
    /// - 1D: `["H"]`
    /// - HSQC: `["N", "H"]`
    pub axis_labels: Vec<String>,

    /// Spectral width for each dimension in ppm
    pub spectral_widths_ppm: Vec<f64>,

    /// Carrier frequency for each dimension in ppm
    pub carrier_offsets_ppm: Vec<f64>,

    /// The ground truth peaks
    pub peaks: Vec<GroundTruthPeak>,
}

impl GroundTruthPeaks {
    /// Create a new empty ground truth collection for 1D spectra.
    pub fn new_1d(molecule_name: String, sequence: String) -> Self {
        Self {
            molecule_name,
            sequence,
            experiment_type: "1H".to_string(),
            dimensions: 1,
            axis_labels: vec!["H".to_string()],
            spectral_widths_ppm: vec![14.0],
            carrier_offsets_ppm: vec![4.7],
            peaks: Vec::new(),
        }
    }

    /// Create a new empty ground truth collection for 15N-HSQC spectra.
    pub fn new_hsqc(molecule_name: String, sequence: String) -> Self {
        Self {
            molecule_name,
            sequence,
            experiment_type: "HSQC".to_string(),
            dimensions: 2,
            axis_labels: vec!["N".to_string(), "H".to_string()],
            spectral_widths_ppm: vec![35.0, 4.0], // N: ~100-135, H: ~6-10
            carrier_offsets_ppm: vec![117.5, 8.0],
            peaks: Vec::new(),
        }
    }

    /// Create a new empty ground truth collection for 13C-HSQC spectra.
    pub fn new_chsqc(molecule_name: String, sequence: String) -> Self {
        Self {
            molecule_name,
            sequence,
            experiment_type: "CHSQC".to_string(),
            dimensions: 2,
            axis_labels: vec!["C".to_string(), "H".to_string()],
            spectral_widths_ppm: vec![80.0, 10.0], // C: ~10-90, H: ~0-10
            carrier_offsets_ppm: vec![50.0, 5.0],
            peaks: Vec::new(),
        }
    }

    /// Create a new empty ground truth collection for NOESY spectra.
    pub fn new_noesy(molecule_name: String, sequence: String) -> Self {
        Self {
            molecule_name,
            sequence,
            experiment_type: "NOESY".to_string(),
            dimensions: 2,
            axis_labels: vec!["H".to_string(), "H".to_string()],
            spectral_widths_ppm: vec![12.0, 12.0],
            carrier_offsets_ppm: vec![4.7, 4.7],
            peaks: Vec::new(),
        }
    }

    /// Create a new empty ground truth collection for TOCSY spectra.
    pub fn new_tocsy(molecule_name: String, sequence: String) -> Self {
        Self {
            molecule_name,
            sequence,
            experiment_type: "TOCSY".to_string(),
            dimensions: 2,
            axis_labels: vec!["H".to_string(), "H".to_string()],
            spectral_widths_ppm: vec![12.0, 12.0],
            carrier_offsets_ppm: vec![4.7, 4.7],
            peaks: Vec::new(),
        }
    }

    /// Add a peak to the collection.
    pub fn add_peak(&mut self, peak: GroundTruthPeak) {
        self.peaks.push(peak);
    }

    /// Get number of peaks.
    pub fn len(&self) -> usize {
        self.peaks.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.peaks.is_empty()
    }

    /// Find peaks within tolerance of a given position.
    ///
    /// # Arguments
    /// * `position` - Position in ppm (same dimensionality as peaks)
    /// * `tolerance` - Tolerance in ppm for each dimension
    ///
    /// # Returns
    /// Vector of matching peaks.
    pub fn find_peaks_near(&self, position: &[f64], tolerance: &[f64]) -> Vec<&GroundTruthPeak> {
        self.peaks
            .iter()
            .filter(|p| {
                p.position_ppm
                    .iter()
                    .zip(position.iter())
                    .zip(tolerance.iter())
                    .all(|((peak_pos, query_pos), tol)| (peak_pos - query_pos).abs() <= *tol)
            })
            .collect()
    }

    /// Calculate match rate between picked peaks and ground truth.
    ///
    /// # Arguments
    /// * `picked_positions` - Picked peak positions in ppm
    /// * `tolerance` - Matching tolerance in ppm per dimension
    ///
    /// # Returns
    /// Tuple of (true_positives, false_positives, false_negatives, precision, recall)
    pub fn calculate_match_rate(
        &self,
        picked_positions: &[Vec<f64>],
        tolerance: &[f64],
    ) -> (usize, usize, usize, f64, f64) {
        let mut matched_gt: Vec<bool> = vec![false; self.peaks.len()];
        let mut true_positives = 0;
        let mut false_positives = 0;

        for picked in picked_positions {
            let mut found_match = false;
            for (i, gt_peak) in self.peaks.iter().enumerate() {
                if !matched_gt[i] {
                    let matches = gt_peak
                        .position_ppm
                        .iter()
                        .zip(picked.iter())
                        .zip(tolerance.iter())
                        .all(|((gt, p), t)| (gt - p).abs() <= *t);

                    if matches {
                        matched_gt[i] = true;
                        found_match = true;
                        true_positives += 1;
                        break;
                    }
                }
            }
            if !found_match {
                false_positives += 1;
            }
        }

        let false_negatives = matched_gt.iter().filter(|&&m| !m).count();

        let precision = if true_positives + false_positives > 0 {
            true_positives as f64 / (true_positives + false_positives) as f64
        } else {
            0.0
        };

        let recall = if true_positives + false_negatives > 0 {
            true_positives as f64 / (true_positives + false_negatives) as f64
        } else {
            0.0
        };

        (true_positives, false_positives, false_negatives, precision, recall)
    }
}

impl GroundTruthPeak {
    /// Create a 1D peak from a single proton.
    pub fn new_1d(
        shift_ppm: f64,
        intensity: f64,
        residue_seq_code: i32,
        residue_name: &str,
        atom_name: &str,
    ) -> Self {
        Self {
            position_ppm: vec![shift_ppm],
            intensity,
            assignments: vec![AtomAssignment {
                residue_seq_code,
                residue_name: residue_name.to_string(),
                atom_name: atom_name.to_string(),
            }],
            line_width_hz: None,
        }
    }

    /// Create a 2D HSQC peak (H-N correlation).
    pub fn new_hsqc(
        n_shift: f64,
        h_shift: f64,
        intensity: f64,
        residue_seq_code: i32,
        residue_name: &str,
    ) -> Self {
        Self {
            position_ppm: vec![n_shift, h_shift],
            intensity,
            assignments: vec![
                AtomAssignment {
                    residue_seq_code,
                    residue_name: residue_name.to_string(),
                    atom_name: "N".to_string(),
                },
                AtomAssignment {
                    residue_seq_code,
                    residue_name: residue_name.to_string(),
                    atom_name: "H".to_string(),
                },
            ],
            line_width_hz: None,
        }
    }

    /// Create a 2D cross-peak (NOESY/TOCSY).
    pub fn new_cross_peak(
        h1_shift: f64,
        h2_shift: f64,
        intensity: f64,
        from_assignment: AtomAssignment,
        to_assignment: AtomAssignment,
    ) -> Self {
        Self {
            position_ppm: vec![h1_shift, h2_shift],
            intensity,
            assignments: vec![from_assignment, to_assignment],
            line_width_hz: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_1d() {
        let gt = GroundTruthPeaks::new_1d("test".to_string(), "ACD".to_string());
        assert_eq!(gt.dimensions, 1);
        assert_eq!(gt.experiment_type, "1H");
    }

    #[test]
    fn test_find_peaks_near() {
        let mut gt = GroundTruthPeaks::new_1d("test".to_string(), "A".to_string());
        gt.add_peak(GroundTruthPeak::new_1d(8.0, 1.0, 1, "ALA", "H"));
        gt.add_peak(GroundTruthPeak::new_1d(4.2, 1.0, 1, "ALA", "HA"));

        let found = gt.find_peaks_near(&[8.05], &[0.1]);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].assignments[0].atom_name, "H");
    }

    #[test]
    fn test_match_rate() {
        let mut gt = GroundTruthPeaks::new_1d("test".to_string(), "A".to_string());
        gt.add_peak(GroundTruthPeak::new_1d(8.0, 1.0, 1, "ALA", "H"));
        gt.add_peak(GroundTruthPeak::new_1d(4.2, 1.0, 1, "ALA", "HA"));
        gt.add_peak(GroundTruthPeak::new_1d(1.4, 1.0, 1, "ALA", "HB"));

        // Picked: 2 correct, 1 false positive, 1 missed
        let picked = vec![vec![8.02], vec![4.18], vec![3.0]];
        let (tp, fp, fn_, prec, recall) = gt.calculate_match_rate(&picked, &[0.1]);

        assert_eq!(tp, 2);
        assert_eq!(fp, 1);
        assert_eq!(fn_, 1);
        assert!((prec - 2.0 / 3.0).abs() < 0.01);
        assert!((recall - 2.0 / 3.0).abs() < 0.01);
    }
}
