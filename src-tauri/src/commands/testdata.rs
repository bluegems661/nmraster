//! Tauri commands for generating realistic test data using BMRB statistics.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tauri::State;
use uuid::Uuid;

use crate::data::molecule::Molecule;
use crate::data::spectrum::Peak;
use crate::error::{NmrError, Result};
use crate::state::AppState;
use crate::testdata::{
    AtomShiftStatistics, BMRBDatabase, GroundTruthPeaks, SpectrumGenerator,
    SpectrumGeneratorParams,
};

/// Parameters for test data generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestDataParams {
    /// Spectrometer frequency in MHz (default: 600.0)
    #[serde(default = "default_spectrometer_freq")]
    pub spectrometer_frequency_mhz: f64,

    /// Number of points for 1D spectra (default: 8192)
    #[serde(default = "default_num_points_1d")]
    pub num_points_1d: usize,

    /// Line width in Hz (default: 5.0)
    #[serde(default = "default_line_width")]
    pub line_width_hz: f64,
}

fn default_spectrometer_freq() -> f64 {
    600.0
}
fn default_num_points_1d() -> usize {
    8192
}
fn default_line_width() -> f64 {
    5.0
}

impl Default for TestDataParams {
    fn default() -> Self {
        Self {
            spectrometer_frequency_mhz: 600.0,
            num_points_1d: 8192,
            line_width_hz: 5.0,
        }
    }
}

/// Result of test data generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestDataResult {
    /// ID of the created molecule
    pub molecule_id: String,

    /// Map of experiment type to spectrum ID
    pub spectra_ids: HashMap<String, String>,

    /// Map of experiment type to ground truth peaks
    pub ground_truth: HashMap<String, GroundTruthPeaks>,

    /// Summary statistics
    pub summary: TestDataSummary,
}

/// Summary of generated test data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestDataSummary {
    pub sequence_length: usize,
    pub num_residues: usize,
    pub experiments_generated: Vec<String>,
    pub total_ground_truth_peaks: usize,
}

/// Convert ground truth peaks to Peak objects for storage.
fn ground_truth_to_peaks(gt: &GroundTruthPeaks, spectrum_id: Uuid, line_width_hz: f64) -> Vec<Peak> {
    gt.peaks
        .iter()
        .map(|gtp| Peak {
            id: Uuid::new_v4(),
            spectrum_id,
            position_ppm: gtp.position_ppm.clone(),
            position_error: vec![0.01; gtp.position_ppm.len()], // Small error for synthetic data
            intensity: gtp.intensity,
            volume: Some(gtp.intensity * 1.0), // Approximate volume
            line_width_hz: vec![line_width_hz; gtp.position_ppm.len()],
            signal_noise_ratio: 100.0, // High SNR for synthetic data
            is_artifact: false,
            figure_of_merit: 1.0, // Perfect quality for synthetic data
            annotation: if !gtp.assignments.is_empty() {
                Some(format!(
                    "{}{}{}",
                    gtp.assignments[0].residue_seq_code,
                    gtp.assignments[0].residue_name,
                    gtp.assignments[0].atom_name
                ))
            } else {
                None
            },
        })
        .collect()
}

/// Generate realistic test data for a peptide sequence.
///
/// Creates a molecule from the sequence and generates synthetic spectra
/// with known ground truth peaks based on BMRB chemical shift statistics.
#[tauri::command]
pub async fn generate_test_data(
    sequence: String,
    name: String,
    experiment_types: Vec<String>,
    params: Option<TestDataParams>,
    state: State<'_, AppState>,
) -> Result<TestDataResult> {
    let params = params.unwrap_or_default();

    tracing::info!(
        "Generating test data for '{}' ({} residues), experiments: {:?}",
        name,
        sequence.len(),
        experiment_types
    );

    // Create molecule from sequence
    let mol = Molecule::from_sequence(&name, &sequence, "A");
    let molecule_id = mol.id.to_string();

    // Store the molecule
    {
        let mut molecules = state.molecules.write();
        molecules.insert(mol.id, mol.clone());
    }
    {
        let mut active = state.active_molecule.write();
        *active = Some(mol.id);
    }

    let mut spectra_ids = HashMap::new();
    let mut ground_truth = HashMap::new();
    let mut total_peaks = 0;

    // Generate spectra for each requested experiment type
    for exp_type in &experiment_types {
        match exp_type.to_uppercase().as_str() {
            "1H" | "PROTON" | "1D" => {
                // Generate 1D proton spectrum
                let gen_params = SpectrumGeneratorParams {
                    spectrometer_frequency_mhz: params.spectrometer_frequency_mhz,
                    num_points: vec![params.num_points_1d],
                    spectral_width_ppm: vec![14.0],
                    carrier_offset_ppm: vec![4.7],
                    line_width_hz: params.line_width_hz,
                    base_intensity: 1.0,
                    noise_level: 0.0,
                };

                let generator = SpectrumGenerator::new(gen_params.clone());
                let (spectrum, gt) = generator.generate_1d_proton(&mol);

                let spectrum_uuid = spectrum.metadata.id;
                let spectrum_id = spectrum_uuid.to_string();
                total_peaks += gt.len();

                // Convert ground truth to peaks and store
                let peaks = ground_truth_to_peaks(&gt, spectrum_uuid, gen_params.line_width_hz);
                state.add_peaks(spectrum_uuid, peaks);

                // Store spectrum
                {
                    let mut spectra = state.spectra_1d.write();
                    spectra.insert(spectrum_uuid, spectrum);
                }

                let num_peaks = gt.len();
                spectra_ids.insert("1H".to_string(), spectrum_id);
                ground_truth.insert("1H".to_string(), gt);

                tracing::info!("Generated 1H spectrum with {} peaks", num_peaks);
            }
            "HSQC" => {
                // Generate HSQC peak list and store as peaks
                let gen_params = SpectrumGeneratorParams::default_hsqc();
                let generator = SpectrumGenerator::new(gen_params.clone());
                let gt = generator.generate_peak_list_hsqc(&mol);

                // Create a virtual spectrum ID for storing peaks
                let virtual_spectrum_id = Uuid::new_v4();
                let peaks = ground_truth_to_peaks(&gt, virtual_spectrum_id, gen_params.line_width_hz);

                // Store peaks
                state.add_peaks(virtual_spectrum_id, peaks);

                total_peaks += gt.len();
                spectra_ids.insert("HSQC".to_string(), virtual_spectrum_id.to_string());
                ground_truth.insert("HSQC".to_string(), gt);

                tracing::info!("Generated HSQC with {} peaks", total_peaks);
            }
            "TOCSY" => {
                let gen_params = SpectrumGeneratorParams::default_tocsy();
                let generator = SpectrumGenerator::new(gen_params.clone());
                let gt = generator.generate_peak_list_tocsy(&mol);

                // Create a virtual spectrum ID for storing peaks
                let virtual_spectrum_id = Uuid::new_v4();
                let peaks = ground_truth_to_peaks(&gt, virtual_spectrum_id, gen_params.line_width_hz);

                // Store peaks
                state.add_peaks(virtual_spectrum_id, peaks);

                let num_peaks = gt.len();
                total_peaks += num_peaks;
                spectra_ids.insert("TOCSY".to_string(), virtual_spectrum_id.to_string());
                ground_truth.insert("TOCSY".to_string(), gt);

                tracing::info!("Generated TOCSY with {} peaks", num_peaks);
            }
            "CHSQC" | "13C-HSQC" | "C-HSQC" => {
                let gen_params = SpectrumGeneratorParams::default_chsqc();
                let generator = SpectrumGenerator::new(gen_params.clone());
                let gt = generator.generate_peak_list_chsqc(&mol);

                // Create a virtual spectrum ID for storing peaks
                let virtual_spectrum_id = Uuid::new_v4();
                let peaks = ground_truth_to_peaks(&gt, virtual_spectrum_id, gen_params.line_width_hz);

                // Store peaks
                state.add_peaks(virtual_spectrum_id, peaks);

                let num_peaks = gt.len();
                total_peaks += num_peaks;
                spectra_ids.insert("CHSQC".to_string(), virtual_spectrum_id.to_string());
                ground_truth.insert("CHSQC".to_string(), gt);

                tracing::info!("Generated 13C-HSQC with {} peaks", num_peaks);
            }
            "NOESY" => {
                let gen_params = SpectrumGeneratorParams::default_noesy();
                let generator = SpectrumGenerator::new(gen_params.clone());
                let gt = generator.generate_peak_list_noesy(&mol);

                // Create a virtual spectrum ID for storing peaks
                let virtual_spectrum_id = Uuid::new_v4();
                let peaks = ground_truth_to_peaks(&gt, virtual_spectrum_id, gen_params.line_width_hz);

                // Store peaks
                state.add_peaks(virtual_spectrum_id, peaks);

                let num_peaks = gt.len();
                total_peaks += num_peaks;
                spectra_ids.insert("NOESY".to_string(), virtual_spectrum_id.to_string());
                ground_truth.insert("NOESY".to_string(), gt);

                tracing::info!("Generated NOESY with {} peaks", num_peaks);
            }
            other => {
                tracing::warn!("Unknown experiment type: {}", other);
            }
        }
    }

    let summary = TestDataSummary {
        sequence_length: sequence.len(),
        num_residues: mol.num_residues(),
        experiments_generated: experiment_types
            .iter()
            .filter(|e| {
                let upper = e.to_uppercase();
                upper == "1H" || upper == "PROTON" || upper == "1D"
                    || upper == "HSQC" || upper == "TOCSY" || upper == "NOESY"
                    || upper == "CHSQC" || upper == "13C-HSQC" || upper == "C-HSQC"
            })
            .cloned()
            .collect(),
        total_ground_truth_peaks: total_peaks,
    };

    tracing::info!(
        "Generated test data: {} spectra, {} total ground truth peaks",
        spectra_ids.len(),
        total_peaks
    );

    Ok(TestDataResult {
        molecule_id,
        spectra_ids,
        ground_truth,
        summary,
    })
}

/// Get BMRB statistics for a specific residue type.
#[tauri::command]
pub async fn get_bmrb_stats(residue_name: String) -> Result<Vec<AtomShiftStatistics>> {
    let db = BMRBDatabase::load_embedded();
    let stats = db.get_residue_atoms(&residue_name);

    Ok(stats.into_iter().cloned().collect())
}

/// Get BMRB statistics for a specific residue and atom.
#[tauri::command]
pub async fn get_bmrb_shift(
    residue_name: String,
    atom_name: String,
) -> Result<Option<AtomShiftStatistics>> {
    let db = BMRBDatabase::load_embedded();
    Ok(db.get(&residue_name, &atom_name).cloned())
}

/// Create a shift list from ground truth peaks.
/// This is useful for testing assignment algorithms against known ground truth.
#[tauri::command]
pub async fn create_shift_list_from_ground_truth(
    ground_truth: GroundTruthPeaks,
    list_name: String,
    state: State<'_, AppState>,
) -> Result<String> {
    use crate::data::{ChemicalShift, ChemicalShiftList, NucleusType};

    // Get active molecule
    let mol_id = {
        let active = state.active_molecule.read();
        active.ok_or_else(|| NmrError::Internal("No active molecule".into()))?
    };

    // Create new shift list
    let mut shift_list = ChemicalShiftList::new(&list_name, mol_id);

    // Get molecule to look up atom IDs
    let mol = state.get_molecule(&mol_id)
        .ok_or_else(|| NmrError::Internal("Molecule not found".into()))?;

    // Add shifts from ground truth peaks
    let mut shifts_added = 0;
    let mut shifts_skipped = 0;

    for gt_peak in &ground_truth.peaks {
        for assignment in &gt_peak.assignments {
            // Get the atom from the molecule
            let atoms = mol.get_atoms_for_residue(assignment.residue_seq_code);
            if let Some(atom) = atoms.iter().find(|a| a.atom_name == assignment.atom_name) {
                // Determine nucleus type from atom name
                let nucleus = if assignment.atom_name.starts_with('H') {
                    NucleusType::H1
                } else if assignment.atom_name.starts_with('C') {
                    NucleusType::C13
                } else if assignment.atom_name.starts_with('N') {
                    NucleusType::N15
                } else {
                    NucleusType::H1
                };

                // Get the appropriate shift value based on experiment type
                let shift_value = if gt_peak.position_ppm.len() == 1 {
                    gt_peak.position_ppm[0]
                } else {
                    // For 2D, use the dimension that matches this atom type
                    if assignment.atom_name.starts_with('H') {
                        // H is typically in the second dimension (F2) for HSQC
                        gt_peak.position_ppm.get(1).copied().unwrap_or(gt_peak.position_ppm[0])
                    } else {
                        // N/C is typically in the first dimension (F1)
                        gt_peak.position_ppm[0]
                    }
                };

                let shift = ChemicalShift::new(
                    atom.id,
                    &assignment.atom_name,
                    assignment.residue_seq_code,
                    &assignment.residue_name,
                    "A", // chain code
                    shift_value,
                    nucleus,
                );

                shift_list.add_shift(shift);
                shifts_added += 1;
            } else {
                tracing::debug!("No atom found for {}_{} in molecule",
                    assignment.residue_seq_code, assignment.atom_name);
                shifts_skipped += 1;
            }
        }
    }

    let list_id = shift_list.id.to_string();
    state.add_shift_list(shift_list);

    tracing::info!("Created shift list '{}' with {} shifts ({} skipped - atom not found)",
        list_name, shifts_added, shifts_skipped);

    Ok(list_id)
}

/// Response containing 2D spectrum data for frontend visualization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Spectrum2DDataResponse {
    /// Spectrum ID
    pub id: String,
    /// 2D intensity data as nested arrays [F1][F2]
    pub data: Vec<Vec<f64>>,
    /// PPM axis for F1 (indirect dimension)
    pub ppm_axis_f1: Vec<f64>,
    /// PPM axis for F2 (direct dimension)
    pub ppm_axis_f2: Vec<f64>,
    /// Experiment type (e.g., "HSQC", "TOCSY", "NOESY")
    pub experiment_type: String,
    /// Estimated noise floor for contour level calculation
    pub noise_floor: f64,
    /// Number of ground truth peaks
    pub num_peaks: usize,
}

/// Generate a demo 2D spectrum for visualization.
///
/// Creates a synthetic 2D spectrum (HSQC, CHSQC, TOCSY, or NOESY) from the active molecule.
/// Returns the spectrum data ready for contour plotting.
#[tauri::command]
pub async fn generate_demo_2d(
    experiment_type: String,
    state: State<'_, AppState>,
) -> Result<Spectrum2DDataResponse> {
    use crate::data::spectrum::Spectrum2D;

    // Get active molecule
    let mol = {
        let active_id = state.active_molecule.read();
        let active_id = active_id.ok_or_else(|| NmrError::Internal("No active molecule".into()))?;

        let molecules = state.molecules.read();
        molecules
            .get(&active_id)
            .cloned()
            .ok_or_else(|| NmrError::Internal("Molecule not found".into()))?
    };

    // Generate the 2D spectrum based on type
    let (spectrum, ground_truth): (Spectrum2D, GroundTruthPeaks) = match experiment_type.to_uppercase().as_str() {
        "HSQC" | "15N-HSQC" | "N-HSQC" => {
            let gen = SpectrumGenerator::new(SpectrumGeneratorParams::default_hsqc());
            gen.generate_2d_hsqc(&mol)
        }
        "CHSQC" | "13C-HSQC" | "C-HSQC" => {
            let gen = SpectrumGenerator::new(SpectrumGeneratorParams::default_chsqc());
            gen.generate_2d_chsqc(&mol)
        }
        "TOCSY" => {
            let gen = SpectrumGenerator::new(SpectrumGeneratorParams::default_tocsy());
            gen.generate_2d_tocsy(&mol)
        }
        "NOESY" => {
            let gen = SpectrumGenerator::new(SpectrumGeneratorParams::default_noesy());
            gen.generate_2d_noesy(&mol)
        }
        other => {
            return Err(NmrError::Internal(format!(
                "Unknown 2D experiment type: {}. Supported: HSQC, CHSQC, TOCSY, NOESY",
                other
            )));
        }
    };

    let spectrum_id = spectrum.metadata.id;
    let experiment_type_str = format!("{:?}", spectrum.metadata.experiment_type);

    // Debug: check spectrum data has values
    let max_val = spectrum.data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_val = spectrum.data.iter().cloned().fold(f64::INFINITY, f64::min);
    let nonzero_count = spectrum.data.iter().filter(|&&v| v.abs() > 1e-10).count();
    tracing::info!(
        "Spectrum data stats: max={:.6}, min={:.6}, nonzero_count={}/{}",
        max_val, min_val, nonzero_count, spectrum.data.len()
    );

    // Convert ndarray to nested Vec for JSON serialization
    let (n_f1, n_f2) = spectrum.data.dim();
    let mut data_vec: Vec<Vec<f64>> = Vec::with_capacity(n_f1);
    for i in 0..n_f1 {
        let row: Vec<f64> = spectrum.data.row(i).to_vec();
        data_vec.push(row);
    }

    // Calculate noise floor estimate (median of absolute values in corners)
    let corner_size = 20.min(n_f1 / 10).min(n_f2 / 10).max(5);
    let mut corner_values: Vec<f64> = Vec::new();
    for i in 0..corner_size {
        for j in 0..corner_size {
            corner_values.push(spectrum.data[[i, j]].abs());
            corner_values.push(spectrum.data[[n_f1 - 1 - i, j]].abs());
            corner_values.push(spectrum.data[[i, n_f2 - 1 - j]].abs());
            corner_values.push(spectrum.data[[n_f1 - 1 - i, n_f2 - 1 - j]].abs());
        }
    }
    corner_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let noise_floor = corner_values.get(corner_values.len() / 2).copied().unwrap_or(0.01);

    // Store the spectrum
    {
        let mut spectra = state.spectra_2d.write();
        spectra.insert(spectrum_id, spectrum.clone());
    }

    let num_peaks = ground_truth.len();
    tracing::info!(
        "Generated {} 2D spectrum with {} peaks, noise floor estimate: {:.4}",
        experiment_type_str,
        num_peaks,
        noise_floor
    );

    Ok(Spectrum2DDataResponse {
        id: spectrum_id.to_string(),
        data: data_vec,
        ppm_axis_f1: spectrum.ppm_axis_f1.to_vec(),
        ppm_axis_f2: spectrum.ppm_axis_f2.to_vec(),
        experiment_type: experiment_type_str,
        noise_floor,
        num_peaks,
    })
}

/// Get a previously generated 2D spectrum by ID.
#[tauri::command]
pub async fn get_spectrum_2d(
    spectrum_id: String,
    state: State<'_, AppState>,
) -> Result<Spectrum2DDataResponse> {
    let uuid = Uuid::parse_str(&spectrum_id)
        .map_err(|_| NmrError::Internal(format!("Invalid spectrum ID: {}", spectrum_id)))?;

    let spectra = state.spectra_2d.read();
    let spectrum = spectra
        .get(&uuid)
        .ok_or_else(|| NmrError::Internal(format!("Spectrum not found: {}", spectrum_id)))?;

    let (n_f1, n_f2) = spectrum.data.dim();
    let mut data_vec: Vec<Vec<f64>> = Vec::with_capacity(n_f1);
    for i in 0..n_f1 {
        data_vec.push(spectrum.data.row(i).to_vec());
    }

    // Calculate noise floor
    let corner_size = 20.min(n_f1 / 10).min(n_f2 / 10).max(5);
    let mut corner_values: Vec<f64> = Vec::new();
    for i in 0..corner_size {
        for j in 0..corner_size {
            corner_values.push(spectrum.data[[i, j]].abs());
        }
    }
    corner_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let noise_floor = corner_values.get(corner_values.len() / 2).copied().unwrap_or(0.01);

    Ok(Spectrum2DDataResponse {
        id: spectrum_id,
        data: data_vec,
        ppm_axis_f1: spectrum.ppm_axis_f1.to_vec(),
        ppm_axis_f2: spectrum.ppm_axis_f2.to_vec(),
        experiment_type: format!("{:?}", spectrum.metadata.experiment_type),
        noise_floor,
        num_peaks: 0, // Unknown for retrieved spectra
    })
}

/// Get expected peak positions for a molecule without generating spectrum data.
#[tauri::command]
pub async fn get_expected_peaks(
    experiment_type: String,
    state: State<'_, AppState>,
) -> Result<GroundTruthPeaks> {
    let mol = {
        let active_id = state.active_molecule.read();
        let active_id = active_id.ok_or_else(|| NmrError::Internal("No active molecule".into()))?;

        let molecules = state.molecules.read();
        molecules
            .get(&active_id)
            .cloned()
            .ok_or_else(|| NmrError::Internal("Molecule not found".into()))?
    };

    let gt = match experiment_type.to_uppercase().as_str() {
        "1H" | "PROTON" | "1D" => {
            let gen = SpectrumGenerator::new_1d();
            gen.generate_peak_list_1d(&mol)
        }
        "HSQC" => {
            let gen = SpectrumGenerator::new(SpectrumGeneratorParams::default_hsqc());
            gen.generate_peak_list_hsqc(&mol)
        }
        "CHSQC" | "13C-HSQC" | "C-HSQC" => {
            let gen = SpectrumGenerator::new(SpectrumGeneratorParams::default_chsqc());
            gen.generate_peak_list_chsqc(&mol)
        }
        "TOCSY" => {
            let gen = SpectrumGenerator::new(SpectrumGeneratorParams::default_tocsy());
            gen.generate_peak_list_tocsy(&mol)
        }
        "NOESY" => {
            let gen = SpectrumGenerator::new(SpectrumGeneratorParams::default_noesy());
            gen.generate_peak_list_noesy(&mol)
        }
        other => {
            return Err(NmrError::Internal(format!(
                "Unknown experiment type: {}",
                other
            )));
        }
    };

    Ok(gt)
}
