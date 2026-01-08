//! Tauri commands for spectrum operations.

use ndarray::Array1;
use serde::{Deserialize, Serialize};
use tauri::State;
use uuid::Uuid;

use crate::data::{ExperimentType, NucleusType, Spectrum1D, SpectrumMetadata};
use crate::error::Result;
use crate::processing::process_fid_1d;
use crate::state::AppState;

/// Response for spectrum data queries.
#[derive(Debug, Serialize)]
pub struct SpectrumDataResponse {
    pub id: String,
    pub name: String,
    pub experiment_type: String,
    pub real: Vec<f64>,
    pub imag: Option<Vec<f64>>,
    pub ppm_axis: Vec<f64>,
    pub is_processed: bool,
}

/// Request for loading spectrum from arrays.
#[derive(Debug, Deserialize)]
pub struct LoadSpectrumRequest {
    pub name: String,
    pub experiment_type: String,
    pub real: Vec<f64>,
    pub imag: Option<Vec<f64>>,
    pub spectral_width_hz: f64,
    pub carrier_offset_ppm: f64,
    pub spectrometer_frequency_mhz: f64,
}

/// Load a 1D spectrum from array data.
#[tauri::command]
pub async fn load_spectrum_1d(
    request: LoadSpectrumRequest,
    state: State<'_, AppState>,
) -> Result<String> {
    let n = request.real.len();

    let metadata = SpectrumMetadata {
        id: Uuid::new_v4(),
        name: request.name,
        experiment_type: parse_experiment_type(&request.experiment_type),
        nucleus_types: vec![NucleusType::H1], // Default to proton
        spectral_width_hz: vec![request.spectral_width_hz],
        carrier_offset_ppm: vec![request.carrier_offset_ppm],
        spectrometer_frequency_mhz: request.spectrometer_frequency_mhz,
        original_points: vec![n],
        processed_points: vec![n],
    };

    let real = Array1::from_vec(request.real);
    let imag = request.imag.map(Array1::from_vec).unwrap_or_else(|| Array1::zeros(n));

    let spectrum = Spectrum1D::from_fid(real, imag, metadata);
    let id = state.add_spectrum_1d(spectrum);

    Ok(id.to_string())
}

/// Get spectrum data by ID.
#[tauri::command]
pub async fn get_spectrum_1d(
    id: String,
    state: State<'_, AppState>,
) -> Result<Option<SpectrumDataResponse>> {
    let uuid = Uuid::parse_str(&id).map_err(|e| crate::error::NmrError::Internal(e.to_string()))?;

    let spectrum = state.get_spectrum_1d(&uuid);

    Ok(spectrum.map(|s| SpectrumDataResponse {
        id: s.metadata.id.to_string(),
        name: s.metadata.name,
        experiment_type: format!("{:?}", s.metadata.experiment_type),
        real: s.real.to_vec(),
        imag: s.imag.map(|i| i.to_vec()),
        ppm_axis: s.ppm_axis.to_vec(),
        is_processed: s.is_processed,
    }))
}

/// Process a 1D FID spectrum (apply FFT).
#[tauri::command]
pub async fn process_spectrum_1d(
    id: String,
    zero_fill_factor: Option<usize>,
    state: State<'_, AppState>,
) -> Result<String> {
    let uuid = Uuid::parse_str(&id).map_err(|e| crate::error::NmrError::Internal(e.to_string()))?;

    let spectrum = state.get_spectrum_1d(&uuid)
        .ok_or_else(|| crate::error::NmrError::Internal("Spectrum not found".to_string()))?;

    if spectrum.is_processed {
        return Ok(id);
    }

    let imag = spectrum.imag.unwrap_or_else(|| Array1::zeros(spectrum.real.len()));
    let zf = zero_fill_factor.unwrap_or(1);

    let (processed_real, processed_imag) = process_fid_1d(&spectrum.real, &imag, zf)?;

    // Create new processed spectrum
    let mut new_metadata = spectrum.metadata.clone();
    new_metadata.id = Uuid::new_v4();
    new_metadata.processed_points = vec![processed_real.len()];

    // Recalculate PPM axis for new size
    let n = processed_real.len();
    let sw = new_metadata.spectral_width_hz[0];
    let offset = new_metadata.carrier_offset_ppm[0];
    let sf = new_metadata.spectrometer_frequency_mhz;
    let sw_ppm = sw / sf;

    let ppm_axis = Array1::linspace(offset + sw_ppm / 2.0, offset - sw_ppm / 2.0, n);

    let processed = Spectrum1D {
        metadata: new_metadata.clone(),
        real: processed_real,
        imag: Some(processed_imag),
        ppm_axis,
        is_processed: true,
    };

    let new_id = state.add_spectrum_1d(processed);
    Ok(new_id.to_string())
}

/// List all loaded spectra.
#[tauri::command]
pub async fn list_spectra(state: State<'_, AppState>) -> Result<Vec<SpectrumInfo>> {
    let spectra_1d = state.spectra_1d.read();

    let mut result: Vec<SpectrumInfo> = spectra_1d
        .values()
        .map(|s| SpectrumInfo {
            id: s.metadata.id.to_string(),
            name: s.metadata.name.clone(),
            dimensions: 1,
            experiment_type: format!("{:?}", s.metadata.experiment_type),
            num_points: s.real.len(),
            is_processed: s.is_processed,
        })
        .collect();

    let spectra_2d = state.spectra_2d.read();
    result.extend(spectra_2d.values().map(|s| SpectrumInfo {
        id: s.metadata.id.to_string(),
        name: s.metadata.name.clone(),
        dimensions: 2,
        experiment_type: format!("{:?}", s.metadata.experiment_type),
        num_points: s.data.len(),
        is_processed: s.is_processed,
    }));

    Ok(result)
}

/// Basic spectrum information.
#[derive(Debug, Serialize)]
pub struct SpectrumInfo {
    pub id: String,
    pub name: String,
    pub dimensions: usize,
    pub experiment_type: String,
    pub num_points: usize,
    pub is_processed: bool,
}

/// Get peaks for a spectrum.
#[tauri::command]
pub async fn get_spectrum_peaks(
    spectrum_id: String,
    state: State<'_, AppState>,
) -> Result<Vec<PeakInfo>> {
    let uuid = Uuid::parse_str(&spectrum_id)
        .map_err(|e| crate::error::NmrError::Internal(e.to_string()))?;

    let peaks = state.get_peaks(&uuid);

    Ok(peaks
        .into_iter()
        .map(|p| PeakInfo {
            id: p.id.to_string(),
            position_ppm: p.position_ppm,
            intensity: p.intensity,
            volume: p.volume,
            annotation: p.annotation,
        })
        .collect())
}

/// Basic peak information for frontend.
#[derive(Debug, Serialize)]
pub struct PeakInfo {
    pub id: String,
    pub position_ppm: Vec<f64>,
    pub intensity: f64,
    pub volume: Option<f64>,
    pub annotation: Option<String>,
}

fn parse_experiment_type(s: &str) -> ExperimentType {
    match s.to_uppercase().as_str() {
        "1H" | "PROTON" | "PROTON1D" => ExperimentType::Proton1D,
        "13C" | "CARBON" | "CARBON1D" => ExperimentType::Carbon1D,
        "HSQC" => ExperimentType::HSQC,
        "HMBC" => ExperimentType::HMBC,
        "COSY" => ExperimentType::COSY,
        "TOCSY" => ExperimentType::TOCSY,
        "NOESY" => ExperimentType::NOESY,
        "ROESY" => ExperimentType::ROESY,
        "HNCO" => ExperimentType::HNCO,
        "HNCA" => ExperimentType::HNCA,
        _ => ExperimentType::Custom(s.to_string()),
    }
}
