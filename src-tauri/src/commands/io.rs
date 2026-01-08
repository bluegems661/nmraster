//! Tauri commands for file format import.

use serde::{Deserialize, Serialize};
use tauri::State;

use crate::error::Result;
use crate::io::{self, BrukerImportOptions};
use crate::state::AppState;

/// Request for importing a Bruker spectrum.
#[derive(Debug, Deserialize)]
pub struct ImportBrukerRequest {
    /// Path to the Bruker experiment directory
    pub path: String,
    /// Import processed data (true) or FID (false)
    pub processed: bool,
    /// Processing number (default: 1)
    #[serde(default = "default_procno")]
    pub procno: u32,
    /// Optional name override
    pub name: Option<String>,
}

fn default_procno() -> u32 {
    1
}

/// Request for importing an NMRPipe spectrum.
#[derive(Debug, Deserialize)]
pub struct ImportNmrPipeRequest {
    /// Path to the NMRPipe file
    pub path: String,
    /// Optional name override
    pub name: Option<String>,
}

/// Result of a successful import.
#[derive(Debug, Serialize)]
pub struct ImportResult {
    /// UUID of the imported spectrum
    pub spectrum_id: String,
    /// Name of the spectrum
    pub name: String,
    /// Number of dimensions (1, 2, 3, or 4)
    pub dimensions: usize,
    /// Experiment type (e.g., "Proton1D", "HSQC")
    pub experiment_type: String,
    /// Number of points in each dimension
    pub num_points: Vec<usize>,
    /// Whether the spectrum is processed (vs FID)
    pub is_processed: bool,
    /// Source format ("Bruker" or "NMRPipe")
    pub source_format: String,
}

/// Import a 1D Bruker spectrum.
#[tauri::command]
pub async fn import_bruker_1d(
    request: ImportBrukerRequest,
    state: State<'_, AppState>,
) -> Result<ImportResult> {
    let options = BrukerImportOptions {
        path: request.path,
        processed: request.processed,
        procno: request.procno,
        name: request.name,
    };

    let spectrum = io::import_bruker_1d(&options)?;

    let result = ImportResult {
        spectrum_id: spectrum.metadata.id.to_string(),
        name: spectrum.metadata.name.clone(),
        dimensions: 1,
        experiment_type: format!("{:?}", spectrum.metadata.experiment_type),
        num_points: vec![spectrum.len()],
        is_processed: spectrum.is_processed,
        source_format: "Bruker".to_string(),
    };

    // Store in app state
    state.add_spectrum_1d(spectrum);

    tracing::info!("Imported Bruker 1D spectrum: {}", result.name);

    Ok(result)
}

/// Import a 2D Bruker spectrum.
#[tauri::command]
pub async fn import_bruker_2d(
    request: ImportBrukerRequest,
    state: State<'_, AppState>,
) -> Result<ImportResult> {
    let options = BrukerImportOptions {
        path: request.path,
        processed: request.processed,
        procno: request.procno,
        name: request.name,
    };

    let spectrum = io::import_bruker_2d(&options)?;
    let (n_f1, n_f2) = spectrum.shape();

    let result = ImportResult {
        spectrum_id: spectrum.metadata.id.to_string(),
        name: spectrum.metadata.name.clone(),
        dimensions: 2,
        experiment_type: format!("{:?}", spectrum.metadata.experiment_type),
        num_points: vec![n_f1, n_f2],
        is_processed: spectrum.is_processed,
        source_format: "Bruker".to_string(),
    };

    // Store in app state
    state.add_spectrum_2d(spectrum);

    tracing::info!("Imported Bruker 2D spectrum: {}", result.name);

    Ok(result)
}

/// Import a 1D NMRPipe spectrum.
#[tauri::command]
pub async fn import_nmrpipe_1d(
    request: ImportNmrPipeRequest,
    state: State<'_, AppState>,
) -> Result<ImportResult> {
    let spectrum = io::import_nmrpipe_1d(&request.path, request.name.as_deref())?;

    let result = ImportResult {
        spectrum_id: spectrum.metadata.id.to_string(),
        name: spectrum.metadata.name.clone(),
        dimensions: 1,
        experiment_type: format!("{:?}", spectrum.metadata.experiment_type),
        num_points: vec![spectrum.len()],
        is_processed: spectrum.is_processed,
        source_format: "NMRPipe".to_string(),
    };

    // Store in app state
    state.add_spectrum_1d(spectrum);

    tracing::info!("Imported NMRPipe 1D spectrum: {}", result.name);

    Ok(result)
}

/// Import a 2D NMRPipe spectrum.
#[tauri::command]
pub async fn import_nmrpipe_2d(
    request: ImportNmrPipeRequest,
    state: State<'_, AppState>,
) -> Result<ImportResult> {
    let spectrum = io::import_nmrpipe_2d(&request.path, request.name.as_deref())?;
    let (n_f1, n_f2) = spectrum.shape();

    let result = ImportResult {
        spectrum_id: spectrum.metadata.id.to_string(),
        name: spectrum.metadata.name.clone(),
        dimensions: 2,
        experiment_type: format!("{:?}", spectrum.metadata.experiment_type),
        num_points: vec![n_f1, n_f2],
        is_processed: spectrum.is_processed,
        source_format: "NMRPipe".to_string(),
    };

    // Store in app state
    state.add_spectrum_2d(spectrum);

    tracing::info!("Imported NMRPipe 2D spectrum: {}", result.name);

    Ok(result)
}

/// Auto-detect file format and import.
#[tauri::command]
pub async fn import_spectrum_auto(
    path: String,
    name: Option<String>,
    state: State<'_, AppState>,
) -> Result<ImportResult> {
    use std::path::Path;

    let path_obj = Path::new(&path);

    // Try to detect format
    // Bruker: directory with acqus file
    // NMRPipe: file with .ft1, .ft2, .fid extension

    if path_obj.is_dir() {
        // Check for Bruker acqus
        let acqus = path_obj.join("acqus");
        if acqus.exists() {
            // Detect if 1D or 2D from PARMODE
            let acqu2s = path_obj.join("acqu2s");
            let is_2d = acqu2s.exists();

            let request = ImportBrukerRequest {
                path,
                processed: true,
                procno: 1,
                name,
            };

            if is_2d {
                return import_bruker_2d(request, state).await;
            } else {
                return import_bruker_1d(request, state).await;
            }
        }
    } else if path_obj.is_file() {
        // Check file extension
        let ext = path_obj
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        let request = ImportNmrPipeRequest { path, name };

        match ext.as_str() {
            "ft1" | "fid" => return import_nmrpipe_1d(request, state).await,
            "ft2" => return import_nmrpipe_2d(request, state).await,
            _ => {
                // Try to detect from header
                // For now, default to 1D
                return import_nmrpipe_1d(request, state).await;
            }
        }
    }

    Err(crate::io::IoError::UnknownFormat { path }.into())
}
