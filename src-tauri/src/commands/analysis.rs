//! Tauri commands for spectral analysis (peak picking, integration).

use ndarray::Array1;
use serde::{Deserialize, Serialize};
use tauri::State;
use uuid::Uuid;

use crate::data::Peak;
use crate::error::Result;
use crate::state::AppState;

/// Peak picking parameters.
#[derive(Debug, Deserialize)]
pub struct PeakPickingParams {
    /// Minimum signal-to-noise ratio
    pub min_snr: f64,
    /// Minimum peak intensity (absolute)
    pub min_intensity: Option<f64>,
    /// Noise region for SNR calculation (start, end) as PPM
    pub noise_region: Option<(f64, f64)>,
}

/// Pick peaks in a 1D spectrum.
#[tauri::command]
pub async fn pick_peaks_1d(
    spectrum_id: String,
    params: PeakPickingParams,
    state: State<'_, AppState>,
) -> Result<Vec<PickedPeak>> {
    let uuid = Uuid::parse_str(&spectrum_id)
        .map_err(|e| crate::error::NmrError::Internal(e.to_string()))?;

    let spectrum = state
        .get_spectrum_1d(&uuid)
        .ok_or_else(|| crate::error::NmrError::Internal("Spectrum not found".to_string()))?;

    // Estimate noise level
    let noise = estimate_noise(&spectrum.real, &spectrum.ppm_axis, params.noise_region);

    // Find peaks using local maxima detection
    let peaks = find_local_maxima(&spectrum.real, &spectrum.ppm_axis, noise, params.min_snr);

    // Filter by minimum intensity if specified
    let mut filtered_peaks: Vec<Peak> = peaks
        .into_iter()
        .filter(|p| params.min_intensity.map_or(true, |min| p.intensity >= min))
        .map(|p| {
            let mut peak = Peak::new(uuid, vec![p.ppm], p.intensity);
            peak.signal_noise_ratio = p.snr;
            peak
        })
        .collect();

    // Store peaks in state
    let peak_infos: Vec<PickedPeak> = filtered_peaks
        .iter()
        .map(|p| PickedPeak {
            id: p.id.to_string(),
            ppm: p.position_ppm[0],
            intensity: p.intensity,
            snr: p.signal_noise_ratio,
        })
        .collect();

    state.add_peaks(uuid, filtered_peaks);

    Ok(peak_infos)
}

/// Picked peak result.
#[derive(Debug, Serialize)]
pub struct PickedPeak {
    pub id: String,
    pub ppm: f64,
    pub intensity: f64,
    pub snr: f64,
}

/// Internal peak candidate.
struct PeakCandidate {
    index: usize,
    ppm: f64,
    intensity: f64,
    snr: f64,
}

/// Estimate noise level from spectrum.
fn estimate_noise(data: &Array1<f64>, ppm_axis: &Array1<f64>, noise_region: Option<(f64, f64)>) -> f64 {
    let n = data.len();

    // If noise region specified, use that
    if let Some((start_ppm, end_ppm)) = noise_region {
        let mut noise_points = Vec::new();

        for i in 0..n {
            let ppm = ppm_axis[i];
            if ppm >= start_ppm.min(end_ppm) && ppm <= start_ppm.max(end_ppm) {
                noise_points.push(data[i]);
            }
        }

        if !noise_points.is_empty() {
            return standard_deviation(&noise_points);
        }
    }

    // Otherwise use edges of spectrum (first and last 5%)
    let edge_size = n / 20;
    let mut edge_points: Vec<f64> = data.slice(ndarray::s![0..edge_size]).to_vec();
    edge_points.extend(data.slice(ndarray::s![(n - edge_size)..n]).iter());

    standard_deviation(&edge_points)
}

/// Find local maxima in spectrum data.
fn find_local_maxima(
    data: &Array1<f64>,
    ppm_axis: &Array1<f64>,
    noise: f64,
    min_snr: f64,
) -> Vec<PeakCandidate> {
    let n = data.len();
    if n < 3 {
        return Vec::new();
    }

    let threshold = noise * min_snr;
    let mut peaks = Vec::new();

    for i in 1..(n - 1) {
        // Local maximum condition
        if data[i] > data[i - 1] && data[i] > data[i + 1] && data[i] > threshold {
            let snr = if noise > 0.0 {
                data[i] / noise
            } else {
                f64::INFINITY
            };

            peaks.push(PeakCandidate {
                index: i,
                ppm: ppm_axis[i],
                intensity: data[i],
                snr,
            });
        }
    }

    peaks
}

/// Calculate standard deviation.
fn standard_deviation(data: &[f64]) -> f64 {
    let n = data.len();
    if n < 2 {
        return 0.0;
    }

    let mean: f64 = data.iter().sum::<f64>() / n as f64;
    let variance: f64 = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    variance.sqrt()
}

/// Integrate a peak.
#[tauri::command]
pub async fn integrate_peak(
    spectrum_id: String,
    center_ppm: f64,
    width_ppm: f64,
    state: State<'_, AppState>,
) -> Result<IntegrationResult> {
    let uuid = Uuid::parse_str(&spectrum_id)
        .map_err(|e| crate::error::NmrError::Internal(e.to_string()))?;

    let spectrum = state
        .get_spectrum_1d(&uuid)
        .ok_or_else(|| crate::error::NmrError::Internal("Spectrum not found".to_string()))?;

    let start_ppm = center_ppm - width_ppm / 2.0;
    let end_ppm = center_ppm + width_ppm / 2.0;

    let start_idx = spectrum.index_at_ppm(start_ppm.max(end_ppm));
    let end_idx = spectrum.index_at_ppm(start_ppm.min(end_ppm));

    let mut volume = 0.0;
    let mut max_intensity = f64::NEG_INFINITY;

    for i in start_idx..=end_idx {
        volume += spectrum.real[i];
        max_intensity = max_intensity.max(spectrum.real[i]);
    }

    // Approximate volume by multiplying by point spacing
    let n = spectrum.real.len();
    let sw_ppm = (spectrum.ppm_axis[0] - spectrum.ppm_axis[n - 1]).abs();
    let point_spacing = sw_ppm / n as f64;
    volume *= point_spacing;

    Ok(IntegrationResult {
        center_ppm,
        width_ppm,
        volume,
        max_intensity,
        num_points: end_idx - start_idx + 1,
    })
}

/// Integration result.
#[derive(Debug, Serialize)]
pub struct IntegrationResult {
    pub center_ppm: f64,
    pub width_ppm: f64,
    pub volume: f64,
    pub max_intensity: f64,
    pub num_points: usize,
}

/// Clear all peaks for a spectrum.
#[tauri::command]
pub async fn clear_spectrum_peaks(
    spectrum_id: String,
    state: State<'_, AppState>,
) -> Result<()> {
    let uuid = Uuid::parse_str(&spectrum_id)
        .map_err(|e| crate::error::NmrError::Internal(e.to_string()))?;

    state.clear_peaks(&uuid);

    Ok(())
}
