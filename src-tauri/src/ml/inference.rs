//! ML inference functions.

use ndarray::Array2;

use super::LoadedModel;
use crate::error::Result;

/// Run inference on a loaded model.
///
/// Note: This is a placeholder. Real implementation would use ort crate.
pub fn run_inference(
    _model: &LoadedModel,
    input: Array2<f32>,
) -> Result<Array2<f32>> {
    // Placeholder: return input as-is for now
    // Real implementation:
    // let input_tensor = TensorRef::try_from(&input)?;
    // let outputs = model.session.run(ort::inputs!["input" => input_tensor]?)?;
    // ...

    Ok(input)
}

/// Preprocess spectrum data for ML model input.
pub fn preprocess_spectrum_for_ml(
    spectrum: &[f64],
    target_size: usize,
) -> Array2<f32> {
    let n = spectrum.len();

    // Resample to target size if needed
    let resampled: Vec<f32> = if n == target_size {
        spectrum.iter().map(|&x| x as f32).collect()
    } else {
        let ratio = n as f64 / target_size as f64;
        (0..target_size)
            .map(|i| {
                let src_idx = (i as f64 * ratio) as usize;
                spectrum[src_idx.min(n - 1)] as f32
            })
            .collect()
    };

    // Normalize
    let min_val = resampled.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = resampled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let range = max_val - min_val;

    let normalized: Vec<f32> = if range > 0.0 {
        resampled.iter().map(|&x| (x - min_val) / range).collect()
    } else {
        resampled
    };

    // Reshape to (1, target_size) for batch processing
    Array2::from_shape_vec((1, target_size), normalized).unwrap()
}

/// Postprocess ML output to peak positions.
pub fn postprocess_peak_predictions(
    predictions: &Array2<f32>,
    threshold: f32,
    ppm_range: (f64, f64),
) -> Vec<(f64, f64)> {
    let (n_samples, n_points) = predictions.dim();
    let mut peaks = Vec::new();

    for i in 0..n_samples {
        for j in 0..n_points {
            if predictions[[i, j]] > threshold {
                let ppm = ppm_range.0 + (j as f64 / n_points as f64) * (ppm_range.1 - ppm_range.0);
                let confidence = predictions[[i, j]] as f64;
                peaks.push((ppm, confidence));
            }
        }
    }

    peaks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preprocess_spectrum() {
        let spectrum: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let processed = preprocess_spectrum_for_ml(&spectrum, 64);

        assert_eq!(processed.dim(), (1, 64));

        // Check normalization: min should be 0, max should be 1
        let min_val = processed.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = processed.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        assert!((min_val - 0.0).abs() < 1e-6);
        assert!((max_val - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_postprocess_peaks() {
        let predictions = Array2::from_shape_vec(
            (1, 10),
            vec![0.1, 0.2, 0.9, 0.3, 0.1, 0.8, 0.1, 0.1, 0.1, 0.1],
        )
        .unwrap();

        let peaks = postprocess_peak_predictions(&predictions, 0.5, (0.0, 10.0));

        assert_eq!(peaks.len(), 2);
        // Peaks at positions 2 and 5
        assert!((peaks[0].0 - 2.0).abs() < 0.1);
        assert!((peaks[1].0 - 5.0).abs() < 0.1);
    }
}
