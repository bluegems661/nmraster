//! FFT processing for NMR spectra using rustfft.

use ndarray::{Array1, Array2, Axis};
use num_complex::Complex64;
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

use crate::error::{Result, SpectrumError};

/// Perform forward FFT on a 1D complex array in-place.
pub fn fft_1d(data: &mut Array1<Complex64>) {
    let n = data.len();
    if n == 0 {
        return;
    }

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);

    let buffer = data.as_slice_mut().unwrap();
    fft.process(buffer);

    // Apply fftshift (move zero frequency to center)
    fftshift_1d(data);
}

/// Perform inverse FFT on a 1D complex array in-place.
pub fn ifft_1d(data: &mut Array1<Complex64>) {
    let n = data.len();
    if n == 0 {
        return;
    }

    // Apply ifftshift first (undo the shift before inverse)
    ifftshift_1d(data);

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_inverse(n);

    let buffer = data.as_slice_mut().unwrap();
    fft.process(buffer);

    // Normalize
    let scale = 1.0 / n as f64;
    data.mapv_inplace(|x| x * scale);
}

/// Shift zero-frequency component to center of spectrum.
pub fn fftshift_1d(data: &mut Array1<Complex64>) {
    let n = data.len();
    let mid = n / 2;

    let mut temp = data.clone();
    for i in 0..n {
        let new_idx = (i + mid) % n;
        data[new_idx] = temp[i];
    }
}

/// Inverse of fftshift.
pub fn ifftshift_1d(data: &mut Array1<Complex64>) {
    let n = data.len();
    let mid = (n + 1) / 2;

    let mut temp = data.clone();
    for i in 0..n {
        let new_idx = (i + mid) % n;
        data[new_idx] = temp[i];
    }
}

/// Perform 2D FFT on a complex array.
pub fn fft_2d(data: &mut Array2<Complex64>) {
    let (n1, n2) = data.dim();
    if n1 == 0 || n2 == 0 {
        return;
    }

    let mut planner = FftPlanner::new();
    let fft_rows: Arc<dyn Fft<f64>> = planner.plan_fft_forward(n2);
    let fft_cols: Arc<dyn Fft<f64>> = planner.plan_fft_forward(n1);

    // FFT along rows (axis 1)
    for mut row in data.axis_iter_mut(Axis(0)) {
        let buffer = row.as_slice_mut().unwrap();
        fft_rows.process(buffer);
    }

    // FFT along columns (axis 0)
    let mut col_buffer = vec![Complex64::new(0.0, 0.0); n1];
    for j in 0..n2 {
        // Copy column to buffer
        for i in 0..n1 {
            col_buffer[i] = data[[i, j]];
        }

        fft_cols.process(&mut col_buffer);

        // Copy back
        for i in 0..n1 {
            data[[i, j]] = col_buffer[i];
        }
    }

    // Apply 2D fftshift
    fftshift_2d(data);
}

/// Shift zero-frequency component to center for 2D data.
pub fn fftshift_2d(data: &mut Array2<Complex64>) {
    let (n1, n2) = data.dim();
    let mid1 = n1 / 2;
    let mid2 = n2 / 2;

    let temp = data.clone();
    for i in 0..n1 {
        for j in 0..n2 {
            let new_i = (i + mid1) % n1;
            let new_j = (j + mid2) % n2;
            data[[new_i, new_j]] = temp[[i, j]];
        }
    }
}

/// Zero-fill an array to a target size (must be >= current size).
pub fn zero_fill_1d(data: &Array1<Complex64>, target_size: usize) -> Result<Array1<Complex64>> {
    let n = data.len();
    if target_size < n {
        return Err(SpectrumError::InvalidRange(format!(
            "Target size {} must be >= current size {}",
            target_size, n
        ))
        .into());
    }

    let mut result = Array1::zeros(target_size);
    for (i, &val) in data.iter().enumerate() {
        result[i] = val;
    }

    Ok(result)
}

/// Zero-fill to the next power of 2.
pub fn zero_fill_to_power_of_2(data: &Array1<Complex64>) -> Array1<Complex64> {
    let n = data.len();
    let target = n.next_power_of_two();

    if target == n {
        return data.clone();
    }

    let mut result = Array1::zeros(target);
    for (i, &val) in data.iter().enumerate() {
        result[i] = val;
    }

    result
}

/// Process a complete 1D FID to spectrum.
pub fn process_fid_1d(
    real: &Array1<f64>,
    imag: &Array1<f64>,
    zero_fill_factor: usize,
) -> Result<(Array1<f64>, Array1<f64>)> {
    if real.len() != imag.len() {
        return Err(SpectrumError::InvalidDimensions {
            expected: real.len(),
            actual: imag.len(),
        }
        .into());
    }

    // Convert to complex
    let mut complex: Array1<Complex64> = Array1::from_iter(
        real.iter()
            .zip(imag.iter())
            .map(|(&r, &i)| Complex64::new(r, i)),
    );

    // Zero-fill if requested
    if zero_fill_factor > 1 {
        let target_size = complex.len() * zero_fill_factor;
        complex = zero_fill_1d(&complex, target_size)?;
    }

    // Apply FFT
    fft_1d(&mut complex);

    // Extract real and imaginary parts
    let result_real = complex.mapv(|c| c.re);
    let result_imag = complex.mapv(|c| c.im);

    Ok((result_real, result_imag))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    #[test]
    fn test_fft_1d_delta() {
        // FFT of delta function should give constant
        let n = 64;
        let mut data = Array1::zeros(n);
        data[0] = Complex64::new(1.0, 0.0);

        fft_1d(&mut data);

        // All values should be approximately 1
        for val in data.iter() {
            assert_relative_eq!(val.norm(), 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_fft_ifft_roundtrip() {
        let n = 32;
        let original: Array1<Complex64> = Array1::from_iter(
            (0..n).map(|i| Complex64::new(i as f64, (i * 2) as f64)),
        );

        let mut data = original.clone();
        fft_1d(&mut data);
        ifft_1d(&mut data);

        for (a, b) in original.iter().zip(data.iter()) {
            assert_relative_eq!(a.re, b.re, epsilon = 1e-10);
            assert_relative_eq!(a.im, b.im, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_fft_1d_sine() {
        // FFT of sine wave should show peaks at the frequency
        let n = 128;
        let freq = 4; // 4 cycles in n points

        let data: Array1<Complex64> = Array1::from_iter((0..n).map(|i| {
            let t = 2.0 * PI * (i as f64) * (freq as f64) / (n as f64);
            Complex64::new(t.sin(), 0.0)
        }));

        let mut fft_data = data.clone();
        fft_1d(&mut fft_data);

        // Find peak magnitude
        let magnitudes: Vec<f64> = fft_data.iter().map(|c| c.norm()).collect();
        let max_mag = magnitudes.iter().cloned().fold(0.0, f64::max);

        // Peak should be significantly larger than noise
        assert!(max_mag > 10.0);
    }

    #[test]
    fn test_zero_fill() {
        let data = Array1::from_vec(vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
        ]);

        let filled = zero_fill_1d(&data, 8).unwrap();

        assert_eq!(filled.len(), 8);
        assert_eq!(filled[0], Complex64::new(1.0, 0.0));
        assert_eq!(filled[3], Complex64::new(4.0, 0.0));
        assert_eq!(filled[4], Complex64::new(0.0, 0.0));
        assert_eq!(filled[7], Complex64::new(0.0, 0.0));
    }

    #[test]
    fn test_zero_fill_power_of_2() {
        let data = Array1::from_vec(vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
        ]);

        let filled = zero_fill_to_power_of_2(&data);
        assert_eq!(filled.len(), 4);
    }

    #[test]
    fn test_process_fid() {
        let n = 64;
        let real = Array1::from_iter((0..n).map(|i| (i as f64 * 0.1).sin()));
        let imag = Array1::zeros(n);

        let result = process_fid_1d(&real, &imag, 2);
        assert!(result.is_ok());

        let (spec_real, spec_imag) = result.unwrap();
        assert_eq!(spec_real.len(), n * 2);
        assert_eq!(spec_imag.len(), n * 2);
    }
}
