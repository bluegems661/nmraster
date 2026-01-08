//! Apodization (window) functions for NMR FID processing.

use ndarray::Array1;
use std::f64::consts::PI;

/// Apply exponential line broadening to FID data.
///
/// # Arguments
/// * `data` - FID data (modified in-place)
/// * `lb_hz` - Line broadening in Hz
/// * `sw_hz` - Spectral width in Hz
pub fn exponential_multiply(data: &mut Array1<f64>, lb_hz: f64, sw_hz: f64) {
    let n = data.len();
    if n == 0 || sw_hz == 0.0 {
        return;
    }

    let dwell_time = 1.0 / sw_hz;

    for (i, val) in data.iter_mut().enumerate() {
        let t = i as f64 * dwell_time;
        let decay = (-PI * lb_hz * t).exp();
        *val *= decay;
    }
}

/// Apply Gaussian multiplication to FID data.
///
/// # Arguments
/// * `data` - FID data (modified in-place)
/// * `gb_hz` - Gaussian broadening in Hz
/// * `sw_hz` - Spectral width in Hz
pub fn gaussian_multiply(data: &mut Array1<f64>, gb_hz: f64, sw_hz: f64) {
    let n = data.len();
    if n == 0 || sw_hz == 0.0 {
        return;
    }

    let dwell_time = 1.0 / sw_hz;
    let sigma = 1.0 / (2.0 * PI * gb_hz);

    for (i, val) in data.iter_mut().enumerate() {
        let t = i as f64 * dwell_time;
        let decay = (-t * t / (2.0 * sigma * sigma)).exp();
        *val *= decay;
    }
}

/// Apply combined Lorentz-to-Gauss transformation.
///
/// This applies negative exponential (resolution enhancement) followed by
/// Gaussian (sensitivity), useful for converting Lorentzian lines to Gaussian.
///
/// # Arguments
/// * `data` - FID data (modified in-place)
/// * `lb_hz` - Negative exponential (typically negative for resolution enhancement)
/// * `gb_fraction` - Gaussian parameter (0-1, fraction of acquisition time)
/// * `sw_hz` - Spectral width in Hz
pub fn lorentz_to_gauss(data: &mut Array1<f64>, lb_hz: f64, gb_fraction: f64, sw_hz: f64) {
    let n = data.len();
    if n == 0 || sw_hz == 0.0 {
        return;
    }

    let dwell_time = 1.0 / sw_hz;
    let aq_time = n as f64 * dwell_time;

    for (i, val) in data.iter_mut().enumerate() {
        let t = i as f64 * dwell_time;

        // Exponential part
        let exp_decay = (-PI * lb_hz * t).exp();

        // Gaussian part
        let gb = gb_fraction * aq_time;
        let gauss_decay = if gb > 0.0 {
            (-(t * t) / (2.0 * gb * gb)).exp()
        } else {
            1.0
        };

        *val *= exp_decay * gauss_decay;
    }
}

/// Apply sine-bell window function.
///
/// # Arguments
/// * `data` - FID data (modified in-place)
/// * `shift` - Phase shift (0 = sine, 0.5 = cosine, etc.)
pub fn sine_bell(data: &mut Array1<f64>, shift: f64) {
    let n = data.len();
    if n == 0 {
        return;
    }

    for (i, val) in data.iter_mut().enumerate() {
        let x = (i as f64 + shift) / n as f64;
        let window = (PI * x).sin();
        *val *= window;
    }
}

/// Apply squared sine-bell window function.
///
/// More aggressive than sine-bell, commonly used for NOESY.
pub fn sine_bell_squared(data: &mut Array1<f64>, shift: f64) {
    let n = data.len();
    if n == 0 {
        return;
    }

    for (i, val) in data.iter_mut().enumerate() {
        let x = (i as f64 + shift) / n as f64;
        let window = (PI * x).sin().powi(2);
        *val *= window;
    }
}

/// Apply cosine-bell window function.
///
/// Equivalent to sine-bell with 0.5 shift.
pub fn cosine_bell(data: &mut Array1<f64>) {
    sine_bell(data, 0.5);
}

/// Apply Hanning window.
pub fn hanning(data: &mut Array1<f64>) {
    let n = data.len();
    if n == 0 {
        return;
    }

    for (i, val) in data.iter_mut().enumerate() {
        let x = i as f64 / (n - 1) as f64;
        let window = 0.5 * (1.0 - (2.0 * PI * x).cos());
        *val *= window;
    }
}

/// Apply Hamming window.
pub fn hamming(data: &mut Array1<f64>) {
    let n = data.len();
    if n == 0 {
        return;
    }

    for (i, val) in data.iter_mut().enumerate() {
        let x = i as f64 / (n - 1) as f64;
        let window = 0.54 - 0.46 * (2.0 * PI * x).cos();
        *val *= window;
    }
}

/// Apply Blackman window.
pub fn blackman(data: &mut Array1<f64>) {
    let n = data.len();
    if n == 0 {
        return;
    }

    for (i, val) in data.iter_mut().enumerate() {
        let x = i as f64 / (n - 1) as f64;
        let window = 0.42 - 0.5 * (2.0 * PI * x).cos() + 0.08 * (4.0 * PI * x).cos();
        *val *= window;
    }
}

/// Apply Kaiser window.
///
/// # Arguments
/// * `data` - FID data (modified in-place)
/// * `beta` - Shape parameter (0 = rectangular, higher = more tapered)
pub fn kaiser(data: &mut Array1<f64>, beta: f64) {
    let n = data.len();
    if n == 0 {
        return;
    }

    let i0_beta = bessel_i0(beta);

    for (i, val) in data.iter_mut().enumerate() {
        let x = 2.0 * i as f64 / (n - 1) as f64 - 1.0;
        let arg = beta * (1.0 - x * x).sqrt();
        let window = bessel_i0(arg) / i0_beta;
        *val *= window;
    }
}

/// Modified Bessel function of the first kind, order 0.
fn bessel_i0(x: f64) -> f64 {
    let ax = x.abs();

    if ax < 3.75 {
        let y = (x / 3.75).powi(2);
        1.0 + y
            * (3.5156229
                + y * (3.0899424
                    + y * (1.2067492
                        + y * (0.2659732 + y * (0.0360768 + y * 0.0045813)))))
    } else {
        let y = 3.75 / ax;
        (ax.exp() / ax.sqrt())
            * (0.39894228
                + y * (0.01328592
                    + y * (0.00225319
                        + y * (-0.00157565
                            + y * (0.00916281
                                + y * (-0.02057706
                                    + y * (0.02635537 + y * (-0.01647633 + y * 0.00392377))))))))
    }
}

/// Create a window function array without applying it.
pub fn create_window(window_type: WindowType, n: usize) -> Array1<f64> {
    let mut data = Array1::ones(n);

    match window_type {
        WindowType::None => {}
        WindowType::Exponential { lb_hz, sw_hz } => exponential_multiply(&mut data, lb_hz, sw_hz),
        WindowType::Gaussian { gb_hz, sw_hz } => gaussian_multiply(&mut data, gb_hz, sw_hz),
        WindowType::SineBell { shift } => sine_bell(&mut data, shift),
        WindowType::SineBellSquared { shift } => sine_bell_squared(&mut data, shift),
        WindowType::Hanning => hanning(&mut data),
        WindowType::Hamming => hamming(&mut data),
        WindowType::Blackman => blackman(&mut data),
        WindowType::Kaiser { beta } => kaiser(&mut data, beta),
    }

    data
}

/// Window function types.
#[derive(Debug, Clone, Copy)]
pub enum WindowType {
    None,
    Exponential { lb_hz: f64, sw_hz: f64 },
    Gaussian { gb_hz: f64, sw_hz: f64 },
    SineBell { shift: f64 },
    SineBellSquared { shift: f64 },
    Hanning,
    Hamming,
    Blackman,
    Kaiser { beta: f64 },
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_exponential_decay() {
        let mut data = Array1::ones(100);
        exponential_multiply(&mut data, 10.0, 1000.0);

        // First point should be 1.0
        assert_relative_eq!(data[0], 1.0, epsilon = 1e-10);

        // Should decay monotonically
        for i in 1..data.len() {
            assert!(data[i] < data[i - 1]);
        }
    }

    #[test]
    fn test_sine_bell_endpoints() {
        let mut data = Array1::ones(100);
        sine_bell(&mut data, 0.0);

        // First point should be close to 0 (sine of small angle)
        assert!(data[0] < 0.1);

        // Middle should be close to 1 (sine of PI/2)
        assert!(data[50] > 0.9);
    }

    #[test]
    fn test_cosine_bell_first_point() {
        let mut data = Array1::ones(100);
        cosine_bell(&mut data);

        // First point should be close to 1 (cosine of small angle)
        assert!(data[0] > 0.9);
    }

    #[test]
    fn test_hanning_symmetry() {
        let mut data = Array1::ones(101);
        hanning(&mut data);

        // Should be symmetric
        for i in 0..50 {
            assert_relative_eq!(data[i], data[100 - i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_kaiser_endpoints() {
        let mut data = Array1::ones(100);
        kaiser(&mut data, 5.0);

        // Endpoints should be smaller than center
        assert!(data[0] < data[50]);
        assert!(data[99] < data[50]);
    }

    #[test]
    fn test_create_window() {
        let window = create_window(WindowType::Hanning, 64);
        assert_eq!(window.len(), 64);

        // First and last should be zero (or close to it)
        assert!(window[0] < 0.01);
    }
}
