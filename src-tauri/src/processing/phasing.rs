//! Phase correction for NMR spectra.

use ndarray::Array1;
use num_complex::Complex64;
use std::f64::consts::PI;

/// Apply zero-order and first-order phase correction to a spectrum.
///
/// # Arguments
/// * `data` - Complex spectrum data (modified in-place)
/// * `ph0` - Zero-order phase correction in degrees
/// * `ph1` - First-order phase correction in degrees
/// * `pivot` - Pivot point for first-order correction (0.0-1.0, fraction of spectrum)
pub fn apply_phase_correction(
    data: &mut Array1<Complex64>,
    ph0: f64,
    ph1: f64,
    pivot: f64,
) {
    let n = data.len();
    if n == 0 {
        return;
    }

    let ph0_rad = ph0 * PI / 180.0;
    let ph1_rad = ph1 * PI / 180.0;

    for (i, val) in data.iter_mut().enumerate() {
        // Calculate position relative to pivot (range -pivot to 1-pivot)
        let x = (i as f64) / (n as f64) - pivot;

        // Total phase at this point
        let phase = ph0_rad + ph1_rad * x;

        // Apply rotation
        let rotation = Complex64::from_polar(1.0, -phase);
        *val *= rotation;
    }
}

/// Apply phase correction and return only the real part.
pub fn phase_correct_real(
    data: &Array1<Complex64>,
    ph0: f64,
    ph1: f64,
    pivot: f64,
) -> Array1<f64> {
    let mut phased = data.clone();
    apply_phase_correction(&mut phased, ph0, ph1, pivot);
    phased.mapv(|c| c.re)
}

/// Automatic phase correction using entropy minimization.
///
/// This finds the ph0/ph1 values that minimize the entropy of the
/// absolute-value spectrum, which tends to give the flattest baseline.
pub fn auto_phase_entropy(
    data: &Array1<Complex64>,
    pivot: f64,
) -> (f64, f64) {
    // Grid search for approximate values
    let mut best_ph0 = 0.0;
    let mut best_ph1 = 0.0;
    let mut best_entropy = f64::INFINITY;

    // Coarse search
    for ph0 in (-180..=180).step_by(10) {
        for ph1 in (-180..=180).step_by(10) {
            let entropy = calculate_phase_entropy(data, ph0 as f64, ph1 as f64, pivot);
            if entropy < best_entropy {
                best_entropy = entropy;
                best_ph0 = ph0 as f64;
                best_ph1 = ph1 as f64;
            }
        }
    }

    // Fine search around best values
    for ph0_delta in -10..=10 {
        for ph1_delta in -10..=10 {
            let ph0 = best_ph0 + ph0_delta as f64;
            let ph1 = best_ph1 + ph1_delta as f64;
            let entropy = calculate_phase_entropy(data, ph0, ph1, pivot);
            if entropy < best_entropy {
                best_entropy = entropy;
                best_ph0 = ph0;
                best_ph1 = ph1;
            }
        }
    }

    (best_ph0, best_ph1)
}

/// Calculate entropy of the derivative of the phased spectrum.
fn calculate_phase_entropy(
    data: &Array1<Complex64>,
    ph0: f64,
    ph1: f64,
    pivot: f64,
) -> f64 {
    let phased = phase_correct_real(data, ph0, ph1, pivot);

    // Calculate first derivative
    let n = phased.len();
    if n < 2 {
        return 0.0;
    }

    let mut derivatives = Vec::with_capacity(n - 1);
    for i in 0..n - 1 {
        derivatives.push((phased[i + 1] - phased[i]).abs());
    }

    // Normalize derivatives
    let sum: f64 = derivatives.iter().sum();
    if sum == 0.0 {
        return 0.0;
    }

    // Calculate entropy
    let mut entropy = 0.0;
    for d in derivatives {
        let p = d / sum;
        if p > 1e-10 {
            entropy -= p * p.ln();
        }
    }

    entropy
}

/// Automatic phase correction using ACME algorithm (Automated Phase Correction
/// based on Minimization of Entropy).
///
/// This is a more sophisticated version that uses simplex optimization.
pub fn auto_phase_acme(
    data: &Array1<Complex64>,
    pivot: f64,
) -> (f64, f64) {
    // Start with entropy-based estimate
    let (init_ph0, init_ph1) = auto_phase_entropy(data, pivot);

    // Simplex refinement (simplified Nelder-Mead)
    let mut simplex = vec![
        (init_ph0, init_ph1),
        (init_ph0 + 5.0, init_ph1),
        (init_ph0, init_ph1 + 5.0),
    ];

    let eval = |ph0: f64, ph1: f64| calculate_phase_entropy(data, ph0, ph1, pivot);

    for _ in 0..50 {
        // Sort by objective
        simplex.sort_by(|a, b| {
            eval(a.0, a.1).partial_cmp(&eval(b.0, b.1)).unwrap()
        });

        let best = simplex[0];
        let worst = simplex[2];

        // Centroid of best two
        let centroid = (
            (simplex[0].0 + simplex[1].0) / 2.0,
            (simplex[0].1 + simplex[1].1) / 2.0,
        );

        // Reflection
        let reflected = (
            2.0 * centroid.0 - worst.0,
            2.0 * centroid.1 - worst.1,
        );

        if eval(reflected.0, reflected.1) < eval(best.0, best.1) {
            // Expansion
            let expanded = (
                3.0 * centroid.0 - 2.0 * worst.0,
                3.0 * centroid.1 - 2.0 * worst.1,
            );
            if eval(expanded.0, expanded.1) < eval(reflected.0, reflected.1) {
                simplex[2] = expanded;
            } else {
                simplex[2] = reflected;
            }
        } else if eval(reflected.0, reflected.1) < eval(simplex[1].0, simplex[1].1) {
            simplex[2] = reflected;
        } else {
            // Contraction
            let contracted = (
                (worst.0 + centroid.0) / 2.0,
                (worst.1 + centroid.1) / 2.0,
            );
            if eval(contracted.0, contracted.1) < eval(worst.0, worst.1) {
                simplex[2] = contracted;
            } else {
                // Shrink
                simplex[1] = (
                    (best.0 + simplex[1].0) / 2.0,
                    (best.1 + simplex[1].1) / 2.0,
                );
                simplex[2] = (
                    (best.0 + simplex[2].0) / 2.0,
                    (best.1 + simplex[2].1) / 2.0,
                );
            }
        }

        // Check convergence
        let range_ph0 = simplex.iter().map(|p| p.0).fold(0.0_f64, |a, b| a.max(b))
            - simplex.iter().map(|p| p.0).fold(f64::INFINITY, |a, b| a.min(b));
        let range_ph1 = simplex.iter().map(|p| p.1).fold(0.0_f64, |a, b| a.max(b))
            - simplex.iter().map(|p| p.1).fold(f64::INFINITY, |a, b| a.min(b));

        if range_ph0 < 0.1 && range_ph1 < 0.1 {
            break;
        }
    }

    simplex.sort_by(|a, b| {
        eval(a.0, a.1).partial_cmp(&eval(b.0, b.1)).unwrap()
    });

    simplex[0]
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_phase_correction_zero() {
        let data = Array1::from_vec(vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
        ]);

        let mut phased = data.clone();
        apply_phase_correction(&mut phased, 0.0, 0.0, 0.5);

        for (a, b) in data.iter().zip(phased.iter()) {
            assert_relative_eq!(a.re, b.re, epsilon = 1e-10);
            assert_relative_eq!(a.im, b.im, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_phase_correction_90() {
        let data = Array1::from_vec(vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 1.0),
        ]);

        let mut phased = data.clone();
        apply_phase_correction(&mut phased, 90.0, 0.0, 0.5);

        // 90 degree rotation: (1,0) -> (0,-1), (0,1) -> (1,0)
        assert_relative_eq!(phased[0].re, 0.0, epsilon = 1e-10);
        assert_relative_eq!(phased[0].im, -1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_auto_phase_simple() {
        // Create data with known phase error
        let n = 128;
        let true_ph0 = 30.0;

        let original: Array1<Complex64> = Array1::from_iter(
            (0..n).map(|i| {
                let x = i as f64 / n as f64;
                Complex64::new((x * 10.0).exp() * (-x * 5.0).exp(), 0.0)
            }),
        );

        // Apply known phase
        let mut data = original.clone();
        apply_phase_correction(&mut data, -true_ph0, 0.0, 0.5);

        // Recover phase
        let (ph0, _) = auto_phase_entropy(&data, 0.5);

        // Should be close to true_ph0
        assert!((ph0 - true_ph0).abs() < 15.0);
    }
}
