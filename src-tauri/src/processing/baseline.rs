//! Baseline correction algorithms for NMR spectra.

use ndarray::Array1;

/// Polynomial baseline correction.
///
/// Fits a polynomial to baseline regions and subtracts it.
///
/// # Arguments
/// * `data` - Spectrum data
/// * `degree` - Polynomial degree (typically 1-5)
/// * `baseline_regions` - Regions to use for fitting (start, end) as fractions 0-1
pub fn polynomial_baseline(
    data: &Array1<f64>,
    degree: usize,
    baseline_regions: &[(f64, f64)],
) -> Array1<f64> {
    let n = data.len();
    if n == 0 {
        return data.clone();
    }

    // Collect baseline points
    let mut x_points = Vec::new();
    let mut y_points = Vec::new();

    for &(start, end) in baseline_regions {
        let start_idx = (start * n as f64) as usize;
        let end_idx = ((end * n as f64) as usize).min(n);

        for i in start_idx..end_idx {
            x_points.push(i as f64 / n as f64);
            y_points.push(data[i]);
        }
    }

    if x_points.is_empty() {
        return data.clone();
    }

    // Fit polynomial using least squares
    let coeffs = fit_polynomial(&x_points, &y_points, degree);

    // Subtract baseline
    let mut corrected = data.clone();
    for i in 0..n {
        let x = i as f64 / n as f64;
        let baseline = evaluate_polynomial(&coeffs, x);
        corrected[i] -= baseline;
    }

    corrected
}

/// Fit a polynomial using least squares.
fn fit_polynomial(x: &[f64], y: &[f64], degree: usize) -> Vec<f64> {
    let n = x.len();
    let m = degree + 1;

    // Build Vandermonde matrix
    let mut v = vec![vec![0.0; m]; n];
    for i in 0..n {
        for j in 0..m {
            v[i][j] = x[i].powi(j as i32);
        }
    }

    // Build V^T * V
    let mut vtv = vec![vec![0.0; m]; m];
    for i in 0..m {
        for j in 0..m {
            for k in 0..n {
                vtv[i][j] += v[k][i] * v[k][j];
            }
        }
    }

    // Build V^T * y
    let mut vty = vec![0.0; m];
    for i in 0..m {
        for k in 0..n {
            vty[i] += v[k][i] * y[k];
        }
    }

    // Solve using Gaussian elimination
    solve_linear_system(&mut vtv, &mut vty);

    vty
}

/// Solve a linear system Ax = b using Gaussian elimination with partial pivoting.
fn solve_linear_system(a: &mut Vec<Vec<f64>>, b: &mut Vec<f64>) {
    let n = b.len();

    for i in 0..n {
        // Find pivot
        let mut max_row = i;
        for k in (i + 1)..n {
            if a[k][i].abs() > a[max_row][i].abs() {
                max_row = k;
            }
        }

        // Swap rows
        a.swap(i, max_row);
        b.swap(i, max_row);

        // Eliminate
        for k in (i + 1)..n {
            if a[i][i].abs() > 1e-10 {
                let factor = a[k][i] / a[i][i];
                for j in i..n {
                    a[k][j] -= factor * a[i][j];
                }
                b[k] -= factor * b[i];
            }
        }
    }

    // Back substitution
    for i in (0..n).rev() {
        for j in (i + 1)..n {
            b[i] -= a[i][j] * b[j];
        }
        if a[i][i].abs() > 1e-10 {
            b[i] /= a[i][i];
        }
    }
}

/// Evaluate a polynomial at a point.
fn evaluate_polynomial(coeffs: &[f64], x: f64) -> f64 {
    let mut result = 0.0;
    for (i, &c) in coeffs.iter().enumerate() {
        result += c * x.powi(i as i32);
    }
    result
}

/// Automatic baseline correction using iterative polynomial fitting.
///
/// This method iteratively fits a polynomial, removes points above the fit,
/// and refits until convergence.
pub fn auto_baseline(data: &Array1<f64>, degree: usize, iterations: usize) -> Array1<f64> {
    let n = data.len();
    if n == 0 {
        return data.clone();
    }

    let mut mask = vec![true; n];
    let mut baseline = Array1::zeros(n);

    for _ in 0..iterations {
        // Collect masked points
        let mut x_points = Vec::new();
        let mut y_points = Vec::new();

        for i in 0..n {
            if mask[i] {
                x_points.push(i as f64 / n as f64);
                y_points.push(data[i]);
            }
        }

        if x_points.len() < degree + 1 {
            break;
        }

        // Fit polynomial
        let coeffs = fit_polynomial(&x_points, &y_points, degree);

        // Evaluate baseline
        for i in 0..n {
            let x = i as f64 / n as f64;
            baseline[i] = evaluate_polynomial(&coeffs, x);
        }

        // Update mask - remove points significantly above baseline
        let residuals: Vec<f64> = (0..n).map(|i| data[i] - baseline[i]).collect();
        let std_dev = standard_deviation(&residuals);

        for i in 0..n {
            if residuals[i] > 2.0 * std_dev {
                mask[i] = false;
            }
        }
    }

    // Return corrected spectrum
    let mut corrected = data.clone();
    for i in 0..n {
        corrected[i] -= baseline[i];
    }

    corrected
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

/// Spline baseline correction using cubic splines.
pub fn spline_baseline(data: &Array1<f64>, num_knots: usize) -> Array1<f64> {
    let n = data.len();
    if n < 4 || num_knots < 2 {
        return data.clone();
    }

    // Select knot positions evenly spaced
    let mut knot_x = Vec::with_capacity(num_knots);
    let mut knot_y = Vec::with_capacity(num_knots);

    for i in 0..num_knots {
        let idx = (i * (n - 1)) / (num_knots - 1);
        knot_x.push(idx as f64);

        // Use minimum in local region as baseline estimate
        let start = idx.saturating_sub(n / (num_knots * 2));
        let end = (idx + n / (num_knots * 2)).min(n);
        let local_min = data.slice(ndarray::s![start..end]).iter().cloned().fold(f64::INFINITY, f64::min);
        knot_y.push(local_min);
    }

    // Interpolate using cubic splines (simplified - linear for now)
    let mut baseline = Array1::zeros(n);

    for i in 0..n {
        let x = i as f64;

        // Find surrounding knots
        let mut k = 0;
        while k < knot_x.len() - 1 && knot_x[k + 1] < x {
            k += 1;
        }

        if k >= knot_x.len() - 1 {
            baseline[i] = knot_y[knot_y.len() - 1];
        } else {
            // Linear interpolation
            let t = (x - knot_x[k]) / (knot_x[k + 1] - knot_x[k]);
            baseline[i] = knot_y[k] * (1.0 - t) + knot_y[k + 1] * t;
        }
    }

    // Subtract baseline
    let mut corrected = data.clone();
    for i in 0..n {
        corrected[i] -= baseline[i];
    }

    corrected
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_polynomial_fit() {
        // y = 2 + 3x
        let x = vec![0.0, 0.5, 1.0];
        let y = vec![2.0, 3.5, 5.0];

        let coeffs = fit_polynomial(&x, &y, 1);

        assert_relative_eq!(coeffs[0], 2.0, epsilon = 1e-6);
        assert_relative_eq!(coeffs[1], 3.0, epsilon = 1e-6);
    }

    #[test]
    fn test_baseline_correction() {
        // Create data with linear baseline
        let n = 100;
        let mut data = Array1::zeros(n);
        for i in 0..n {
            let x = i as f64 / n as f64;
            // Baseline: 5 + 10*x
            // Signal: Gaussian peak at center
            let baseline = 5.0 + 10.0 * x;
            let signal = 100.0 * (-(x - 0.5).powi(2) / 0.01).exp();
            data[i] = baseline + signal;
        }

        // Correct using edge regions
        let corrected = polynomial_baseline(&data, 1, &[(0.0, 0.1), (0.9, 1.0)]);

        // Edges should be close to zero
        let edge_mean: f64 = (corrected.slice(ndarray::s![0..10]).sum()
            + corrected.slice(ndarray::s![90..100]).sum())
            / 20.0;

        assert!(edge_mean.abs() < 1.0);
    }

    #[test]
    fn test_auto_baseline() {
        let n = 100;
        let mut data = Array1::zeros(n);

        // Linear baseline with peak
        for i in 0..n {
            let x = i as f64 / n as f64;
            let baseline = 2.0 * x;
            let peak = if (x - 0.5).abs() < 0.1 { 10.0 } else { 0.0 };
            data[i] = baseline + peak;
        }

        let corrected = auto_baseline(&data, 1, 5);

        // Baseline regions should be near zero
        let baseline_mean: f64 = (corrected.slice(ndarray::s![0..20]).sum()
            + corrected.slice(ndarray::s![80..100]).sum())
            / 40.0;

        assert!(baseline_mean.abs() < 0.5);
    }
}
